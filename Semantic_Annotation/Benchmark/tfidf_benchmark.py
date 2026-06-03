import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

def main():
    # --- Gestione percorsi ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    print(f"Cartella base: {base_dir}")

    # 0. Preparazione cartella
    output_dir = os.path.join(base_dir, 'Results', 'Results_TFIDF_All')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Cartella '{output_dir}' creata con successo.")

    # 1. Caricamento dati e contesto
    print("Caricamento del dataset grezzo...")
    data_path = os.path.join(base_dir, 'Data', 'MI Dataset.csv')
    df = pd.read_csv(data_path)

    df['prev_text'] = df['text'].shift(1)
    df['prev_author'] = df['author'].shift(1)

    # Controllo sicuro del nome della colonna (dialog_id)
    if 'dialogue_id' in df.columns:
        cond_contesto = (df['prev_author'] == 'speaker') & (df['dialogue_id'] == df['dialogue_id'].shift(1))
    elif 'dialog_id' in df.columns:
        cond_contesto = (df['prev_author'] == 'speaker') & (df['dialog_id'] == df['dialog_id'].shift(1))
    elif 'conversation_id' in df.columns:
        cond_contesto = (df['prev_author'] == 'speaker') & (df['conversation_id'] == df['conversation_id'].shift(1))
    else:
        cond_contesto = (df['prev_author'] == 'speaker')

    df['text_with_context'] = np.where(
        cond_contesto,
        "Paziente: " + df['prev_text'].astype(str) + " | Terapeuta: " + df['text'].astype(str),
        "Terapeuta: " + df['text'].astype(str)
    )

    df = df[(df['author'] == 'listener') &
            (df['final agreed label'].notna()) &
            (df['final agreed label'] != '-')].copy()

    # 2. Riduzione delle classi (macro-aree)
    print("Applicazione Class Reduction (8 Classi)...")
    class_mapping = {
        'Simple Reflection': 'Reflection', 'Complex Reflection': 'Reflection',
        'Advise with Permission': 'Advise', 'Advise without Permission': 'Advise',
        'Support': 'Supportive/Affirming', 'Affirm': 'Supportive/Affirming',
        'Direct': 'Directive', 'Confront': 'Directive', 'Warn': 'Directive',
        'Closed Question': 'Question', 'Open Question': 'Question'
    }
    df['final agreed label'] = df['final agreed label'].replace(class_mapping)

    # 3. Vettorizzazione tf-idf (al posto degli embedder)
    print("\nCalcolo delle matrici TF-IDF (Estrazione parole)...")
    # Usa max_features=5000 per prendere le 5000 parole/coppie più importanti
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))

    # Manteniamo la matrice sparsa per ottimizzare la RAM
    X_Senza = tfidf.fit_transform(df['text'])
    X_Con = tfidf.fit_transform(df['text_with_context'])

    y_str = df['final agreed label'].values
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    # 4. Configurazione dei 6 modelli
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(class_weight='balanced', random_state=42, dual=False),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        "Gradient Boosting (LGBM)": LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
        "Naive Bayes (Multinomial)": MultinomialNB(),  # Sostituito Gaussian con Multinomial (migliore per TF-IDF)
        "MLP (Rete Neurale)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42,
                                            early_stopping=True)
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    report_file = os.path.join(output_dir, 'TFIDF_Complete_Benchmark.txt')
    with open(report_file, 'w') as f:
        f.write("=== BENCHMARK COMPLETO TF-IDF (Tutti i Modelli, 4 Fasi) ===\n\n")

    # 5. Esecuzione del ciclo di validazione (240 addestramenti totali)
    for model_name, model in models.items():
        print(f"\n" + "=" * 50)
        print(f" AVVIO TEST: {model_name} (con TF-IDF)")
        print(f"=" * 50)

        # --- FASE 1: Standard No Context ---
        print(" - Fase 1/4: Standard 10-Fold (No Context)...")
        y_pred = np.empty_like(y)
        for train_index, test_index in kf.split(X_Senza, y):
            model.fit(X_Senza[train_index], y[train_index])
            y_pred[test_index] = model.predict(X_Senza[test_index])
        rep_1 = classification_report(y_str, le.inverse_transform(y_pred))

        # --- FASE 2: Stratified No Context ---
        print(" - Fase 2/4: Stratified 10-Fold (No Context)...")
        y_pred = np.empty_like(y)
        for train_index, test_index in skf.split(X_Senza, y):
            model.fit(X_Senza[train_index], y[train_index])
            y_pred[test_index] = model.predict(X_Senza[test_index])
        rep_2 = classification_report(y_str, le.inverse_transform(y_pred))

        # --- FASE 3: Standard With Context ---
        print(" - Fase 3/4: Standard 10-Fold (With Context)...")
        y_pred = np.empty_like(y)
        for train_index, test_index in kf.split(X_Con, y):
            model.fit(X_Con[train_index], y[train_index])
            y_pred[test_index] = model.predict(X_Con[test_index])
        rep_3 = classification_report(y_str, le.inverse_transform(y_pred))

        # --- FASE 4: Stratified With Context ---
        print(" - Fase 4/4: Stratified 10-Fold (With Context)...")
        y_pred = np.empty_like(y)
        for train_index, test_index in skf.split(X_Con, y):
            model.fit(X_Con[train_index], y[train_index])
            y_pred[test_index] = model.predict(X_Con[test_index])
        rep_4 = classification_report(y_str, le.inverse_transform(y_pred))

        # Scrittura su file per il modello corrente
        with open(report_file, 'a') as f:
            f.write(f"\n\n{'#' * 60}\n")
            f.write(f"### MODELLO: {model_name} ###\n")
            f.write(f"{'#' * 60}\n\n")

            f.write("=== [FASE 1] STANDARD K-FOLD (NO CONTEXT) ===\n")
            f.write(rep_1 + "\n")
            f.write("=== [FASE 2] STRATIFIED K-FOLD (NO CONTEXT) ===\n")
            f.write(rep_2 + "\n")
            f.write("=== [FASE 3] STANDARD K-FOLD (WITH CONTEXT) ===\n")
            f.write(rep_3 + "\n")
            f.write("=== [FASE 4] STRATIFIED K-FOLD (WITH CONTEXT) ===\n")
            f.write(rep_4 + "\n")

    print(f"\nTest TF-IDF conclusi! Il report completo è in: {report_file}")

if __name__ == "__main__":
    main()