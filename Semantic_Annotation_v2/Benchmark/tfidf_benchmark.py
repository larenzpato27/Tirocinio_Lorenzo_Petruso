import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")

def main():
    # --- Gestione percorsi dei percorsi ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    print(f"Cartella base: {base_dir}")

    # Costruiamo i percorsi di input e output
    data_path = os.path.join(base_dir, 'Data', 'MI_Dataset_Bilanciato.csv')

    # NUOVA CARTELLA: Results / Results_TFIDF
    output_dir = os.path.join(base_dir, 'Results', 'Results_TFIDF')
    os.makedirs(output_dir, exist_ok=True)

    output_file_txt = os.path.join(output_dir, 'TFIDF_Complete_Benchmark_Bilanciato.txt')

    # 1. Caricamento dataset bilanciato
    print("\n1. Caricamento Dataset Bilanciato...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"ERRORE: Impossibile trovare il file in {data_path}")
        return

    # Estraiamo le etichette, il testo e la flag "is_synthetic"
    y_str = df['final agreed label'].values
    texts = df['text'].tolist()
    is_synth = df['is_synthetic'].values

    # Estraiamo le etichette uniche in ordine alfabetico per i grafici
    unique_labels = sorted(list(set(y_str)))

    # 2. Vettorizzazione tf-idf (senza contesto)
    print("\n2. Calcolo matrice TF-IDF (Estrazione parole)...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))

    # MANTENIAMO LA MATRICE SPARSA (Tolto il distruttivo .toarray()!)
    X = tfidf.fit_transform(texts)
    y = y_str

    # 3. Configurazione dei 6 modelli (ottimizzati per matrici sparse)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(random_state=42, dual=False),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting (LGBM)": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        "Naive Bayes (Multinomial)": MultinomialNB(),
        "MLP (Rete Neurale)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Inizializziamo il file di report
    with open(output_file_txt, 'w') as f:
        f.write("=== BENCHMARK COMPLETO TF-IDF SU DATASET BILANCIATO ===\n")
        f.write("Note: Addestramento ibrido (Veri + Sintetici). Test SOLO sui dati reali.\n\n")

    # Funzione custom k-fold (anti-leakage)
    def run_evaluation(model, model_name, splitter, nome_fase):
        print(f" -> {nome_fase} in corso...")
        y_true_all = []
        y_pred_all = []

        for fold, (train_idx, raw_test_idx) in enumerate(splitter.split(X, y)):
            # Filtro: Solo reali nel test
            test_idx = [i for i in raw_test_idx if not is_synth[i]]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            y_true_all.extend(y_test)
            y_pred_all.extend(preds)

        report = classification_report(y_true_all, y_pred_all, digits=2)
        return report

    # 4. Esecuzione del ciclo di validazione
    print("\n" + "=" * 50)
    print(" INIZIO ADDESTRAMENTO MODELLI (TF-IDF)")
    print("=" * 50)

    for model_name, model in models.items():
        print(f"\n[{model_name}]")

        # Test 1: Standard
        rep_std = run_evaluation(model, model_name, kf, "Standard 10-Fold")

        # Test 2: Stratified
        rep_strat = run_evaluation(model, model_name, skf, "Stratified 10-Fold")

        # Scrittura su file
        with open(output_file_txt, 'a') as f:
            f.write(f"\n\n{'#' * 60}\n")
            f.write(f"### MODELLO: {model_name} ###\n")
            f.write(f"{'#' * 60}\n\n")

            f.write("=== STANDARD K-FOLD (k=10) ===\n")
            f.write(rep_std + "\n")
            f.write("=== STRATIFIED K-FOLD (k=10) ===\n")
            f.write(rep_strat + "\n")

    print(f"\nTutti i test TF-IDF conclusi! Il report completo è in:\n{output_file_txt}")

if __name__ == "__main__":
    main()