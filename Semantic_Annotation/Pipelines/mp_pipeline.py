import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignoriamo i noiosi avvisi di convergenza che le reti neurali spesso generano
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
    # 0. Preparazione cartella risultati
    # Rinominare la cartella con il nome dell'embedder che si vuole utilizzare: BGELarge - MPNET- MiniL6
    output_dir = 'Results_MP_XXXXXXXXXX'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Cartella '{output_dir}' creata con successo.")

    # 1. Caricamento dati e costruzione del contesto
    print("Caricamento del dataset grezzo...")
    df = pd.read_csv('Data/MI Dataset.csv')

    df['prev_text'] = df['text'].shift(1)
    df['prev_author'] = df['author'].shift(1)

    if 'dialogue_id' in df.columns:
        cond_contesto = (df['prev_author'] == 'speaker') & (df['dialogue_id'] == df['dialogue_id'].shift(1))
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
    print("\nApplicazione Class Reduction (8 Classi)...")
    class_mapping = {
        'Simple Reflection': 'Reflection', 'Complex Reflection': 'Reflection',
        'Advise with Permission': 'Advise', 'Advise without Permission': 'Advise',
        'Support': 'Supportive/Affirming', 'Affirm': 'Supportive/Affirming',
        'Direct': 'Directive', 'Confront': 'Directive', 'Warn': 'Directive',
        'Closed Question': 'Question', 'Open Question': 'Question'
    }
    df['final agreed label'] = df['final agreed label'].replace(class_mapping)

    # 3. Estrazione degli embedding
    print("\nCalcolo degli embedding...")
    # Selezionare l'embedder che si vuole utilizzare: BAAI/bge-large-en-v1.5 - all-mpnet-base-v2 - all-MiniLM-L6-v2
    embedder = SentenceTransformer('XXXXXXXXXX')

    print(" - Vettorizzazione SENZA contesto...")
    X_Senza = embedder.encode(df['text'].tolist(), show_progress_bar=True)
    print(" - Vettorizzazione CON contesto...")
    X_Con = embedder.encode(df['text_with_context'].tolist(), show_progress_bar=True)

    # Conversione delle etichette da testo a numeri interi
    y_str = df['final agreed label'].values
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    # 4. Preparazione modello mlp e validazioni
    # Lasciamo 100 neuroni e max_iter 300 per vedere come si comporta con input da 1024
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42, early_stopping=True)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Fase 1: standard 10-fold (senza contesto)
    print("\n" + "=" * 50)
    print("\n 1: MLP - Standard 10-Fold (senza contesto)...")
    y_pred_kf_senza_int = np.empty_like(y)
    fold = 1
    for train_index, test_index in kf.split(X_Senza, y):
        print(f"   Esecuzione Fold {fold}/10 (Standard)...")
        mlp_model.fit(X_Senza[train_index], y[train_index])
        y_pred_kf_senza_int[test_index] = mlp_model.predict(X_Senza[test_index])
        fold += 1

    y_pred_kf_senza = le.inverse_transform(y_pred_kf_senza_int)
    report_kf_senza = classification_report(y_str, y_pred_kf_senza)

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y_str, y_pred_kf_senza, labels=np.unique(y_str)), annot=True, fmt='d', cmap='Purples',
                xticklabels=np.unique(y_str), yticklabels=np.unique(y_str))
    plt.title('MLP - Standard 10-Fold (No Context)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_MLP_Standard_NoContext.png'), dpi=300)
    plt.close()

    # Fase 2: stratified 10-fold (senza contesto)
    print("\n" + "=" * 50)
    print("\n 2: MLP - Stratified 10-Fold (senza contesto)...")
    y_pred_skf_senza_int = np.empty_like(y)
    fold = 1
    for train_index, test_index in skf.split(X_Senza, y):
        print(f"   Esecuzione Fold {fold}/10 (Stratified)...")
        mlp_model.fit(X_Senza[train_index], y[train_index])
        y_pred_skf_senza_int[test_index] = mlp_model.predict(X_Senza[test_index])
        fold += 1

    y_pred_skf_senza = le.inverse_transform(y_pred_skf_senza_int)
    report_skf_senza = classification_report(y_str, y_pred_skf_senza)

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y_str, y_pred_skf_senza, labels=np.unique(y_str)), annot=True, fmt='d', cmap='Purples',
                xticklabels=np.unique(y_str), yticklabels=np.unique(y_str))
    plt.title('MLP - Stratified 10-Fold (No Context)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_MLP_Stratified_NoContext.png'), dpi=300)
    plt.close()

    # Fase 3: standard 10-fold (con contesto)
    print("\n" + "=" * 50)
    print("\n 3: MLP - Standard 10-Fold (con contesto)...")
    y_pred_kf_con_int = np.empty_like(y)
    fold = 1
    for train_index, test_index in kf.split(X_Con, y):
        print(f"   Esecuzione Fold {fold}/10 (Standard Context)...")
        mlp_model.fit(X_Con[train_index], y[train_index])
        y_pred_kf_con_int[test_index] = mlp_model.predict(X_Con[test_index])
        fold += 1

    y_pred_kf_con = le.inverse_transform(y_pred_kf_con_int)
    report_kf_con = classification_report(y_str, y_pred_kf_con)

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y_str, y_pred_kf_con, labels=np.unique(y_str)), annot=True, fmt='d', cmap='Purples',
                xticklabels=np.unique(y_str), yticklabels=np.unique(y_str))
    plt.title('MLP - Standard 10-Fold (With Context)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_MLP_Standard_Context.png'), dpi=300)
    plt.close()

    # Fase 4: stratified 10-fold (con contesto)
    print("\n" + "=" * 50)
    print("\n 4: MLP - Stratified 10-Fold (con contesto)...")
    y_pred_skf_con_int = np.empty_like(y)
    fold = 1
    for train_index, test_index in skf.split(X_Con, y):
        print(f"   Esecuzione Fold {fold}/10 (Stratified Context)...")
        mlp_model.fit(X_Con[train_index], y[train_index])
        y_pred_skf_con_int[test_index] = mlp_model.predict(X_Con[test_index])
        fold += 1

    y_pred_skf_con = le.inverse_transform(y_pred_skf_con_int)
    report_skf_con = classification_report(y_str, y_pred_skf_con)

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y_str, y_pred_skf_con, labels=np.unique(y_str)), annot=True, fmt='d', cmap='Purples',
                xticklabels=np.unique(y_str), yticklabels=np.unique(y_str))
    plt.title('MLP - Stratified 10-Fold (With Context)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_MLP_Stratified_Context.png'), dpi=300)
    plt.close()

    # 5. Salvataggio report
    # Sostituire il nome del file con l'embedder utilizzato: BGELarge - MPNET- MiniL6
    with open(os.path.join(output_dir, 'classification_reports_MLP_XXXXXXXXXX.txt'), 'w') as f:
        f.write("=== [FASE 1] MLP STANDARD K-FOLD (NO CONTEXT) ===\n")
        f.write(report_kf_senza)
        f.write("\n\n==================================================\n")
        f.write("=== [FASE 2] MLP STRATIFIED K-FOLD (NO CONTEXT) ===\n")
        f.write(report_skf_senza)
        f.write("\n\n==================================================\n")
        f.write("=== [FASE 3] MLP STANDARD K-FOLD (WITH CONTEXT) ===\n")
        f.write(report_kf_con)
        f.write("\n\n==================================================\n")
        f.write("=== [FASE 4] MLP STRATIFIED K-FOLD (WITH CONTEXT) ===\n")
        f.write(report_skf_con)

    print(f"\nTutti i 4 report testuali e le matrici sono stati salvati in: {output_dir}")

if __name__ == "__main__":
    main()