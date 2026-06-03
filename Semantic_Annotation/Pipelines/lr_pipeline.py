import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

def main():
    # 0. Preparazione cartella risultati
    # Rinominare la cartella con il nome dell'embedder che si vuole utilizzare: BGELarge - MPNET- MiniL6
    output_dir = 'Results_LR_XXXXXXXXXX'
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

    y = df['final agreed label'].values

    # 4. Preparazione modello e validazioni
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # === fase 1: standard (no context) ===
    print("\nInizio fase 1: lr - standard 10-fold (senza contesto)...")
    y_pred_kf_senza = np.empty_like(y)
    for train_index, test_index in kf.split(X_Senza, y):
        lr_model.fit(X_Senza[train_index], y[train_index])
        y_pred_kf_senza[test_index] = lr_model.predict(X_Senza[test_index])
    report_kf_senza = classification_report(y, y_pred_kf_senza)

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y, y_pred_kf_senza, labels=np.unique(y)), annot=True, fmt='d', cmap='Oranges',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('LR - Standard 10-Fold (No Context)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_LR_Standard_NoContext.png'), dpi=300)
    plt.close()

    # === fase 2: stratified (no context) ===
    print("\nInizio fase 2: lr - stratified 10-fold (senza contesto)...")
    y_pred_skf_senza = np.empty_like(y)
    for train_index, test_index in skf.split(X_Senza, y):
        lr_model.fit(X_Senza[train_index], y[train_index])
        y_pred_skf_senza[test_index] = lr_model.predict(X_Senza[test_index])
    report_skf_senza = classification_report(y, y_pred_skf_senza)

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y, y_pred_skf_senza, labels=np.unique(y)), annot=True, fmt='d', cmap='Oranges',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('LR - Stratified 10-Fold (No Context)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_LR_Stratified_NoContext.png'), dpi=300)
    plt.close()

    # === fase 3: standard (with context) ===
    print("\nInizio fase 3: lr - standard 10-fold (con contesto)...")
    y_pred_kf_con = np.empty_like(y)
    for train_index, test_index in kf.split(X_Con, y):
        lr_model.fit(X_Con[train_index], y[train_index])
        y_pred_kf_con[test_index] = lr_model.predict(X_Con[test_index])
    report_kf_con = classification_report(y, y_pred_kf_con)

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y, y_pred_kf_con, labels=np.unique(y)), annot=True, fmt='d', cmap='Oranges',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('LR - Standard 10-Fold (With Context)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_LR_Standard_Context.png'), dpi=300)
    plt.close()

    # === fase 4: stratified (with context) ===
    print("\nInizio fase 4: lr - stratified 10-fold (con contesto)...")
    y_pred_skf_con = np.empty_like(y)
    for train_index, test_index in skf.split(X_Con, y):
        lr_model.fit(X_Con[train_index], y[train_index])
        y_pred_skf_con[test_index] = lr_model.predict(X_Con[test_index])
    report_skf_con = classification_report(y, y_pred_skf_con)

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y, y_pred_skf_con, labels=np.unique(y)), annot=True, fmt='d', cmap='Oranges',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('LR - Stratified 10-Fold (With Context)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_LR_Stratified_Context.png'), dpi=300)
    plt.close()

    # 5. Salvataggio report
    # Sostituire il nome del file con l'embedder utilizzato: BGELarge - MPNET- MiniL6
    with open(os.path.join(output_dir, 'classification_reports_LR_XXXXXXXXXX.txt'), 'w') as f:
        f.write("=== [FASE 1] LR STANDARD K-FOLD (NO CONTEXT) ===\n")
        f.write(report_kf_senza)
        f.write("\n\n==================================================\n")
        f.write("=== [FASE 2] LR STRATIFIED K-FOLD (NO CONTEXT) ===\n")
        f.write(report_skf_senza)
        f.write("\n\n==================================================\n")
        f.write("=== [FASE 3] LR STANDARD K-FOLD (WITH CONTEXT) ===\n")
        f.write(report_kf_con)
        f.write("\n\n==================================================\n")
        f.write("=== [FASE 4] LR STRATIFIED K-FOLD (WITH CONTEXT) ===\n")
        f.write(report_skf_con)

    print(f"\nTutti i 4 report testuali e le matrici sono stati salvati in: {output_dir}")

if __name__ == "__main__":
    main()