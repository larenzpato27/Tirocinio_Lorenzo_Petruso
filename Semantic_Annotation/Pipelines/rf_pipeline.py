import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def main():
    # 0. Preparazione cartella risultati
    # Rinominare la cartella con il nome dell'embedder che si vuole utilizzare: BGELarge - MPNET- MiniL6
    output_dir = 'Results_RF_XXXXXXXXXX'
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

    y = df['final agreed label'].values

    # 4. Preparazione modello random forest e validazioni
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Fase 1: standard 10-fold (senza contesto)
    print("\n" + "=" * 50)
    print("\n 1: Random Forest - Standard 10-Fold (senza contesto)...")
    y_pred_kf_senza = np.empty_like(y)
    fold = 1
    for train_index, test_index in kf.split(X_Senza, y):
        print(f"   Esecuzione Fold {fold}/10 (Standard)...")
        rf_model.fit(X_Senza[train_index], y[train_index])
        y_pred_kf_senza[test_index] = rf_model.predict(X_Senza[test_index])
        fold += 1
    report_kf_senza = classification_report(y, y_pred_kf_senza)

    # Matrice di confusione
    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y, y_pred_kf_senza, labels=np.unique(y)), annot=True, fmt='d', cmap='Greens',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('RF - Standard 10-Fold (No Context)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_RF_Standard_NoContext.png'), dpi=300)
    plt.close()

    # Fase 2: stratified 10-fold (senza contesto)
    print("\n" + "=" * 50)
    print("\n 2: Random Forest - Stratified 10-Fold (senza contesto)...")
    y_pred_skf_senza = np.empty_like(y)
    fold = 1
    for train_index, test_index in skf.split(X_Senza, y):
        print(f"   Esecuzione Fold {fold}/10 (Stratified)...")
        rf_model.fit(X_Senza[train_index], y[train_index])
        y_pred_skf_senza[test_index] = rf_model.predict(X_Senza[test_index])
        fold += 1
    report_skf_senza = classification_report(y, y_pred_skf_senza)

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y, y_pred_skf_senza, labels=np.unique(y)), annot=True, fmt='d', cmap='Greens',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('RF - Stratified 10-Fold (No Context)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_RF_Stratified_NoContext.png'), dpi=300)
    plt.close()

    # Fase 3: standard 10-fold (con contesto)
    print("\n" + "=" * 50)
    print("\n 3: Random Forest - Standard 10-Fold (con contesto)...")
    y_pred_kf_con = np.empty_like(y)
    fold = 1
    for train_index, test_index in kf.split(X_Con, y):
        print(f"   Esecuzione Fold {fold}/10 (Standard Context)...")
        rf_model.fit(X_Con[train_index], y[train_index])
        y_pred_kf_con[test_index] = rf_model.predict(X_Con[test_index])
        fold += 1
    report_kf_con = classification_report(y, y_pred_kf_con)

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y, y_pred_kf_con, labels=np.unique(y)), annot=True, fmt='d', cmap='Greens',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('RF - Standard 10-Fold (With Context)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_RF_Standard_Context.png'), dpi=300)
    plt.close()

    # Fase 4: stratified 10-fold (con contesto)
    print("\n" + "=" * 50)
    print("\n 4: Random Forest - Stratified 10-Fold (con contesto)...")
    y_pred_skf_con = np.empty_like(y)
    fold = 1
    for train_index, test_index in skf.split(X_Con, y):
        print(f"   Esecuzione Fold {fold}/10 (Stratified Context)...")
        rf_model.fit(X_Con[train_index], y[train_index])
        y_pred_skf_con[test_index] = rf_model.predict(X_Con[test_index])
        fold += 1
    report_skf_con = classification_report(y, y_pred_skf_con)

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y, y_pred_skf_con, labels=np.unique(y)), annot=True, fmt='d', cmap='Greens',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('RF - Stratified 10-Fold (With Context)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_RF_Stratified_Context.png'), dpi=300)
    plt.close()

    # 5. Salvataggio report
    # Sostituire il nome del file con l'embedder utilizzato: BGELarge - MPNET- MiniL6
    with open(os.path.join(output_dir, 'classification_reports_RF_XXXXXXXXXX.txt'), 'w') as f:
        f.write("=== [FASE 1] RF STANDARD K-FOLD (NO CONTEXT) ===\n")
        f.write(report_kf_senza)
        f.write("\n\n==================================================\n")
        f.write("=== [FASE 2] RF STRATIFIED K-FOLD (NO CONTEXT) ===\n")
        f.write(report_skf_senza)
        f.write("\n\n==================================================\n")
        f.write("=== [FASE 3] RF STANDARD K-FOLD (WITH CONTEXT) ===\n")
        f.write(report_kf_con)
        f.write("\n\n==================================================\n")
        f.write("=== [FASE 4] RF STRATIFIED K-FOLD (WITH CONTEXT) ===\n")
        f.write(report_skf_con)

    print(f"\nTutti i 4 report testuali e le matrici sono stati salvati in: {output_dir}")

if __name__ == "__main__":
    main()