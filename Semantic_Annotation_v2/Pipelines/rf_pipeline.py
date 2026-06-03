import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings('ignore')

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    print(f"Cartella base: {base_dir}")

    data_path = os.path.join(base_dir, 'Data', 'MI_Dataset_Bilanciato.csv')

    # Rinominare la cartella con il nome dell'embedder che si vuole utilizzare: BGELarge - MPNET- MiniL6
    output_dir = os.path.join(base_dir, 'Results', 'Results_RF', 'XXXXXXXX')
    os.makedirs(output_dir, exist_ok=True)

    #Rinominare il file con il nome dell'embedder che si vuole utilizzare: BGELarge - MPNET- MiniL6
    output_file_txt = os.path.join(output_dir, 'Risultati_RF_XXXXXXXX.txt')

    # 1. Caricamento dataset bilanciato
    print("\n1. Caricamento Dataset Bilanciato...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"ERRORE: Impossibile trovare il file in {data_path}")
        return

    # 2. Calcolo embedding
    print(f"\n2. Calcolo embedding (Senza Contesto)...")
    # Sostituire l'embedder che si vuole utilizzare: BAAI/bge-large-en-v1.5 - all-mpnet-base-v2 - all-MiniLM-L6-v2
    embedder = SentenceTransformer('XXXXXXXX')

    X = embedder.encode(df['text'].tolist(), show_progress_bar=True)
    y = df['final agreed label'].values
    is_synth = df['is_synthetic'].values

    # Estraiamo le etichette uniche in ordine alfabetico per i grafici
    unique_labels = sorted(list(set(y)))

    # Inizializzazione Modello Random Forest (n_jobs=-1 usa tutti i core)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Funzione custom k-fold (prevenzione data leakage e matrice di confusione)
    def custom_kfold_evaluation(splitter, nome_fase):
        print(f"\nInizio {nome_fase}...")
        y_true_all = []
        y_pred_all = []

        for fold, (train_idx, raw_test_idx) in enumerate(splitter.split(X, y)):
            # Filtro: Rimuove le frasi sintetiche dal test set
            test_idx = [i for i in raw_test_idx if not is_synth[i]]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Addestramento sul set misto (veri + sintetici)
            model.fit(X_train, y_train)

            # Predizione SOLO sui veri
            preds = model.predict(X_test)

            y_true_all.extend(y_test)
            y_pred_all.extend(preds)
            print(f" -> Fold {fold + 1}/10 completato")

        print(f"\n=== Report globale {nome_fase} ===")
        # AGGIORNATO: Tornati a 2 cifre decimali
        report = classification_report(y_true_all, y_pred_all, digits=2)
        print(report)

        # --- Creazione e salvataggio matrice di confusione ---
        cm = confusion_matrix(y_true_all, y_pred_all, labels=unique_labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=unique_labels, yticklabels=unique_labels)

        plt.title(f'Confusion Matrix - {nome_fase}\n(Solo Dati Reali)')
        plt.ylabel('True Label (Etichetta Reale)')
        plt.xlabel('Predicted Label (Predizione del Modello)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Nome file pulito (es. CM_STANDARD_K-FOLD.png)
        nome_file_img = f"CM_{nome_fase.replace(' ', '_')}.png"
        percorso_img = os.path.join(output_dir, nome_file_img)
        plt.savefig(percorso_img, dpi=300)
        plt.close()

        print(f" -> Matrice di Confusione salvata in: {nome_file_img}")

        return report, cm

    # 3. Esecuzione dei test
    print("\n1: Random Forest Standard K-Fold (k=10)")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    report_std, cm_std = custom_kfold_evaluation(kf, "STANDARD K-FOLD")

    print("\n2: Random Forest Stratified K-Fold (k=10)")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    report_strat, cm_strat = custom_kfold_evaluation(skf, "STRATIFIED K-FOLD")

    # 4. Salvataggio risultati testuali
    with open(output_file_txt, 'w') as f:
        f.write("=== TEST: Random Forest su Dataset Bilanciato (Max 1000/classe) ===\n")
        f.write("Note: Addestramento ibrido (Veri + Sintetici). Test SOLO sui dati reali.\n\n")
        f.write("=== STANDARD K-FOLD (k=10) ===\n")
        f.write(report_std)
        f.write("\n\n=== STRATIFIED K-FOLD (k=10) ===\n")
        f.write(report_strat)

    print(f"\nEsecuzione completata. Controlla la cartella {output_dir} per i risultati completi e le immagini.")

if __name__ == "__main__":
    main()