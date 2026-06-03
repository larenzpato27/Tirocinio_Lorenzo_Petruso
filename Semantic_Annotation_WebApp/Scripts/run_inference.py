import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("AVVIO INFERENZA: BGE-Large + GradientBoosting")
    print("=" * 60)

    # 1. Gestione percorsi
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, 'Data')

    train_path = os.path.join(data_dir, 'MI_Dataset_Bilanciato.csv')
    test_path = os.path.join(data_dir, 'AnnoMI.csv')
    output_path = os.path.join(data_dir, 'AnnoMI_Predicted.csv')

    if not os.path.exists(train_path):
        print(f"ERRORE: Non trovo il dataset di addestramento in {train_path}")
        return
    if not os.path.exists(test_path):
        print(f"ERRORE: Non trovo il dataset AnnoMI in {test_path}")
        return

    # 2. Caricamento e preparazione dati di training
    print("\n1/5: Caricamento dati di addestramento (MI_Dataset_Bilanciato)...")
    df_train = pd.read_csv(train_path)
    df_train = df_train[df_train['final agreed label'].notna()]

    texts_train = df_train['text'].tolist()
    y_train = df_train['final agreed label'].values

    # 3. Caricamento dati AnnoMI (da predirre)
    print("2/5: Caricamento nuovo dataset (AnnoMI)...")
    df_test = pd.read_csv(test_path)

    texts_test = df_test['utterance_text'].astype(str).tolist()

    # 4. Calcolo embeddings (BGE-Large)
    print("\n3/5: Calcolo Embedding con BAAI/bge-large-en-v1.5...")
    print("   (Potrebbe richiedere qualche minuto. Attendi il caricamento...)")
    embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')

    print("  - Embedding Training Set...")
    X_train = embedder.encode(texts_train, show_progress_bar=True)

    print("  - Embedding AnnoMI Set...")
    X_test = embedder.encode(texts_test, show_progress_bar=True)

    # 5. Addestramento e predizione
    print("\n⚙4/5: Addestramento Modello (GradientBoosting)...")
    model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    model.fit(X_train, y_train)

    print("5/5: Generazione Predizioni su AnnoMI...")
    predictions = model.predict(X_test)

    df_test['miti_prediction'] = predictions

    # 6. Salvataggio
    df_test.to_csv(output_path, index=False)
    print("\n" + "=" * 60)
    print(f"FATTO! Il file con le predizioni è stato salvato in:\n{output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()