import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')


def main():
    print("=" * 60)
    print("AVVIO VALUTAZIONE CROSS-DATASET (MAPPING MITI -> ANNOMI)")
    print("=" * 60)

    # 1. Gestione percorsi
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    input_path = os.path.join(base_dir, 'Data', 'AnnoMI_Final.csv')

    # Nuova cartella Results nella root del progetto
    results_dir = os.path.join(base_dir, 'Results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists(input_path):
        print(f"ERRORE: Non trovo il file {input_path}")
        return

    # 2. Caricamento Dataset
    print("\n1/4: Caricamento Dataset e filtraggio battute terapeuta...")
    df = pd.read_csv(input_path)
    df_therapist = df[df['main_therapist_behaviour'].notna() & df['miti_prediction'].notna()]
    tot_iniziali = len(df_therapist)

    # 3. Mapping (Dizionario di traduzione aggiornato)
    print("2/4: Applicazione dell'Allineamento Semantico...")
    mapping_dict = {
        'question': 'question',
        'reflection': 'reflection',
        'give information': 'therapist_input',
        'advise': 'therapist_input',
        'directive': 'therapist_input',
        'other': 'other'
    }

    df_therapist['truth_mapped'] = df_therapist['main_therapist_behaviour'].str.lower()
    df_therapist['pred_mapped'] = df_therapist['miti_prediction'].str.lower()
    df_therapist['pred_mapped'] = df_therapist['pred_mapped'].map(mapping_dict)

    # Filtraggio Classi
    df_filtered = df_therapist.dropna(subset=['pred_mapped'])
    tot_filtrati = len(df_filtered)

    print(f"  -> Battute analizzate: {tot_filtrati} (Scartate {tot_iniziali - tot_filtrati} non in comune)")

    # 4. Calcolo metriche
    print("3/4: Calcolo delle metriche di classificazione...")
    y_true = df_filtered['truth_mapped']
    y_pred = df_filtered['pred_mapped']

    labels = sorted(y_true.unique())

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # 5. Salvataggio metriche
    print(f"4/4: Salvataggio dei risultati nella cartella {results_dir}...")

    report_text = (
        "============================================================\n"
        "RISULTATI VALUTAZIONE CROSS-DATASET (MAPPING MITI -> ANNOMI)\n"
        "============================================================\n\n"
        f"Totale battute valutate: {tot_filtrati}\n"
        f"Accuracy Globale Sulle Classi in Comune: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n"
        "REPORT DI CLASSIFICAZIONE DETTAGLIATO:\n"
        f"{report}\n"
    )

    txt_path = os.path.join(results_dir, 'cross_dataset_metrics.txt')
    with open(txt_path, 'w') as f:
        f.write(report_text)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Matrice di Confusione (Cross-Dataset)')
    plt.ylabel('Etichetta Reale (AnnoMI)')
    plt.xlabel('Etichetta Predetta (BGE-Large Mapped)')
    plt.tight_layout()

    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print("\n" + "=" * 60)
    print(f"FATTO! Risultati salvati con successo:")
    print(f"  - Report: {txt_path}")
    print(f"  - Matrice: {cm_path}")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()