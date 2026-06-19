import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')


def main():
    print("=" * 60)
    print("AVVIO VALUTAZIONE CROSS-DATASET (FINE-GRAINED MAPPING)")
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

    # 3. Fine-Grained Mapping (Costruzione del Ground Truth Dettagliato)
    print("2/4: Applicazione dell'Allineamento Semantico Fine-Grained...")

    def map_ground_truth(row):
        main_label = str(row['main_therapist_behaviour']).lower()
        sub_label = str(row['therapist_input_subtype']).lower()

        # Mapping diretto per le classi semplici
        if main_label == 'question': return 'question'
        if main_label == 'reflection': return 'reflection'
        if main_label == 'other': return 'other'

        # Mapping della macro classe 'therapist_input' usando i sottotipi
        if main_label == 'therapist_input':
            if sub_label == 'information': return 'give information'
            if sub_label == 'advice': return 'advise'
            if sub_label in ['negotiation', 'options']: return 'directive'

        return 'unmapped'

    # Applichiamo la funzione per creare la vera etichetta umana dettagliata
    df_therapist['truth_mapped'] = df_therapist.apply(map_ground_truth, axis=1)
    df_therapist['pred_mapped'] = df_therapist['miti_prediction'].str.lower()

    # Le 6 classi perfette in comune da confrontare
    valid_classes = ['question', 'reflection', 'other', 'give information', 'advise', 'directive']

    # Filtraggio Classi
    df_filtered = df_therapist[
        df_therapist['truth_mapped'].isin(valid_classes) &
        df_therapist['pred_mapped'].isin(valid_classes)
        ]
    tot_filtrati = len(df_filtered)

    print(f"  -> Valutazioni analizzate: {tot_filtrati} (Scartate {tot_iniziali - tot_filtrati} fuori mapping)")

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
        "========================================================================\n"
        "RISULTATI VALUTAZIONE CROSS-DATASET (ANNOMI FULL - FINE GRAINED MAPPING)\n"
        "========================================================================\n\n"
        f"Totale valutazioni umane comparate: {tot_filtrati}\n"
        f"Accuracy Globale Sulle Classi in Comune: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n"
        "REPORT DI CLASSIFICAZIONE DETTAGLIATO (SUB-LABELS):\n"
        f"{report}\n"
    )

    txt_path = os.path.join(results_dir, 'cross_dataset_metrics.txt')
    with open(txt_path, 'w') as f:
        f.write(report_text)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.title('Matrice di Confusione (Cross-Dataset Fine-Grained)')
    plt.ylabel('Etichetta Reale Umana (AnnoMI Sub-Labels)')
    plt.xlabel('Etichetta Predetta dal Modello (MITI)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
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