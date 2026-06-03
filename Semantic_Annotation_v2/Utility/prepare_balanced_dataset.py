import pandas as pd
import numpy as np
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    print(f"Cartella base: {base_dir}")

    # Costruiamo i percorsi assoluti ai file
    real_data_path = os.path.join(base_dir, 'Data', 'MI_Dataset.csv')
    synth_data_path = os.path.join(base_dir, 'Data', 'MI_Dataset_Sintetico.csv')
    output_path = os.path.join(base_dir, 'Data', 'MI_Dataset_Bilanciato.csv')

    # 1. Caricamento dataset reale
    print("\n1. Caricamento Dataset Reale...")
    df_real = pd.read_csv(real_data_path)

    # Filtriamo l'autore e i valori nulli
    df_real = df_real[(df_real['author'] == 'listener') & (df_real['final agreed label'].notna())].copy()

    # Eliminiamo la classe "-"
    df_real = df_real[df_real['final agreed label'] != '-']

    # Mappatura
    class_mapping = {
        'Simple Reflection': 'Reflection', 'Complex Reflection': 'Reflection',
        'Advise with Permission': 'Advise', 'Advise without Permission': 'Advise',
        'Support': 'Supportive/Affirming', 'Affirm': 'Supportive/Affirming',
        'Direct': 'Directive', 'Confront': 'Directive', 'Warn': 'Directive',
        'Closed Question': 'Question', 'Open Question': 'Question'
    }
    df_real['final agreed label'] = df_real['final agreed label'].replace(class_mapping)
    df_real['is_synthetic'] = False

    # Creiamo il contesto
    df_real['context_text'] = df_real['text'].shift(1).fillna("")
    df_real['context_speaker'] = df_real['author'].shift(1).fillna("")
    df_real['text_with_context'] = df_real.apply(
        lambda x: f"Patient said: {x['context_text']} | Therapist replied: {x['text']}"
        if x['context_speaker'] == 'client' else f"Therapist replied: {x['text']}",
        axis=1
    )

    # 2. Caricamento dataset sintetico
    print("2. Caricamento Dataset Sintetico (Gemini)...")
    df_synth = pd.read_csv(synth_data_path)
    df_synth['is_synthetic'] = True
    df_synth['text_with_context'] = "Therapist replied: " + df_synth['text']

    df_synth = df_synth.drop_duplicates(subset=['text'])

    # 3. Costruzione del mega-dataset bilanciato
    print("3. Costruzione del Mega-Dataset Bilanciato (Max 1000 per classe)...")
    target_count = 1000
    balanced_frames = []
    classes = df_real['final agreed label'].unique()

    for cls in classes:
        real_subset = df_real[df_real['final agreed label'] == cls]

        if len(real_subset) >= target_count:
            # Undersampling
            sampled = real_subset.sample(n=target_count, random_state=42)
            balanced_frames.append(sampled[['text', 'text_with_context', 'final agreed label', 'is_synthetic']])
            print(f"[{cls}] -> {target_count} frasi VERE (Undersampled)")
        else:
            # Oversampling
            synth_subset = df_synth[df_synth['final agreed label'] == cls]
            missing = target_count - len(real_subset)

            if len(synth_subset) >= missing:
                synth_sampled = synth_subset.sample(n=missing, random_state=42)
            else:
                synth_sampled = synth_subset

            combined = pd.concat([
                real_subset[['text', 'text_with_context', 'final agreed label', 'is_synthetic']],
                synth_sampled[['text', 'text_with_context', 'final agreed label', 'is_synthetic']]
            ])
            balanced_frames.append(combined)
            print(f"[{cls}] -> {len(real_subset)} VERE + {len(synth_sampled)} SINTETICHE = {len(combined)} Totali")

    # 4. Salvataggio
    final_df = pd.concat(balanced_frames).reset_index(drop=True)
    final_df.to_csv(output_path, index=False)
    print(f"\nFile salvato: '{output_path}' con {len(final_df)} righe totali.")

if __name__ == "__main__":
    main()