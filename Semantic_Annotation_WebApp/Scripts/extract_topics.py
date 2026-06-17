import pandas as pd
from google import genai
import os
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 1. Configurazione
# Incolla qui la tua api key di google ai studio
API_KEY = "[ENCRYPTION_KEY]"
client = genai.Client(api_key=API_KEY)


def main():
    print("=" * 60)
    print("AVVIO ESTRAZIONE KEY TOPICS TRAMITE GEMINI API")
    print("=" * 60)

    # 2. Gestione percorsi
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    input_path = os.path.join(base_dir, 'Data', 'AnnoMI_Predicted.csv')
    output_path = os.path.join(base_dir, 'Data', 'AnnoMI_Final.csv')

    if not os.path.exists(input_path):
        print(f"ERRORE: Non trovo il file originale in {input_path}")
        return

    # 3. Caricamento dataset e Checkpoint
    print("\n1/4: Controllo salvataggi precedenti...")

    # Se esiste già il file finale, partiamo da quello per non perdere il lavoro fatto
    if os.path.exists(output_path):
        print("  -> Trovato un salvataggio parziale! Riprendo da dove ci eravamo fermati.")
        df = pd.read_csv(output_path)
    else:
        print("  -> Nessun salvataggio trovato. Parto da zero.")
        df = pd.read_csv(input_path)
        # Creiamo la colonna vuota se non esiste
        if 'key_topics' not in df.columns:
            df['key_topics'] = None

    # Troviamo quali video mancano da elaborare (quelli con key_topics nullo o vuoto)
    video_titles = df['video_title'].unique()
    videos_to_process = [v for v in video_titles if pd.isna(df.loc[df['video_title'] == v, 'key_topics'].iloc[0])]

    if len(videos_to_process) == 0:
        print("\nTUTTI I VIDEO SONO GIA' STATI ELABORATI! Nessuna azione necessaria.")
        return

    print(f"\n2/4: Inizio elaborazione di {len(videos_to_process)} dialoghi rimanenti...")

    # 4. Ciclo di elaborazione
    for video in tqdm(videos_to_process, desc="Elaborazione Video"):

        # Estraiamo le battute del video
        dialogue_df = df[df['video_title'] == video]

        # Teniamo solo una riga per ogni utterance_id per creare un dialogo pulito
        unique_dialogue_df = dialogue_df.drop_duplicates(subset=['utterance_id']).sort_values(by='utterance_id')

        # Costruiamo il transcript pulito
        transcript = ""
        for _, row in unique_dialogue_df.iterrows():
            speaker = str(row['interlocutor']).upper()
            text = str(row['utterance_text'])
            transcript += f"{speaker}: {text}\n"

        prompt = f"""
        Sei un assistente esperto in psicologia clinica. Leggi la seguente trascrizione di una seduta psicologica (Intervista Motivazionale).
        Estrai dai 2 ai 5 "Key Topics" (i temi chiave o gli argomenti principali) espressi ESCLUSIVAMENTE dal cliente/paziente nel dialogo. Non includere concetti introdotti dal terapeuta.

        REGOLE FONDAMENTALI:
        1. Lingua: i topic devono essere scritti ESCLUSIVAMENTE in INGLESE.
        2. Focus sul cliente: basati solo ed esclusivamente sui problemi, pensieri o vissuti raccontati dal cliente.
        3. Semplicità: usa termini comuni e concetti basilari, evitando frasi complicate o troppo cliniche.
        4. Concisione: sii molto breve (massimo 3-4 parole per topic).
        5. Formato: restituisci SOLO un elenco puntato usando il simbolo '-', nessuna frase introduttiva o conclusiva.

        TRASCRIZIONE:
        {transcript}
        """

        try:
            response = client.models.generate_content(
                # Seleziona il modello di Gemini che vuoi utilizzare: es. gemini-2.5-flash, gemini-3.5-flash...
                model='XXXXXXXXXX',
                contents=prompt
            )
            extracted_topics = response.text.strip()

            # 5. Salvataggio Immediato (Checkpoint)
            # Applicando questo a df, copiamo i topics in TUTTE le righe dei vari annotatori per quel video
            df.loc[df['video_title'] == video, 'key_topics'] = extracted_topics

            # Salviamo su disco IMMEDIATAMENTE dopo ogni video
            df.to_csv(output_path, index=False)

        except Exception as e:
            print(f"\nErrore sul video {video}: {e}")
            # In caso di errore API, ci fermiamo così puoi riavviare pulito
            break

        # Pausa strategica per rispettare il limite di 15 richieste/minuto di Google
        time.sleep(8)

    print("\n" + "=" * 60)
    print(f"FATTO! Il file aggiornato è disponibile in:\n{output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()