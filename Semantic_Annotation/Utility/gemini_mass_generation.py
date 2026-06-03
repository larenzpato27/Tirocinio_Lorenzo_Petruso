import pandas as pd
import random
import time
import os
from google import genai

# 1. Configurazione
# Incolla qui la tua api key di google ai studio
API_KEY = "[ENCRYPTION_KEY]"
client = genai.Client(api_key=API_KEY)

TARGET_COUNT = 4856
OUTPUT_FILE = 'Data/MI_Dataset_Sintetico.csv'

def main():
    print("Caricamento Dataset originale...")
    df = pd.read_csv('Data/MI Dataset.csv')
    df = df[(df['author'] == 'listener') & (df['final agreed label'].notna())].copy()

    class_mapping = {
        'Simple Reflection': 'Reflection', 'Complex Reflection': 'Reflection',
        'Advise with Permission': 'Advise', 'Advise without Permission': 'Advise',
        'Support': 'Supportive/Affirming', 'Affirm': 'Supportive/Affirming',
        'Direct': 'Directive', 'Confront': 'Directive', 'Warn': 'Directive',
        'Closed Question': 'Question', 'Open Question': 'Question'
    }
    df['final agreed label'] = df['final agreed label'].replace(class_mapping)

    class_definitions = {
        'Emphasize Autonomy': "Affermazioni che evidenziano chiaramente la libertà di scelta e il controllo del paziente, ricordandogli che solo lui può decidere se e come cambiare.",
        'Advise': "Il terapeuta offre consigli, suggerimenti o soluzioni al paziente, con o senza il suo esplicito permesso.",
        'Supportive/Affirming': "Affermazioni che esprimono accordo, apprezzamento, comprensione o compassione verso il paziente.",
        'Reflection': "Il terapeuta riassume o riflette il significato, le emozioni o i pensieri di ciò che il paziente ha appena detto.",
        'Self-Disclose': "Il terapeuta condivide informazioni personali su di sé o sulle proprie reazioni.",
        'Question': "Il terapeuta pone domande aperte o chiuse per esplorare la situazione del paziente.",
        'Directive': "Il terapeuta dà ordini, avvertimenti o si confronta in modo diretto con il paziente."
    }

    # Creiamo/Carichiamo il file di output per non ripartire da zero se si interrompe
    if os.path.exists(OUTPUT_FILE):
        df_synthetic = pd.read_csv(OUTPUT_FILE)
        print(f"Trovato file sintetico esistente con {len(df_synthetic)} frasi già generate.")
    else:
        df_synthetic = pd.DataFrame(columns=['text', 'final agreed label', 'is_synthetic'])
        df_synthetic.to_csv(OUTPUT_FILE, index=False)

    # 2. Calcolo mancanti e generazione
    for target_class, definition in class_definitions.items():
        # Quante frasi VERE abbiamo?
        real_count = len(df[df['final agreed label'] == target_class])

        # Aggiorniamo il conteggio dei sintetici leggendo il file
        if os.path.exists(OUTPUT_FILE):
            df_synthetic_current = pd.read_csv(OUTPUT_FILE)
            synth_count = len(df_synthetic_current[df_synthetic_current[
                                                       'final agreed label'] == target_class]) if not df_synthetic_current.empty else 0
        else:
            synth_count = 0

        total_current = real_count + synth_count
        missing = TARGET_COUNT - total_current

        if missing <= 0:
            print(f"[{target_class}] Completata! ({total_current}/{TARGET_COUNT})")
            continue

        print(f"\n⏳ [{target_class}] Inizio generazione. Mancano {missing} frasi.")
        real_examples = df[df['final agreed label'] == target_class]['text'].tolist()

        # Inizia il ciclo finché non raggiungiamo il target
        while missing > 0:
            # Chiediamo 20 frasi alla volta (o meno se siamo quasi alla fine)
            batch_size = min(20, missing)

            few_shot_examples = random.sample(real_examples, min(5, len(real_examples)))
            examples_text = "\n".join([f"- {ex}" for ex in few_shot_examples])

            prompt = f"""
            Sei un esperto psicoterapeuta che utilizza il Colloquio Motivazionale (Motivational Interviewing).
            Genera {batch_size} nuove e distinte frasi dette dal terapeuta per la categoria: "{target_class}".

            Definizione: {definition}

            Esempi reali da cui trarre ispirazione per tono e lunghezza:
            {examples_text}

            Regole TASSATIVE:
            1. Scrivi in lingua INGLESE.
            2. Usa contesti diversi (fumo, studio, relazioni, lavoro, dieta, alcol).
            3. Restituisci SOLO le {batch_size} frasi. Una frase per riga. 
            4. NON usare numeri iniziali (es. "1.", "2."), NON usare trattini, NON mettere virgolette. Solo testo puro.
            """

            try:
                response = client.models.generate_content(
                    # Seleziona il modello di Gemini che vuoi utilizzare: es. gemini-2.5-flash, gemini-3.5-flash...
                    model='XXXXXXXX',
                    contents=prompt
                )

                # Pulizia del testo restituito
                lines = response.text.strip().split('\n')
                new_sentences = [line.strip().lstrip('-*1234567890. ').strip('"\'') for line in lines if line.strip()]

                if len(new_sentences) > 0:
                    new_data = pd.DataFrame({
                        'text': new_sentences,
                        'final agreed label': target_class,
                        'is_synthetic': True
                    })
                    new_data.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

                    missing -= len(new_sentences)
                    print(f" -> Generate {len(new_sentences)} frasi. Ne mancano ancora: {missing}")

                # Pausa strategica per rispettare il limite di 15 richieste/minuto di Google
                time.sleep(8)

            except Exception as e:
                print(f"Errore API: {e}. Faccio una pausa di 60 secondi...")
                time.sleep(60)

if __name__ == "__main__":
    main()