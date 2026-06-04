# 🎙️🧠 Petruso Semantic Annotation Framework

> **Framework avanzato per l'annotazione semantica automatica di sedute psicoterapeutiche.**

Questo progetto è un framework completo progettato per addestrare modelli di **Machine Learning** in grado di classificare automaticamente gli interventi del terapeuta e del paziente in contesti clinici reali, offrendo anche una **Dashboard Interattiva** per l'esplorazione e la visualizzazione dei risultati.

Il sistema si compone di due macro-aree principali:

1.  🤖 **Training & NLP Pipelines (`Semantic_Annotation_v2`)**
    Utilizza modelli di *Sentence Transformers* (es. `BAAI/bge-large-en-v1.5`) per calcolare gli embedding delle frasi e addestra diversi classificatori (Gradient Boosting, Random Forest, SVM, ecc.) per prevedere le etichette semantiche e i codici comportamentali (es. standard MITI).
2.  📊 **Web Dashboard Interattiva (`Semantic_Annotation_WebApp`)**
    Un'interfaccia basata su *Streamlit* che permette di analizzare le sedute psicoterapeutiche processate, visualizzando le dinamiche del dialogo sotto forma di chat e calcolando metriche chiave tramite grafici dinamici.

-----

## 🛠️ Prerequisiti

Per eseguire le pipeline e la Web App è necessario disporre di:

* 🐍 **Python 3.8+**
* 🎧 **Dataset:** File `.csv` contenenti le trascrizioni e i metadati dei dialoghi clinici (es. `MI_Dataset.csv`, `AnnoMI.csv`).

### Installazione Dipendenze
Per installare tutte le librerie necessarie (sia per l'addestramento che per la Web App), esegui il seguente comando nella cartella principale del progetto:
```bash
pip install -r requirements.txt
```

-----

## 📂 Struttura del Progetto

Organizzare la root del progetto come segue:
```text
Tirocinio/
│
├── 📁 Semantic_Annotation_v2/       <-- Pipeline di addestramento e bilanciamento dati
│   ├── 📁 Data/                     <-- Dataset originali, sintetici e bilanciati
│   ├── 📁 Pipelines/                <-- Script ML (gb_pipeline.py, rf_pipeline.py, ecc.)
│   ├── 📁 Results/                  <-- Matrici di confusione (.png) e report di test (.txt)
│   └── 📁 Utility/                  <-- Script per preparazione dati (es. prepare_balanced_dataset.py)
│
├── 📁 Semantic_Annotation_WebApp/   <-- Dashboard interattiva
│   ├── 📁 Data/                     <-- Dataset con le predizioni finali (es. AnnoMI_Predicted.csv)
│   ├── 📁 Results/                  <-- Risultati del cross dataset (metriche e matrice di confusione)
│   ├── 📁 Scripts/                  <-- Script per inferenza e analisi (run_inference.py, cross_dataset_metrics.py, extract_topics.py)
│   └── 🐍 app.py                    <-- Entry point dell'interfaccia grafica (Streamlit)
│
└── 📄 README.md                     <-- Questa documentazione
```

-----

## 🏗️ Fase 1: Preparazione Dati e Addestramento

### 1\. Bilanciamento del Dataset
Per garantire prestazioni ottimali, i modelli vengono addestrati su un dataset bilanciato che unisce frasi reali a frasi sintetiche (generate tramite AI generativa).
Eseguire il file `Utility/prepare_balanced_dataset.py` per creare il file `MI_Dataset_Bilanciato.csv`.

### 2\. Addestramento dei Classificatori
Spostarsi nella cartella `Pipelines/` ed eseguire lo script del modello prescelto.
```bash
python gb_pipeline.py
```
> ⚠️ **IMPORTANTE:** Prima di avviare l'addestramento, apri lo script (es. `gb_pipeline.py`) e sostituisci i segnaposto `XXXXXXXX` con il nome dell'embedder che intendi utilizzare (es. `BAAI/bge-large-en-v1.5`, `all-mpnet-base-v2`, `all-MiniLM-L6-v2`).

**Modelli disponibili:**
*   `gb_pipeline.py` (Gradient Boosting)
*   `rf_pipeline.py` (Random Forest)
*   `svm_pipeline.py` (Support Vector Machine)
*   `nb_pipeline.py` (Gaussian Naive Bayes)
*   `mp_pipeline.py` (Multilayer Perceptron / Reti Neurali)
*   `lr_pipeline.py` (Logistic Regression)

-----

## 🚀 Fase 2: Inferenza e Web App

### 1\. Generazione delle Predizioni
Una volta scelto il modello migliore, utilizzare lo script di inferenza per processare nuove sedute (es. il dataset AnnoMI).
```bash
cd ../Semantic_Annotation_WebApp/Scripts
python run_inference.py
```
*Questo genererà il file `AnnoMI_Predicted.csv` contenente le predizioni del modello, che verrà salvato nella cartella `Data` della Web App.*

### 2\. Analisi Cross Dataset e Topic Extraction
Nella cartella `Scripts` sono presenti ulteriori strumenti di analisi:
*   `cross_dataset_metrics.py`: Calcola le metriche di valutazione incrociata (cross-dataset) e salva i risultati (file di testo e matrice di confusione) nella cartella `Results`.
*   `extract_topics.py`: Permette di estrarre e analizzare i topic principali dalle trascrizioni.

### 3\. Avvio della Dashboard Interattiva
Avviare l'interfaccia grafica per esplorare i risultati:
```bash
cd ../  # Torna nella cartella Semantic_Annotation_WebApp
streamlit run app.py
```
La dashboard permetterà di:
* Selezionare una specifica seduta dalla barra laterale.
* Leggere la **trascrizione completa** come una chat reale, con badge colorati per le annotazioni (Codici MITI e Talk Type).
* Consultare la **Dashboard Analitica** con grafici interattivi (torte e barre) sui comportamenti rilevati.

-----

## 📈 Metriche e Valutazione Modelli

Il framework valuta in maniera rigorosa l'affidabilità dei classificatori per prevenire il *data leakage*. Vengono applicate due validazioni incrociate:
1.  **Standard K-Fold** (k=10)
2.  **Stratified K-Fold** (k=10)

Il cuore del sistema risiede nel testing: mentre l'addestramento sfrutta un dataset ibrido (dati reali + sintetici), **il test viene effettuato ESCLUSIVAMENTE sui dati reali**. Le metriche calcolate (`classification_report`) e le relative **Matrici di Confusione**, che verranno salvate automaticamente in `Results/`, riflettono le vere capacità del modello in uno scenario puramente umano.

-----

### 🎓 Autore

**Progetto di Tirocinio - UniCa**
*Lorenzo Petruso*

-----