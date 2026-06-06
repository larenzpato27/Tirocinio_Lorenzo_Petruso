import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- 1. configurazione pagina ---
st.set_page_config(
    page_title="Semantic Annotation - MITI Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- 2. stili css ---
st.markdown("""
<style>
    .chat-row { display: flex; margin-bottom: 15px; width: 100%; }
    .row-client { justify-content: flex-start; }
    .row-therapist { justify-content: flex-end; }

    .bubble { 
        padding: 12px 18px; 
        border-radius: 15px; 
        max-width: 100%; 
        font-size: 1.05em;
        line-height: 1.4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .bubble-client { 
        background-color: #F1F3F5; 
        color: #212529; 
        border-bottom-left-radius: 0px; 
    }
    .bubble-therapist { 
        background-color: #E7F5FF; 
        color: #0B509E; 
        border-bottom-right-radius: 0px; 
    }

    /* Etichette Colorate Comuni */
    .miti-badge {
        font-size: 0.75em;
        padding: 4px 10px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        margin-bottom: 5px;
        display: inline-block;
        letter-spacing: 0.5px;
    }

    /* Colori Etichette Terapeuta (MITI) */
    .badge-question { background-color: #17A2B8; } 
    .badge-reflection { background-color: #007BFF; } 
    .badge-supportive { background-color: #28A745; } 
    .badge-directive { background-color: #DC3545; } 
    .badge-advise { background-color: #FD7E14; } 
    .badge-autonomy { background-color: #6F42C1; } 
    .badge-information { background-color: #6C757D; } 
    .badge-selfdisclose { background-color: #E83E8C; } 
    .badge-other { background-color: #343A40; } 

    /* Colori Etichette Paziente (Talk Type) */
    .badge-change { background-color: #20C997; } /* Verde Acqua per affermazioni positive */
    .badge-sustain { background-color: #FD7E14; } /* Arancione per le resistenze */
    .badge-neutral { background-color: #ADB5BD; } /* Grigio chiaro per frasi neutre */

    .speaker-name {
        font-size: 0.8em;
        font-weight: bold;
        color: #868e96;
        margin-bottom: 3px;
    }
</style>
""", unsafe_allow_html=True)


# Funzione colori Terapeuta
def get_badge_class(label):
    if pd.isna(label): return "badge-other"
    label = str(label).lower()
    if "question" in label: return "badge-question"
    if "reflection" in label: return "badge-reflection"
    if "supportive" in label or "affirm" in label: return "badge-supportive"
    if "direct" in label: return "badge-directive"
    if "advise" in label: return "badge-advise"
    if "autonomy" in label: return "badge-autonomy"
    if "information" in label: return "badge-information"
    if "disclose" in label: return "badge-selfdisclose"
    return "badge-other"


# Funzione colori Paziente
def get_client_badge_class(label):
    if pd.isna(label): return "badge-other"
    label = str(label).lower()
    if "change" in label: return "badge-change"
    if "sustain" in label: return "badge-sustain"
    if "neutral" in label: return "badge-neutral"
    return "badge-other"


# --- 3. Caricamento dati ---
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'Data', 'AnnoMI_Final.csv')
    df = pd.read_csv(file_path)

    # Teniamo una sola riga per ogni 'utterance_id' così la chat non fa l'eco.
    df = df.drop_duplicates(subset=['video_title', 'utterance_id'])

    return df


try:
    df = load_data()
except FileNotFoundError:
    st.error("Errore: Impossibile trovare 'AnnoMI_Final.csv' nella cartella 'Data'.")
    st.stop()

# --- 4. SIDEBAR (Selezione Dialogo) ---
st.sidebar.title("Navigazione Sedute")
video_titles = df['video_title'].unique()
selected_video = st.sidebar.selectbox("Seleziona una seduta da analizzare:", video_titles)

# Filtriamo i dati solo per il video selezionato
dialogue_df = df[df['video_title'] == selected_video].sort_values(by='utterance_id')

# Estraiamo le informazioni generali della seduta
topic = dialogue_df['topic'].iloc[0] if not pd.isna(dialogue_df['topic'].iloc[0]) else None
key_topics = dialogue_df['key_topics'].iloc[0] if 'key_topics' in dialogue_df.columns and not pd.isna(
    dialogue_df['key_topics'].iloc[0]) else None

st.sidebar.markdown("---")
st.sidebar.markdown("### Dettagli Seduta")

# Stampiamo i Key Topics di Gemini (se esistono)
if key_topics:
    st.sidebar.markdown("### ⭐ Key Topics (LLM Analysis)")
    st.sidebar.markdown(key_topics)
st.sidebar.markdown("---")

# --- 5. SCHERMATA PRINCIPALE ---
st.title(f"Seduta: {selected_video}")
st.markdown("Analisi semantica e comportamentale generata tramite **BGE-Large + Gradient Boosting**.")

tab1, tab2 = st.tabs(["💬 Trascrizione Seduta", "📊 Dashboard Analitica"])

# Tab 1: La Chat
with tab1:
    st.markdown("### Dialogo Paziente - Terapeuta")

    for _, row in dialogue_df.iterrows():
        speaker = row['interlocutor']
        text = row['utterance_text']

        if speaker == 'client':
            client_label = row['client_talk_type']

            if not pd.isna(client_label) and str(client_label).strip() != "":
                badge_css = get_client_badge_class(client_label)
                badge_html = f'<div class="miti-badge {badge_css}">{str(client_label).upper()}</div>'
            else:
                badge_html = ""

            st.markdown(f"""
            <div class="chat-row row-client">
                <div style="display:flex; flex-direction:column; align-items:flex-start; max-width:75%;">
                    {badge_html}
                    <div class="bubble bubble-client">
                        <div class="speaker-name" style="text-align: left;">CLIENT ({row['timestamp']})</div>
                        {text}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif speaker == 'therapist':
            miti_label = row['miti_prediction']
            badge_css = get_badge_class(miti_label)

            st.markdown(f"""
            <div class="chat-row row-therapist">
                <div style="display:flex; flex-direction:column; align-items:flex-end; max-width:75%;">
                    <div class="miti-badge {badge_css}">{miti_label.upper()}</div>
                    <div class="bubble bubble-therapist">
                        <div class="speaker-name" style="text-align: right;">THERAPIST ({row['timestamp']})</div>
                        {text}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Tab 2: La Dashboard
with tab2:
    st.markdown("### Metriche della Seduta")

    col1, col2, col3 = st.columns(3)
    total_utterances = len(dialogue_df)
    therapist_count = len(dialogue_df[dialogue_df['interlocutor'] == 'therapist'])
    client_count = len(dialogue_df[dialogue_df['interlocutor'] == 'client'])
    col1.metric("Totale Battute", total_utterances)
    col2.metric("Interventi Terapeuta", therapist_count)
    col3.metric("Interventi Paziente", client_count)

    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Comportamenti del Terapeuta (AI Predicted)")
        therapist_df = dialogue_df[dialogue_df['interlocutor'] == 'therapist']

        if not therapist_df.empty:
            pie_data = therapist_df['miti_prediction'].value_counts().reset_index()
            pie_data.columns = ['Etichetta MITI', 'Conteggio']

            fig_pie = px.pie(pie_data, values='Conteggio', names='Etichetta MITI',
                             hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')

            # Legenda in basso
            fig_pie.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Nessun dato del terapeuta in questa seduta.")

    with col_chart2:
        st.markdown("#### Tipo di Risposta del Paziente")
        client_df = dialogue_df[dialogue_df['interlocutor'] == 'client']

        if not client_df.empty and 'client_talk_type' in client_df.columns:
            client_types = client_df['client_talk_type'].dropna()
            if not client_types.empty:
                bar_data = client_types.value_counts().reset_index()
                bar_data.columns = ['Tipo di Talk', 'Conteggio']

                fig_bar = px.bar(bar_data, x='Tipo di Talk', y='Conteggio',
                                 color='Tipo di Talk', text='Conteggio',
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Non ci sono etichette disponibili per il paziente in questo dialogo.")