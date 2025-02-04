import streamlit as st
from RAG_set_up import query_rag  # Importieren Sie die Funktion aus Ihrem bestehenden Skript
import os

def main():
    st.set_page_config(
        page_title="RAG Dokumentensuche", 
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Dokumentensuche mit RAG")

    # Suchleiste
    query = st.text_input(
        "Geben Sie Ihre Suchanfrage ein:", 
        placeholder="Z.B. Details zur Ölexploitation"
    )

    # Optionen für Anzahl der Ergebnisse
    top_k = st.slider(
        "Anzahl der Suchergebnisse:", 
        min_value=1, 
        max_value=10, 
        value=5
    )

    # Suchbutton
    if st.button("Suchen", type="primary"):
        if query:
            with st.spinner('Suche läuft...'):
                try:
                    # Aufruf der query_rag Funktion
                    result = query_rag(query, top_k)
                    
                    # Ergebnisse anzeigen
                    st.success("Suche abgeschlossen")
                    st.write(result)
                    
                except Exception as e:
                    st.error(f"Ein Fehler ist aufgetreten: {e}")
        else:
            st.warning("Bitte geben Sie eine Suchanfrage ein")

    # Sidebar mit Systeminformationen
    st.sidebar.header("🛠 Systemdetails")
    st.sidebar.write(f"Modell: Ollama LLaMA 3.2")
    st.sidebar.write(f"Embedding: all-MiniLM-L12-v2")
    st.sidebar.write(f"Datenbank-Pfad: {os.path.abspath('chroma_db')}")

if __name__ == "__main__":
    main()