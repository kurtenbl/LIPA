import streamlit as st
from RAG_set_up import query_rag
import os

def main():
    st.set_page_config(
        page_title="Busca documentos RAG", 
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Busca documentos RAG")

    # Session State für die Ergebnisse initialisieren
    if 'result' not in st.session_state:
        st.session_state.result = None

    # Suchleiste
    query = st.text_input(
        "Pon tu pregunta aquí:", 
        placeholder="p. ej. detalles sobre explotación petrolera"
    )

    # Optionen für Anzahl der Ergebnisse
    top_k = st.slider(
        "Numero de resultados:", 
        min_value=1, 
        max_value=10, 
        value=5
    )

    # Suchbutton
    if st.button("Busca", type="primary"):
        if query:
            with st.spinner('Buscando...'):
                try:
                    # Aufruf der query_rag Funktion und Speichern im Session State
                    st.session_state.result = query_rag(query, top_k)
                    
                except Exception as e:
                    st.error(f"Ein Fehler ist aufgetreten: {e}")
        else:
            st.warning("Bitte geben Sie eine Suchanfrage ein")

    # Ergebnisse anzeigen, wenn vorhanden
    if st.session_state.result is not None:
        # Container für das Ergebnis
        result_container = st.container()
        
        with result_container:
            st.success("Busqueda completada!")
            
            # Antwort und Quellen extrahieren
            answer = st.session_state.result.split("\nSources:")[0] if "\nSources:" in st.session_state.result else st.session_state.result
            sources = st.session_state.result.split("\nSources:")[1] if "\nSources:" in st.session_state.result else "Keine Quellen verfügbar"
            
            # Antwort anzeigen mit Kopier-Button
            st.subheader("Respuesta:")
            col1, col2 = st.columns([10,1])
            with col1:
                st.write(answer)
            with col2:
                st.code(answer, language=None)  # Dies erzeugt einen Copy-Button
            
            # Quellen anzeigen mit Kopier-Button
            st.subheader("Fuentes:")
            col3, col4 = st.columns([10,1])
            with col3:
                st.write(sources)
            with col4:
                st.code(sources, language=None)  # Dies erzeugt einen Copy-Button

    # Sidebar mit Systeminformationen
    st.sidebar.header("🛠 Detalles del sistema")
    st.sidebar.write("Modell: DeepSeek-r1:14b")
    st.sidebar.write("Embedding: all-MiniLM-L12-v2")
    st.sidebar.write(f"Base de datos: {os.path.abspath('chroma_db')}")

if __name__ == "__main__":
    main()