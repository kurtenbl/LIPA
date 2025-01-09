import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

BASE_PATH = "C:/Users/linus/Universitaet/LAI/Praktikum/Textgrundlage/Archiv/Archiv2"
CHROMA_PATH = "C:/Users/linus/chroma_db"

# 1. Lade PDFs und extrahiere Dokumente
def load_documents_from_folder(folder_path):
    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    return documents

# 2. Splitte die Dokumente in Chunks
def split_documents(documents, chunk_size=800, chunk_overlap=80):
    print("Splitte Dokumente in kleinere Chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"Anzahl der Chunks: {len(chunks)}")
    return chunks

def embed_and_store(chunks, chroma_path, embedding_model_name="all-MiniLM-L12-v2"):
    print("Initialisiere Embedding-Modell...")
    model = SentenceTransformer(embedding_model_name)
    
    # Chroma-Client und Sammlung erstellen
    print("Verbinde mit Chroma-Datenbank...")
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_or_create_collection(name="pdf_documents")
    
    # Embeddings erstellen
    print("Erstelle Embeddings...")
    for chunk in tqdm(chunks, desc="Verarbeite Chunks", unit="chunk"):
        text = chunk.page_content
        embedding = model.encode(text).tolist()
        
        # Extrahiere den Dokumentnamen und die Seitenzahl aus den Metadaten
        source_filename = os.path.basename(chunk.metadata.get('source', 'unbekannt'))
        page_number = chunk.metadata.get('page', 0)
        
        collection.add(
            ids=[f"{source_filename}_page_{page_number}"],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "source": source_filename,
                "page": page_number
            }]
        )
    
    print("Embeddings erfolgreich in Chroma-DB gespeichert!")

def process_all_folders():
    # Fortschrittsanzeige für Ordner
    for folder_number in tqdm(range(1, 106), desc="Verarbeite Ordner", unit="folder"):
        folder_path = os.path.join(BASE_PATH, str(folder_number))
        
        if os.path.exists(folder_path):
            try:
                # Dokumente laden
                documents = load_documents_from_folder(folder_path)
                
                # Dokumente in Chunks splitten
                chunks = split_documents(documents)
                
                # Embeddings erstellen und speichern
                embed_and_store(chunks, CHROMA_PATH)
                
            except Exception as e:
                print(f"Fehler bei Ordner {folder_number}: {e}")
        else:
            print(f"Ordner {folder_number} existiert nicht.")

# Hauptprogramm
if __name__ == "__main__":
    process_all_folders()

PROMPT_TEMPLATE = """
Context: 
{context}

Question: {question}

Based strictly on the provided context, give a precise and informative answer. If no relevant information is found, say "I cannot find specific information about this in the given context."
"""

from sentence_transformers import SentenceTransformer
import chromadb
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
def query_rag(query_text: str, top_k: int = 5):
    # Verwenden Sie die gleiche Embedding-Funktion wie beim Speichern
    embedding_function = SentenceTransformer("all-MiniLM-L12-v2")
    
    # Chroma-Client initialisieren
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name="pdf_documents")
    
    # Embedding der Suchanfrage
    query_embedding = embedding_function.encode(query_text).tolist()
    
    # Ähnlichkeitssuche
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Kontext aus den Dokumenten extrahieren
    context_text = "\n\n---\n\n".join(results['documents'][0])
    
    # Prompt template vorbereiten
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # LLM Aufruf
    model = OllamaLLM(model="llama3.2:latest")
    response_text = model.invoke(prompt)
    
    # Quellen extrahieren
    sources = results['ids'][0]
    
    # Formatierte Ausgabe
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)
    return response_text

query_rag("Those are legal documents concerning 40 years of oil exploitation. Please reconstruct key events of those years.")
