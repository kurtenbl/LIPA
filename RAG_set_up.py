import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

BASE_PATH = "C:/Users/linus/Universitaet/LAI/Praktikum/Textgrundlage/Archiv/Archiv2"
CHROMA_PATH = "C:/Users/linus/chroma_db"

PROMPT_TEMPLATE = """
Context: 
{context}

Question: {question}

Provide a detailed and precise answer based strictly on the provided context. Include all relevant information and explain the key points comprehensively. If no relevant information is found, say "I cannot find specific information about this in the given context."
"""

from sentence_transformers import SentenceTransformer
import chromadb
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from concurrent.futures import ThreadPoolExecutor

# Pre-initialize embedding function and chroma client
embedding_function = SentenceTransformer("all-MiniLM-L12-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name="pdf_documents")

def encode_query(query_text):
    return embedding_function.encode(query_text).tolist()

def query_rag(query_text: str, top_k: int = 5):
    query_embedding = encode_query(query_text)
    
    # Perform similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Extract context from documents
    context_text = "\n\n---\n\n".join(results['documents'][0])
    
    # Prepare prompt template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Call LLM
    model = OllamaLLM(model="deepseek-r1:14b")
    response_text = model.invoke(prompt)
    
    # Extract sources
    sources = results['ids'][0]
    
    # Formatted output
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)
    return response_text

query_rag("Las empresas bloquean el flujo del agua en un sistema interconectado donde el agua del rio llega a descansar. Hay alguna evidencia de permiso para estos bloqueos?")