
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_vector_index(chunks):
    # Initialize Hugging Face embeddings with a specific model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create a FAISS vector store from text chunks using embeddings
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    return vector_store

def query_vector_index(vector_store, query):
    # Perform a similarity search in the vector store for the query
    docs = vector_store.similarity_search(query)
    return docs
