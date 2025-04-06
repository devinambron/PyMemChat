# vector_store.py
import faiss
from langchain_community.vectorstores import FAISS
from custom_embeddings import NomicEmbeddings  # Import the custom embedding class
import logging
import numpy as np
from langchain.schema import Document

logger = logging.getLogger(__name__)

# Initialize OpenAIEmbeddings
def create_embedding_model():
    return NomicEmbeddings(
        model="nomic-ai/nomic-embed-text-v1.5-GGUF",
        api_key=None,  # Set to your actual API key if required
        api_base="http://localhost:1234/v1"
    )

def create_vector_store():
    texts = ["sample text"]

    embedding_model = create_embedding_model()

    # Generate embeddings for the texts
    embeddings = embedding_model.embed_documents(texts)
    logger.debug(f"Generated embeddings: {embeddings}")

    # Create FAISS vector store using embeddings
    # The API has changed: We need to provide the embedding model and the documents
    documents = [Document(page_content=text) for text in texts]
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model
    )

    logger.debug("Vector store created successfully using NomicEmbeddings.")
    return vector_store

def test_embedding():
    texts = ["This is a test."]
    embedding_model = create_embedding_model()
    try:
        embeddings = embedding_model.embed_documents(texts)
        logger.debug(f"Embeddings: {embeddings}")
        print(f"Embeddings: {embeddings}")  # {{ edit_1 }} - Added print statement to display embeddings
    except Exception as e:
        logger.error(f"Error embedding documents: {e}")
        raise e

if __name__ == "__main__":
    test_embedding()