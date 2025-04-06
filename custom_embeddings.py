# custom_embeddings.py
import httpx
import logging
from langchain.embeddings.base import Embeddings
from typing import List

logger = logging.getLogger(__name__)

class NomicEmbeddings(Embeddings):
    def __init__(
        self,
        api_base: str,
        api_key: str = None,
        model: str = "nomic-ai/nomic-embed-text-v1.5-GGUF",
    ):
        """
        Initializes the NomicEmbeddings class.

        :param api_base: Base URL of the embedding API.
        :param api_key: (Optional) API key for authentication.
        :param model: Model name to use for embeddings.
        """
        self.api_base = api_base
        self.api_key = api_key
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents.

        :param texts: List of text strings to embed.
        :return: List of embedding vectors.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "input": texts
        }
        logger.debug(f"Sending payload to embedding API: {payload}")
        
        try:
            response = httpx.post(
                f"{self.api_base}/embeddings",
                json=payload,
                headers=headers,
                timeout=30  # Adjust timeout as needed
            )
            response.raise_for_status()
            data = response.json()
            embeddings = [item['embedding'] for item in data['data']]
            logger.debug(f"Received embeddings: {embeddings}")
            return embeddings
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error while fetching embeddings: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while fetching embeddings: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Generates an embedding for a single query.

        :param text: Text string to embed.
        :return: Embedding vector.
        """
        embedding = self.embed_documents([text])[0]
        logger.debug(f"Embedding for query '{text}': {embedding}")
        return embedding