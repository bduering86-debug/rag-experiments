from typing import List
import requests
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from bin.config import EmbeddingConfig


class Embeddings(LangChainEmbeddings):
    """
    Embedding-Wrapper.
    Nutzt /v1/embeddings mit Key 'input' und 'model'.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self.batch_size = 32  # Kleinere Batches zum Server senden

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Sende Texts in kleineren Batches zum Embeddings-Server."""
        all_embeddings = []
        
        # Teile Texts in kleinere Batches auf
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = requests.post(
                self.config.base_url,
                json={"input": batch, "model": self.config.model},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            batch_embeddings = [item["embedding"] for item in data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # LangChain ruft das beim Ingest auf
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        # LangChain ruft das bei der Suche auf
        return self._embed([text])[0]
