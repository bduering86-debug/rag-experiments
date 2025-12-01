from typing import List
import requests
from langchain_core.embeddings import Embeddings
from bin.config import EmbeddingConfig


class Embeddings(Embeddings):
    """
    Embedding-Wrapper.
    Nutzt /v1/embeddings mit Key 'input' und 'model'.
    """
s
    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        resp = requests.post(
            self.config.base_url,
            json={"input": texts, "model": self.config.model},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # LangChain ruft das beim Ingest auf
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        # LangChain ruft das bei der Suche auf
        return self._embed([text])[0]
