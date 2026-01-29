from typing import List
import requests
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from bin.config import EmbeddingConfig


class Embeddings(LangChainEmbeddings):
    """
    Embedding-Wrapper für Ollama.
    Unterstützt beide Endpoints:
    - /api/embed (nativer Ollama, einzelnes 'input' String pro Request)
    - /v1/embeddings (OpenAI-kompatibel, 'input' kann Liste sein)
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self.batch_size = 32 if self._is_openai_compatible() else 1  # Batching nur für v1/embeddings
        
    def _is_openai_compatible(self) -> bool:
        """Prüfe ob URL den OpenAI-kompatiblen Endpoint nutzt."""
        return "/v1/embeddings" in self.config.base_url

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Sende Texts zu Ollama (mit automatischer Endpoint-Erkennung)."""
        all_embeddings = []
        
        if self._is_openai_compatible():
            # OpenAI-kompatibel: Batch-Support mit "input" als Liste
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                resp = requests.post(
                    self.config.base_url,
                    json={"model": self.config.model, "input": batch},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                batch_embeddings = data.get("embeddings", [])
                all_embeddings.extend(batch_embeddings)
        else:
            # Nativer Ollama /api/embed: "input" als String, kein Batch-Support
            # Response: {"embeddings": [[float, float, ...]]} - 2D-Array mit einem Element
            for text in texts:
                resp = requests.post(
                    self.config.base_url,
                    json={"model": self.config.model, "input": text},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                # /api/embed gibt {"embeddings": [[...]]} zurück, wir wollen nur das erste Element
                embedding = data.get("embeddings", [[]])[0]
                all_embeddings.append(embedding)
        
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # LangChain ruft das beim Ingest auf
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        # LangChain ruft das bei der Suche auf
        return self._embed([text])[0]
