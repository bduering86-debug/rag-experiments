from typing import List
import requests
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from rag_csv.config.settings import EmbeddingConfig
from rag_csv.config.logging import get_logger

logger = get_logger("embeddings")


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
        """Sende Texts zu Ollama (mit automatischer Endpoint-Erkennung und Fallback)."""
        all_embeddings = []
        
        # Versuche primären Server, bei Fehler Fallback
        urls_to_try = [self.config.base_url]
        if self.config.fallback_url:
            urls_to_try.append(self.config.fallback_url)
        
        last_error = None
        for url_index, url in enumerate(urls_to_try):
            try:
                if url_index > 0:
                    logger.warning(f"Primärer Server nicht erreichbar, versuche Fallback: {url}")
                
                if self._is_openai_compatible():
                    # OpenAI-kompatibel: Batch-Support mit "input" als Liste
                    for i in range(0, len(texts), self.batch_size):
                        batch = texts[i:i + self.batch_size]
                        resp = requests.post(
                            url,
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
                            url,
                            json={"model": self.config.model, "input": text},
                            timeout=120,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        # /api/embed gibt {"embeddings": [[...]]} zurück, wir wollen nur das erste Element
                        embedding = data.get("embeddings", [[]])[0]
                        all_embeddings.append(embedding)
                
                # Erfolgreich - return
                if url_index > 0:
                    logger.info(f"Fallback-Server erfolgreich: {url}")
                return all_embeddings
                
            except Exception as e:
                last_error = e
                logger.debug(f"Fehler bei Server {url}: {e}")
                if url_index == len(urls_to_try) - 1:
                    # Letzter Versuch fehlgeschlagen
                    raise
                # Sonst: nächster Server
                all_embeddings = []  # Reset für nächsten Versuch
        
        # Sollte nie erreicht werden, aber sicherheitshalber
        if last_error:
            raise last_error
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # LangChain ruft das beim Ingest auf
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        # LangChain ruft das bei der Suche auf
        return self._embed([text])[0]
