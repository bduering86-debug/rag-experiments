from typing import List
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from rag_csv.config.settings import EmbeddingConfig
from rag_csv.config.logging import get_logger
from langchain_core.embeddings import Embeddings as LangChainEmbeddings

logger = get_logger("embeddings")


class Embeddings(LangChainEmbeddings):
    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self.batch_size = 1  # /api/embed kann nicht batchen

        # Parallelität für /api/embed
        self.max_workers = getattr(self.config, "max_workers", 14)  # Start: 12 bei 14 vCPU
        self.chunk_size = getattr(self.config, "chunk_size", 256)   # wie viele Texte pro Runde submitten
        self.http_pool_size = getattr(self.config, "http_pool_size", self.max_workers)

    def _make_session(self) -> requests.Session:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self.http_pool_size,
            pool_maxsize=self.http_pool_size,
            max_retries=0,
        )
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return s

    def _post_single(self, session: requests.Session, url: str, text: str) -> List[float]:
        resp = session.post(
            url,
            json={"model": self.config.model, "input": text},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("embeddings", [[]])[0]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        urls_to_try = [self.config.base_url]
        if self.config.fallback_url:
            urls_to_try.append(self.config.fallback_url)

        last_error = None

        for url_index, url in enumerate(urls_to_try):
            session = self._make_session()

            try:
                if url_index > 0:
                    logger.warning(f"Primärer Server nicht erreichbar, versuche Fallback: {url}")

                results: List[List[float]] = [None] * len(texts)  # type: ignore

                with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                    # in Chunks submitten -> weniger Overhead bei großen Mengen
                    for start in range(0, len(texts), self.chunk_size):
                        end = min(start + self.chunk_size, len(texts))
                        futures = {
                            ex.submit(self._post_single, session, url, texts[i]): i
                            for i in range(start, end)
                        }

                        for fut in as_completed(futures):
                            i = futures[fut]
                            results[i] = fut.result()

                if url_index > 0:
                    logger.info(f"Fallback-Server erfolgreich: {url}")
                return results

            except Exception as e:
                last_error = e
                logger.debug(f"Fehler bei Server {url}: {e}")
                if url_index == len(urls_to_try) - 1:
                    raise

        if last_error:
            raise last_error
        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]
