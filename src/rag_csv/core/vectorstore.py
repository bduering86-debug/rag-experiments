from typing import Literal, List
import types
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
from rag_csv.config.settings import QdrantConfig, EmbeddingConfig
from rag_csv.config.logging import get_logger
from .embeddings import Embeddings

logger = get_logger(__name__)


def get_embeddings() -> Embeddings:
    """Erstelle eine Embeddings-Instanz."""
    return Embeddings(config=EmbeddingConfig())


def _qdrant_client() -> QdrantClient:
    config = QdrantConfig()
    return QdrantClient(url=config.url)


def get_collection_name(kind: str) -> str:
    config = QdrantConfig()
    if kind in ("kb", "knowledgebase"):
        return config.kb_collection
    return config.inc_collection


def _ensure_search_method_on_client(client: QdrantClient) -> None:
    """Attach a compatible `search` method to older/newer QdrantClient instances.

    LangChain's Qdrant wrapper calls `client.search(...)`. Some installed
    `qdrant-client` versions expose a different method name (`query_points`).
    If `search` is missing, provide a small adapter that maps parameters to
    `query_points` so the LangChain code works without upgrading the package.
    """

    if hasattr(client, "search"):
        return

    def _search(self, *, collection_name, query_vector=None, query_filter=None,
                search_params=None, limit=10, offset=0, with_payload=True,
                with_vectors=False, score_threshold=None, consistency=None, **kwargs):
        # langchain may pass (vector_name, vector) as query_vector; map to `using` param
        using = None
        q = query_vector
        if isinstance(query_vector, tuple) and len(query_vector) == 2 and isinstance(query_vector[0], str):
            using = query_vector[0]
            q = query_vector[1]

        # call query_points which supports a universal `query` argument
        resp = self.query_points(
            collection_name=collection_name,
            query=q,
            query_filter=query_filter,
            search_params=search_params,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
            using=using,
            consistency=consistency,
            **kwargs,
        )

        # `query_points` may return different shapes depending on qdrant-client version.
        # Prefer `.points` (pydantic model list). Fall back to `.result` or raw response.
        if hasattr(resp, "points"):
            return resp.points
        if hasattr(resp, "result"):
            return resp.result
        # final fallback: try model_dump
        try:
            data = resp.model_dump()
            return data.get("points") or data.get("result") or resp
        except Exception:
            return resp

    client.search = types.MethodType(_search, client)


def get_vectorstore(kind: str) -> QdrantVectorStore:
    collection = get_collection_name(kind)
    client = _qdrant_client()
    # ensure client has a `search` method expected by langchain_qdrant
    _ensure_search_method_on_client(client)
    embeddings = get_embeddings()

    return QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
    )


def recreate_collection(kind: str) -> None:
    """
    Löscht Collection und legt sie neu an (für reproduzierbare Experimente).
    """
    collection = get_collection_name(kind)
    client = _qdrant_client()
    embedding_config = EmbeddingConfig()
    
    if client.collection_exists(collection):
        logger.warning("Lösche bestehende Collection: %s", collection)
        client.delete_collection(collection_name=collection)
    
    # Erstelle Collection mit korrekten Einstellungen
    vector_params = VectorParams(
        size=embedding_config.dim,
        distance=Distance.COSINE
    )
    client.create_collection(
        collection_name=collection,
        vectors_config=vector_params
    )
    logger.info("Collection %s neu erstellt.", collection)


def ingest_documents(kind: str, docs: List[Document], batch_size: int = 64) -> int:
    vs = get_vectorstore(kind)
    total = len(docs)
    logger.info("Ingest %s: %d Dokument-Chunks", kind, total)

    # batching, damit RAM stabil bleibt
    for i in range(0, total, batch_size):
        batch = docs[i:i + batch_size]
        vs.add_documents(batch)

    return total


def count_points(kind: str) -> int:
    collection = get_collection_name(kind)
    client = _qdrant_client()
    info = client.get_collection(collection)
    return int(info.points_count or 0)