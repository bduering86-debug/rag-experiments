from typing import Literal, List
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
from bin.config import QdrantConfig, EmbeddingConfig
from bin.logging_utils import get_logger
from .embeddings import Embeddings

logger = get_logger(__name__)


def get_embeddings() -> Embeddings:
    """Erstelle eine Embeddings-Instanz."""
    return Embeddings(config=EmbeddingConfig())


def _qdrant_client() -> QdrantClient:
    return QdrantClient(url=QdrantConfig.url)


def get_collection_name(kind: str) -> str:
    if kind in ("kb", "knowledgebase"):
        return QdrantConfig.kb_collection
    return QdrantConfig.inc_collection


def get_vectorstore(kind: str) -> Qdrant:
    collection = get_collection_name(kind)
    client = _qdrant_client()
    embeddings = get_embeddings()

    return Qdrant(
        client=client,
        collection_name=collection,
        embeddings=embeddings,
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