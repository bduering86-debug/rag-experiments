from typing import Literal
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from bin.config import QdrantConfig, EmbeddingConfig
from .embeddings import Embeddings

def get_vectorstore(kind: Literal["incidents", "kb"]) -> Qdrant:
    cfg = QdrantConfig()
    emb_cfg = EmbeddingConfig()
    embeddings = Embeddings(emb_cfg)

    if kind == "incidents":
        collection = cfg.inc_collection
    else:
        collection = cfg.kb_collection

    client = QdrantClient(url=cfg.url)

    vs = Qdrant(
        client=client,
        collection_name=collection,
        embeddings=embeddings,
    )

    return vs

def index_documents(
    docs: list[Document],
    kind: Literal["incidents", "kb"],
    batch_size: int = 64,
) -> None:
    vs = get_vectorstore(kind)

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        vs.add_documents(batch)
