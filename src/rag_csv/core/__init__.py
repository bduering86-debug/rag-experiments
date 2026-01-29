"""
Kernfunktionalität für Embeddings, Retrieval und Vektorspeicherung.
"""

from .embeddings import Embeddings
from .retrieval import search_collection, get_vectorstore as get_vectorstore_retrieval
from .vectorstore import get_vectorstore, recreate_collection, ingest_documents

__all__ = [
    "OllamaEmbeddings",
    "search_collection",
    "get_vectorstore",
    "create_vectorstore",
]
