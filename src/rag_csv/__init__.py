"""
RAG CSV - Retrieval Augmented Generation System für CSV-basierte Knowledge Bases.

Ein intelligentes System zur Verwaltung und Abfrage von Wissensdatenbanken
mit modernen NLP-Technologien (Embeddings, Vektorsuche, LLMs).
"""

__version__ = "1.0.0"
__author__ = "Team RAG"

# Expose wichtige Module und Klassen
from .config.settings import (
    QdrantConfig,
    EmbeddingConfig,
    OllamaConfig,
    DataConfig,
    GeneratorConfig,
    LoggingConfig,
)
from .core.embeddings import Embeddings
from .core.retrieval import search_collection
from .data.loaders import load_incidents_csv, load_kb_csv

__all__ = [
    "QdrantConfig",
    "EmbeddingConfig",
    "OllamaConfig",
    "DataConfig",
    "GeneratorConfig",
    "LoggingConfig",
    "OllamaEmbeddings",
    "search_collection",
    "load_incidents_csv",
    "load_kb_csv",
]
