"""
Datenverarbeitung: Laden, Chunking und Vorbereitung von Dokumenten.
"""

from .loaders import load_incidents_csv, load_kb_csv
from .chunking import chunk_documents

__all__ = [
    "load_incidents_csv",
    "load_kb_csv",
    "chunk_text",
]
