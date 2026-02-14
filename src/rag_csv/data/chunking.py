# app/chunking.py
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def chunk_documents(docs: List[Document], kind: str) -> List[Document]:
    """
    Applies RAG chunking strategy to documents.

    kind:
      - 'kb'        → Knowledge Base articles
      - 'incident'  → ITSM Incidents
    """
    if kind == "kb":
        chunk_size = int(os.getenv("KB_CHUNK_SIZE", "1100"))
        chunk_overlap = int(os.getenv("KB_CHUNK_OVERLAP", "180"))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " | ", ". ", " "],
        )
    else:
        chunk_size = int(os.getenv("INCIDENT_CHUNK_SIZE", "850"))
        chunk_overlap = int(os.getenv("INCIDENT_CHUNK_OVERLAP", "140"))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " | ", ". ", " "],
        )

    chunks = splitter.split_documents(docs)

    for idx, d in enumerate(chunks):
        d.metadata["chunk_index"] = idx
        d.metadata["chunk_type"] = kind

    return chunks
