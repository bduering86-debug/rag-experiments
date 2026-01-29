# app/chunking.py
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
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1100,
            chunk_overlap=180,
            separators=["\n\n", "\n", " | ", ". ", " "],
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=850,
            chunk_overlap=140,
            separators=["\n\n", "\n", " | ", ". ", " "],
        )

    chunks = splitter.split_documents(docs)

    for idx, d in enumerate(chunks):
        d.metadata["chunk_index"] = idx
        d.metadata["chunk_type"] = kind

    return chunks
