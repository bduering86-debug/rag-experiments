# app/retrieval.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

from langchain_core.documents import Document
from bin.logging_utils import get_logger
from app.vectorstore import get_vectorstore

logger = get_logger(__name__)


@dataclass
class SearchHit:
    collection: str          # "kb" oder "incident"
    score: float             # bei Qdrant/LangChain meist "distance" (kleiner = besser)
    text: str                # page_content (ggf. gekürzt)
    metadata: Dict[str, Any] # Document.metadata


def _trim(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …"


def search_collection(
    query: str,
    collection: str,
    top_k: int,
    preview_chars: int = 500,
) -> List[SearchHit]:
    """
    Sucht in einer Collection (kb oder incident) und gibt Treffer inkl. Score zurück.
    """
    vs = get_vectorstore(collection)

    # LangChain gibt (Document, score) zurück
    hits: List[Tuple[Document, float]] = vs.similarity_search_with_score(query, k=top_k)

    out: List[SearchHit] = []
    for doc, score in hits:
        out.append(
            SearchHit(
                collection=collection,
                score=float(score),
                text=_trim(doc.page_content, preview_chars),
                metadata=dict(doc.metadata or {}),
            )
        )
    return out


def search(
    query: str,
    top_k: int = 5,
    use_kb: bool = True,
    use_incidents: bool = True,
    preview_chars: int = 500,
    merge: bool = True,
) -> List[SearchHit]:
    """
    Führt Retrieval auf KB und/oder Incidents aus.
    Wenn merge=True, werden Treffer zusammengeführt und nach score sortiert.
    """
    results: List[SearchHit] = []

    if use_kb:
        results.extend(search_collection(query, "kb", top_k, preview_chars=preview_chars))

    if use_incidents:
        results.extend(search_collection(query, "incident", top_k, preview_chars=preview_chars))

    if merge:
        # Qdrant/LangChain Score ist i.d.R. distance -> kleiner ist besser
        results.sort(key=lambda x: x.score)

    return results


def hit_to_dict(hit: SearchHit) -> Dict[str, Any]:
    return asdict(hit)
