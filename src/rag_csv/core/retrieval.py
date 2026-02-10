# app/retrieval.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import os

from langchain_core.documents import Document
from rag_csv.config.logging import get_logger
from rag_csv.core.vectorstore import get_vectorstore

logger = get_logger(__name__)

# Lazy-load Reranker (nur wenn aktiviert)
_reranker = None

def _get_reranker():
    """Lazy-loading des Rerankers."""
    global _reranker
    if _reranker is None:
        from rag_csv.core.reranking import CrossEncoderReranker
        model = os.getenv("RERANKING_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        _reranker = CrossEncoderReranker(model_name=model)
    return _reranker


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
    rerank: Optional[bool] = None,
) -> List[SearchHit]:
    """
    Führt Retrieval auf KB und/oder Incidents aus.
    Wenn merge=True, werden Treffer zusammengeführt und nach score sortiert.
    
    Args:
        query: Suchquery
        top_k: Anzahl der finalen Ergebnisse
        use_kb: KB durchsuchen
        use_incidents: Incidents durchsuchen
        preview_chars: Max. Zeichen im Text-Preview
        merge: Ergebnisse mergen und sortieren
        rerank: Cross-Encoder Reranking (None=aus .env, True/False=override)
        
    Returns:
        Liste von SearchHits
    """
    # Reranking-Einstellung aus .env wenn nicht explizit angegeben
    if rerank is None:
        rerank = os.getenv("USE_RERANKING", "false").lower() == "true"
    
    # Wenn Reranking aktiv: hole mehr Kandidaten
    initial_k = top_k
    if rerank:
        multiplier = int(os.getenv("RERANKING_TOP_K_MULTIPLIER", "3"))
        initial_k = top_k * multiplier
        logger.debug("Reranking aktiviert - hole initial %d Dokumente (top_k=%d * %d)",
                    initial_k, top_k, multiplier)
    
    results: List[SearchHit] = []

    if use_kb:
        results.extend(search_collection(query, "kb", initial_k, preview_chars=preview_chars))

    if use_incidents:
        results.extend(search_collection(query, "incident", initial_k, preview_chars=preview_chars))

    if merge:
        # Qdrant/LangChain Score ist i.d.R. distance -> kleiner ist besser
        results.sort(key=lambda x: x.score)
        # Limitiere auf initial_k Dokumente nach Merge
        results = results[:initial_k]
    
    # Optional: Reranking mit Cross-Encoder
    if rerank and results:
        try:
            reranker = _get_reranker()
            if reranker.enabled:
                logger.debug("Starte Reranking von %d Dokumenten auf Top-%d", len(results), top_k)
                results = reranker.rerank(query, results, top_k)
                logger.debug("Reranking abgeschlossen")
            else:
                # Fallback: normale Top-K Limitierung
                results = results[:top_k]
        except Exception as e:
            logger.warning("Reranking fehlgeschlagen: %s - verwende Original-Ranking", e)
            results = results[:top_k]
    elif not rerank:
        # Ohne Reranking: normale Top-K Limitierung
        results = results[:top_k]

    return results


def hit_to_dict(hit: SearchHit) -> Dict[str, Any]:
    return asdict(hit)
