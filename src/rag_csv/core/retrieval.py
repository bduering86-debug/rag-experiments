# app/retrieval.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import os

from langchain_core.documents import Document
from rag_csv.config.logging import get_logger
from rag_csv.config.settings import RetrievalConfig
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
    score: float             # bei Qdrant/LangChain (Cosine) ist es Similarity (groesser = besser)
    text: str                # page_content (ggf. gekürzt)
    metadata: Dict[str, Any] # Document.metadata


def _trim(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …"

def dedup_by_kb_id(docs_with_scores: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """Dedup nach kb_id; behalte den ersten Treffer pro kb_id in Ranking-Reihenfolge."""
    seen: set[str] = set()
    deduped: List[Tuple[Document, float]] = []
    for doc, score in docs_with_scores:
        meta = doc.metadata or {}
        doc_id = getattr(doc, "id", None)
        key = meta.get("kb_id") or doc_id or meta.get("_id") or meta.get("id") or None
        if key is None:
            # kein Key vorhanden: behalte separat, aber vermeide Crash
            deduped.append((doc, score))
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append((doc, score))
    return deduped

def _search_collection_raw(
    query: str,
    collection: str,
    top_k: int,
) -> List[Tuple[Document, float]]:
    """
    Sucht in einer Collection (kb oder incident) und gibt Treffer inkl. Score zurück.
    """
    vs = get_vectorstore(collection)
    # LangChain gibt (Document, score) zurück
    hits = vs.similarity_search_with_score(query, k=top_k)
    # sicherstellen, dass die Collection im Payload steckt
    for doc, _ in hits:
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata.setdefault("_collection_name", collection)
    return hits

def search_collection(
    query: str,
    collection: str,
    top_k: int,
    preview_chars: int = 500,
) -> List[SearchHit]:
    """
    Sucht in einer Collection (kb oder incident) und gibt Treffer inkl. Score zurück.
    """
    hits: List[Tuple[Document, float]] = _search_collection_raw(query, collection, top_k)

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
    
    if rerank:
        # Reranking-Pfad unveraendert lassen
        initial_k = top_k
        multiplier = int(os.getenv("RERANKING_TOP_K_MULTIPLIER", "3"))
        initial_k = initial_k * multiplier
        logger.debug("Reranking aktiviert - hole initial %d Dokumente (top_k=%d * %d)",
                    initial_k, top_k, multiplier)

        results: List[SearchHit] = []
        if use_kb:
            results.extend(search_collection(query, "kb", initial_k, preview_chars=preview_chars))
        if use_incidents:
            results.extend(search_collection(query, "incident", initial_k, preview_chars=preview_chars))

        if merge:
            # Qdrant/LangChain mit Cosine liefert Similarity -> groesser ist besser
            results.sort(key=lambda x: x.score, reverse=True)
            # Limitiere auf initial_k Dokumente nach Merge
            results = results[:initial_k]

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

        return results

    # Reines Retrieval (ohne Reranking)
    # Chunk-level Retrieval-K: mehr Chunks holen, dann deduplizieren
    retrieve_cfg = RetrievalConfig()
    chunk_multiplier = max(int(retrieve_cfg.retrieve_k_chunks_multiplier), 1)
    initial_k = top_k * chunk_multiplier
    
    raw_hits: List[Tuple[Document, float]] = []

    if use_kb:
        raw_hits.extend(_search_collection_raw(query, "kb", initial_k))

    if use_incidents:
        raw_hits.extend(_search_collection_raw(query, "incident", initial_k))

    if merge:
        # Qdrant/LangChain mit Cosine liefert Similarity -> groesser ist besser
        raw_hits.sort(key=lambda x: x[1], reverse=True)
        # Limitiere auf initial_k Dokumente nach Merge
        raw_hits = raw_hits[:initial_k]

    logger.debug("Raw chunk hits count: %d", len(raw_hits))
    raw_hits = dedup_by_kb_id(raw_hits)
    logger.debug("Unique kb_ids count (nach Dedup): %d", len(raw_hits))

    # Top-K nach Dedup
    raw_hits = raw_hits[:top_k]
    logger.debug("Final top_k_docs count: %d", len(raw_hits))

    # In SearchHit umwandeln
    results: List[SearchHit] = []
    for doc, score in raw_hits:
        results.append(
            SearchHit(
                collection=doc.metadata.get("_collection_name", ""),
                score=float(score),
                text=_trim(doc.page_content, preview_chars),
                metadata=dict(doc.metadata or {}),
            )
        )

    # Ohne Reranking: normale Top-K Limitierung
    results = results[:top_k]

    return results


def hit_to_dict(hit: SearchHit) -> Dict[str, Any]:
    return asdict(hit)
