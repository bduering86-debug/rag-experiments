"""
Cross-Encoder Reranking für verbessertes Retrieval.

Nutzt Cross-Encoder Modelle zur präziseren Bewertung von Query-Document-Paaren.
Cross-Encoder sind deutlich präziser als Bi-Encoder (Embeddings), da sie
die Interaktion zwischen Query und Dokument direkt modellieren.
"""

from typing import List, Optional
from dataclasses import dataclass

from rag_csv.config.logging import get_logger
from rag_csv.core.retrieval import SearchHit

logger = get_logger(__name__)


@dataclass
class RerankingConfig:
    """Konfiguration für Cross-Encoder Reranking."""
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    enabled: bool = True
    top_k_multiplier: int = 3  # Hole 3x mehr Dokumente für Reranking


class CrossEncoderReranker:
    """
    Rerankt Dokumente mit Cross-Encoder Modell.
    
    Cross-Encoder bewerten Query-Document-Paare direkt und sind daher
    präziser als reine Embedding-basierte Suche (Bi-Encoder).
    
    Empfohlene Modelle:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (schnell, gut)
    - cross-encoder/ms-marco-TinyBERT-L-2-v2 (sehr schnell)
    - cross-encoder/ms-marco-electra-base (langsamer, besser)
    """
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialisiert Cross-Encoder Reranker.
        
        Args:
            model_name: HuggingFace Modell-Name
        """
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.enabled = True
            logger.info("Cross-Encoder Reranker initialisiert - Model: %s", model_name)
        except ImportError:
            logger.warning("sentence-transformers nicht installiert - Reranking deaktiviert")
            logger.warning("Installiere mit: pip install sentence-transformers")
            self.enabled = False
        except Exception as e:
            logger.error("Fehler beim Laden des Cross-Encoder: %s", e)
            self.enabled = False
    
    def rerank(
        self,
        query: str,
        hits: List[SearchHit],
        top_k: Optional[int] = None
    ) -> List[SearchHit]:
        """
        Rerankt Hits mit Cross-Encoder.
        
        Args:
            query: Suchquery
            hits: Liste von SearchHits
            top_k: Anzahl der zurückzugebenden Hits (None = alle)
            
        Returns:
            Gerankte Liste von SearchHits
        """
        if not self.enabled:
            logger.debug("Reranking deaktiviert - gebe Original-Hits zurück")
            return hits[:top_k] if top_k else hits
        
        if not hits:
            return []
        
        logger.debug("Reranke %d Dokumente mit Cross-Encoder", len(hits))
        
        try:
            # Erstelle Query-Document Paare
            pairs = [[query, hit.text] for hit in hits]
            
            # Score mit Cross-Encoder
            scores = self.model.predict(pairs)
            
            # Kombiniere Hits mit neuen Scores
            reranked_hits = []
            for hit, score in zip(hits, scores):
                # Erstelle neuen Hit mit aktualisiertem Score
                # Achtung: Cross-Encoder Score ist höher=besser (im Gegensatz zu Distance)
                reranked_hit = SearchHit(
                    collection=hit.collection,
                    score=float(score),  # Neuer Score vom Cross-Encoder
                    text=hit.text,
                    metadata=hit.metadata
                )
                reranked_hits.append(reranked_hit)
            
            # Sortiere nach Score (absteigend - höher ist besser)
            reranked_hits.sort(key=lambda x: x.score, reverse=True)
            
            logger.debug("Reranking abgeschlossen - Top Score: %.4f", 
                        reranked_hits[0].score if reranked_hits else 0)
            
            # Limitiere auf top_k
            if top_k:
                reranked_hits = reranked_hits[:top_k]
            
            return reranked_hits
            
        except Exception as e:
            logger.error("Fehler beim Reranking: %s", e)
            # Fallback zu Original-Hits
            return hits[:top_k] if top_k else hits
    
    def batch_rerank(
        self,
        queries: List[str],
        hits_list: List[List[SearchHit]],
        top_k: Optional[int] = None
    ) -> List[List[SearchHit]]:
        """
        Rerankt mehrere Queries auf einmal (effizienter).
        
        Args:
            queries: Liste von Queries
            hits_list: Liste von Hit-Listen (eine pro Query)
            top_k: Anzahl der zurückzugebenden Hits pro Query
            
        Returns:
            Liste von gerankten Hit-Listen
        """
        if not self.enabled:
            return [hits[:top_k] if top_k else hits for hits in hits_list]
        
        results = []
        for query, hits in zip(queries, hits_list):
            reranked = self.rerank(query, hits, top_k)
            results.append(reranked)
        
        return results
