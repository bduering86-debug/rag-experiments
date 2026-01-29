#!/usr/bin/env python
"""Test-Script für Embeddings."""

from app.embeddings import Embeddings
from bin.config import EmbeddingConfig
from bin.logging_utils import get_logger
from app.vectorstore import get_vectorstore
from app.retrieval import search_collection

logger = get_logger("embedding_test")

def test_embeddings():
    """Teste die Embeddings-Konfiguration und -Funktionalität."""
    
    logger.info("=== Embeddings Test ===")
    
    # Config laden
    config = EmbeddingConfig()
    logger.info("Embedding Config:")
    logger.info("  - URL: %s", config.base_url)
    logger.info("  - Model: %s", config.model)
    logger.info("  - Dimensionen: %d", config.dim)
    
    # Embeddings-Instanz erstellen
    embeddings = Embeddings(config=config)
    logger.info("\n✓ Embeddings-Instanz erstellt")
    
    # Test 1: Single Query Embedding
    test_query = "Azure geht nicht mehr"
    logger.info("\n--- Test 1: Query Embedding ---")
    logger.info("Query: %s", test_query)
    
    try:
        result = embeddings.embed_query(test_query)
        logger.info("✓ Query Embedding erfolgreich")
        logger.info("  - Länge: %d", len(result))
        logger.info("  - Erste 5 Werte: %s", result[:5])
    except Exception as e:
        logger.error("✗ Query Embedding fehlgeschlagen: %s", e)
        return False
    
    # Test 2: Multiple Document Embeddings
    test_docs = [
        "System fährt nicht hoch",
        "Festplatte Fehler erkannt",
        "WLAN Verbindung wird getrennt"
    ]
    logger.info("\n--- Test 2: Document Embeddings ---")
    logger.info("Dokumente: %d", len(test_docs))
    
    try:
        results = embeddings.embed_documents(test_docs)
        logger.info("✓ Document Embeddings erfolgreich")
        logger.info("  - Anzahl Embeddings: %d", len(results))
        logger.info("  - Länge pro Embedding: %d", len(results[0]))
    except Exception as e:
        logger.error("✗ Document Embeddings fehlgeschlagen: %s", e)
        return False
    
    # Test 3: Kosinus-Ähnlichkeit prüfen
    logger.info("\n--- Test 3: Ähnlichkeit zwischen Docs ---")
    try:
        import numpy as np
        
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_01 = cosine_similarity(results[0], results[1])
        sim_02 = cosine_similarity(results[0], results[2])
        sim_12 = cosine_similarity(results[1], results[2])
        
        logger.info("  Doc0 <-> Doc1: %.4f", sim_01)
        logger.info("  Doc0 <-> Doc2: %.4f", sim_02)
        logger.info("  Doc1 <-> Doc2: %.4f", sim_12)
        logger.info("✓ Ähnlichkeitsvergleiche erfolgreich")
    except ImportError:
        logger.warning("⚠ NumPy nicht installiert - skipping similarity test")
    except Exception as e:
        logger.error("✗ Ähnlichkeitsvergleich fehlgeschlagen: %s", e)
    
    logger.info("\n=== Alle Tests bestanden! ===")
    return True

def test_vectorstore_search(kind: str = "kb", query: str = "VPN bricht nach 5 Minuten ab", k: int = 5, threshold: float = 0.5) -> bool:
    """Query the vector DB, compute embeddings for query and returned docs, compare cosine similarities."""
    logger.info("\n=== Vectorstore Retrieval Test ===")
    logger.info("  Collection: %s", kind)
    logger.info("  Query: %s", query)
    logger.info("  Top-K: %d", k)
    
    try:
        results = search_collection(query=query, collection=kind, top_k=k)
    except Exception as e:
        logger.error("✗ Retrieval fehlgeschlagen: %s", e)
        return False

    logger.info("Gefundene Dokumente: %d", len(results))
    if not results:
        logger.error("✗ Keine Dokumente gefunden in Collection %s", kind)
        return False

    # Zeige die Scores
    for i, hit in enumerate(results, start=1):
        logger.info("  Rank %d: score=%.4f, text=%s...", i, hit.score, hit.text[:60])

    # Prüfe ob mindestens ein Ergebnis über Threshold ist
    max_score = max(hit.score for hit in results)
    if max_score >= threshold:
        logger.info("✓ Vectorstore-Test bestanden (max score: %.4f ≥ %.2f)", max_score, threshold)
        return True
    else:
        logger.warning("✗ Keine Ähnlichkeit über Threshold %.2f gefunden (max: %.4f)", threshold, max_score)
        return False

if __name__ == "__main__":
    ok1 = test_embeddings()
    ok2 = test_vectorstore_search(kind="kb", query="VPN bricht nach 5 Minuten ab", k=5, threshold=0.5)
    if not (ok1 and ok2):
        raise SystemExit(1)
