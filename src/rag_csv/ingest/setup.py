#!/usr/bin/env python
"""
Setup-Script für Qdrant Collections.
Legt die benötigten Collections an oder recreated sie.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rag_csv.config.settings import QdrantConfig, EmbeddingConfig
from rag_csv.config.logging import get_logger

logger = get_logger(__name__)


def setup_collections(recreate: bool = False):
    """
    Erstelle Qdrant Collections mit korrekten Einstellungen.
    
    Args:
        recreate: Falls True, lösche bestehende Collections und lege sie neu an
    """
    qdrant_config = QdrantConfig()
    client = QdrantClient(url=qdrant_config.url)
    embedding_config = EmbeddingConfig()
    
    # Einstellungen für beide Collections
    collections = [
        {
            "name": qdrant_config.kb_collection,
            "description": "Knowledge Base Artikel"
        },
        {
            "name": qdrant_config.inc_collection,
            "description": "ITSM Incidents"
        }
    ]
    
    vector_params = VectorParams(
        size=embedding_config.dim,  # 384 Dimensionen (BAAI/bge-small)
        distance=Distance.COSINE
    )
    
    for collection in collections:
        name = collection["name"]
        desc = collection["description"]
        
        # Prüfe ob Collection existiert
        exists = client.collection_exists(name)
        
        if exists:
            if recreate:
                logger.info("Lösche bestehende Collection: %s", name)
                client.delete_collection(collection_name=name)
                client.create_collection(
                    collection_name=name,
                    vectors_config=vector_params
                )
                logger.info("✓ Collection %s (neu) erstellt: %s", name, desc)
            else:
                info = client.get_collection(name)
                logger.info("✓ Collection %s existiert bereits: %s (Points: %d)", 
                           name, desc, info.points_count or 0)
        else:
            logger.info("Erstelle neue Collection: %s", name)
            client.create_collection(
                collection_name=name,
                vectors_config=vector_params
            )
            logger.info("✓ Collection %s erstellt: %s", name, desc)


if __name__ == "__main__":
    import sys
    recreate = "--recreate" in sys.argv
    setup_collections(recreate=recreate)
    logger.info("Setup abgeschlossen!")
