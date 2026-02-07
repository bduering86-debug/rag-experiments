#!/usr/bin/env python3
"""
RAG Ingestion Main Orchestrator.

Orchestriert den kompletten RAG-Flow:
1. Laden von Datenquellen (Incidents, KB-Artikel)
2. Chunking der Dokumente
3. Embedding-Erstellung
4. Vektorisierung und Speicherung in Qdrant
"""

from typing import List, Literal
from pathlib import Path

from langchain_core.documents import Document

from rag_csv.config.logging import get_logger
from rag_csv.config.settings import DataConfig
from rag_csv.data.loaders import load_incidents_csv, load_kb_csv
from rag_csv.data.chunking import chunk_documents
from rag_csv.core.vectorstore import recreate_collection, ingest_documents, count_points

logger = get_logger(__name__)


class RAGIngestor:
    """
    Hauptklasse für RAG-Ingestion.
    
    Orchestriert den kompletten Flow von Datenquellen bis zur Vektorisierung.
    """
    
    def __init__(self, data_config: DataConfig = None):
        """
        Initialisiert den RAG Ingestor.
        
        Args:
            data_config: Optional DataConfig, falls None wird neue Instanz erstellt
        """
        self.config = data_config or DataConfig()
        logger.info("RAG Ingestor initialisiert")
    
    def load_source(self, source: Literal["incidents", "kb"], file_path: str = None) -> List[Document]:
        """
        Lädt Datenquelle und konvertiert zu LangChain Documents.
        
        Args:
            source: "incidents" oder "kb"
            file_path: Optional: Pfad zur Datei (überschreibt Config)
            
        Returns:
            List[Document]: Geladene Dokumente
        """
        logger.info("📁 Lade Datenquelle: %s", source)
        logger.info("📁 Pfad: %s", file_path)
        
        if source == "incidents":
            path = file_path or self.config.incident_path
            docs = load_incidents_csv(path)
        elif source == "kb":
            path = file_path or self.config.kb_path
            docs = load_kb_csv(path)
        else:
            raise ValueError(f"Unbekannte Datenquelle: {source}")
        
        logger.info("✓ %d Dokumente geladen aus %s", len(docs), path)
        return docs
    
    def chunk(self, docs: List[Document], kind: str) -> List[Document]:
        """
        Chunking der Dokumente.
        
        Args:
            docs: Liste von Documents
            kind: "kb" oder "incident" für verschiedene Chunking-Strategien
            
        Returns:
            List[Document]: Gechunkte Dokumente
        """
        logger.info("✂️  Chunking: %d Dokumente (Typ: %s)", len(docs), kind)
        chunks = chunk_documents(docs, kind=kind)
        logger.info("✓ %d Chunks erstellt", len(chunks))
        return chunks
    
    def embed_and_store(self, chunks: List[Document], collection: str, recreate: bool = True) -> int:
        """
        Erstellt Embeddings und speichert in Qdrant Vectorstore.
        
        Args:
            chunks: Gechunkte Dokumente
            collection: Name der Collection
            recreate: Falls True, wird Collection neu erstellt (Default: True)
            
        Returns:
            int: Anzahl der gespeicherten Punkte
        """
        logger.info("🔄 Embedding & Vektorisierung für Collection '%s'", collection)
        
        # Collection neu erstellen falls gewünscht
        if recreate:
            logger.info("🗄️  Recreate Collection '%s'", collection)
            recreate_collection(collection)
        
        # Embeddings erstellen und speichern
        logger.info("💾 Speichere %d Chunks in Qdrant...", len(chunks))
        ingest_documents(collection, chunks, batch_size=64)
        
        # Prüfe Anzahl der Punkte
        points = count_points(collection)
        logger.info("✅ %d Punkte in Collection '%s' gespeichert", points, collection)
        
        return points
    
    def ingest_full_pipeline(
        self,
        source: Literal["incidents", "kb"],
        collection: str = None,
        file_path: str = None,
        recreate: bool = True
    ) -> int:
        """
        Kompletter Ingestion-Pipeline: Laden -> Chunking -> Embedding -> Vektorisierung.
        
        Args:
            source: "incidents" oder "kb"
            collection: Name der Collection (optional, wird aus source abgeleitet)
            file_path: Optional: Pfad zur Datei
            recreate: Collection neu erstellen (Default: True)
            
        Returns:
            int: Anzahl der gespeicherten Punkte
        """
        logger.info("=== RAG Ingestion Pipeline gestartet ===")
        logger.info("Source: %s", source)
        
        # Collection-Name bestimmen
        if collection is None:
            collection = "incidents" if source == "incidents" else "knowledgebase"
        
        # 1. Laden
        docs = self.load_source(source, file_path)
        
        # 2. Chunking
        kind = "kb" if source == "kb" else "incident"
        chunks = self.chunk(docs, kind)
        
        # 3. Embedding & Vektorisierung
        points = self.embed_and_store(chunks, collection, recreate)
        
        logger.info("=== Pipeline abgeschlossen ===")
        return points
    
    def ingest_all(self, recreate: bool = True) -> dict:
        """
        Ingestiert alle konfigurierten Datenquellen (Incidents + KB).
        
        Args:
            recreate: Collections neu erstellen (Default: True)
            
        Returns:
            dict: Dictionary mit Anzahl der Punkte pro Collection
        """
        logger.info("=== Vollständige RAG Ingestion gestartet ===")
        
        results = {}
        
        # Incidents
        try:
            points_inc = self.ingest_full_pipeline("incidents", recreate=recreate)
            results["incidents"] = points_inc
        except Exception as e:
            logger.error("Fehler bei Incidents-Ingestion: %s", e)
            results["incidents"] = 0
        
        # KB
        try:
            points_kb = self.ingest_full_pipeline("kb", recreate=recreate)
            results["knowledgebase"] = points_kb
        except Exception as e:
            logger.error("Fehler bei KB-Ingestion: %s", e)
            results["knowledgebase"] = 0
        
        logger.info("=== Vollständige Ingestion abgeschlossen ===")
        logger.info("Ergebnis: %s", results)
        
        return results


def main():
    """CLI Entry Point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Ingestion Orchestrator")
    parser.add_argument(
        "source",
        nargs="?",
        choices=["incidents", "kb", "all"],
        default="all",
        help="Datenquelle: incidents, kb oder all (default: all)"
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Collection nicht neu erstellen (erweitert bestehende)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Pfad zur Datei (überschreibt Config)"
    )
    
    args = parser.parse_args()
    
    ingestor = RAGIngestor()
    recreate = not args.no_recreate
    
    try:
        if args.source == "all":
            results = ingestor.ingest_all(recreate=recreate)
            print(f"\n✅ Ingestion abgeschlossen:")
            for coll, count in results.items():
                print(f"   - {coll}: {count} Punkte")
        else:
            points = ingestor.ingest_full_pipeline(
                source=args.source,
                file_path=args.file,
                recreate=recreate
            )
            print(f"\n✅ Ingestion abgeschlossen: {points} Punkte")
    except Exception as e:
        logger.error("❌ Fehler: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
