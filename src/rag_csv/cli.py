#!/usr/bin/env python3
"""
Command-Line Interface für RAG CSV System.

Provides CLI commands for ingesting, querying, and benchmarking.
"""

import sys
import argparse
from pathlib import Path

# Füge src zum Python-Pfad hinzu für relative Imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_csv.config.settings import DataConfig, QdrantConfig, EmbeddingConfig
from rag_csv.core.retrieval import search_collection
from rag_csv.ingest.incidents import main as ingest_incidents_main
from rag_csv.ingest.kb import main as ingest_kb_main
from rag_csv.ingest.setup import setup_collections


def query_cmd(args):
    """Query Command: Suche in Incidents oder KB."""
    query = args.query
    collection = args.collection or "incidents"
    top_k = args.top_k
    
    results = search_collection(
        query=query,
        collection=collection,
        top_k=top_k,
    )
    
    for i, hit in enumerate(results, 1):
        print(f"\n{i}. Score: {hit.score:.4f}")
        print(f"Content: {hit.text[:200]}...")
        print(f"Metadata: {hit.metadata}")


def ingest_cmd(args):
    """Ingest Command: Ingestion von Daten."""
    config = DataConfig()
    
    if args.source == "incidents" or args.source == "all":
        print("🔄 Ingestion of incidents...")
        ingest_incidents_main()
        print("✅ Incidents ingested")
    
    if args.source == "kb" or args.source == "all":
        print("🔄 Ingestion of KB...")
        ingest_kb_main()
        print("✅ KB ingested")


def benchmark_cmd(args):
    """Benchmark Command: Modelle evaluieren."""
    from rag_csv.generator.evaluation import benchmark_models
    
    print("🚀 Starting benchmark...")
    benchmark_models()
    print("✅ Benchmark completed")


def generate_cmd(args):
    """Generate Command: Daten generieren."""
    from rag_csv.generator.kb import KB_Generator
    
    if args.generator == "kb":
        print("🔄 Generating KB articles...")
        gen = KB_Generator()
        gen.generate()
        print("✅ KB generated")
    elif args.generator == "tickets":
        print("🔄 Generating tickets...")
        from rag_csv.generator.tickets import TicketGenerator
        gen = TicketGenerator()
        gen.generate()
        print("✅ Tickets generated")


def main():
    """Main CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        description="RAG CSV - Retrieval Augmented Generation System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Verfügbare Befehle")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Suche in Datenbank")
    query_parser.add_argument("query", type=str, help="Suchtext")
    query_parser.add_argument(
        "-c", "--collection",
        choices=["incidents", "kb"],
        help="Sammlung (incidents oder kb)"
    )
    query_parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Anzahl der Ergebnisse (default: 5)"
    )
    query_parser.set_defaults(func=query_cmd)
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Daten ingestion")
    ingest_parser.add_argument(
        "source",
        choices=["incidents", "kb", "all"],
        help="Datenquelle"
    )
    ingest_parser.set_defaults(func=ingest_cmd)
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Modelle benchmarken")
    bench_parser.set_defaults(func=benchmark_cmd)
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Daten generieren")
    gen_parser.add_argument(
        "generator",
        choices=["kb", "tickets"],
        help="Generator-Typ"
    )
    gen_parser.set_defaults(func=generate_cmd)
    
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
