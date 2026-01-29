#!/usr/bin/env python3
"""Test-Script zum Validieren der neuen Projektstruktur."""

import sys
from pathlib import Path

# Füge src zum sys.path hinzu
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("=== Teste neue Projektstruktur ===\n")

# Test 1: Config imports
try:
    from rag_csv.config.settings import (
        QdrantConfig,
        EmbeddingConfig,
        DataConfig,
        OllamaConfig,
    )
    print("✅ Config imports working")
except Exception as e:
    print(f"❌ Config import failed: {e}")
    sys.exit(1)

# Test 2: Instanziiere Config classes
try:
    q = QdrantConfig()
    print(f"✅ QdrantConfig: {q.url}")
    
    # EmbeddingConfig kann fehlschlagen wenn .env nicht vollständig
    try:
        e = EmbeddingConfig()
        print(f"✅ EmbeddingConfig: {e.url}")
    except Exception as e:
        print(f"⚠️  EmbeddingConfig (optional): {e}")
    
    d = DataConfig()
    print(f"✅ DataConfig: incidents_path={Path(d.incident_path).name}")
    
    o = OllamaConfig()
    urls_count = len([u for u in [o.url, o.url_low_profile, o.url_mid_profile, 
                                   o.url_high_profile, o.url_ultra_profile, o.url_test] if u])
    print(f"✅ OllamaConfig: loaded {urls_count} profile URLs")
except Exception as e:
    print(f"❌ Config instantiation failed: {e}")
    sys.exit(1)

# Test 3: Core imports
try:
    from rag_csv.core.embeddings import Embeddings
    from rag_csv.core.retrieval import search_collection
    from rag_csv.core.vectorstore import get_vectorstore
    print("✅ Core imports working")
except Exception as e:
    print(f"❌ Core import failed: {e}")
    sys.exit(1)

# Test 4: Data imports
try:
    from rag_csv.data.loaders import load_incidents_csv, load_kb_csv
    from rag_csv.data.chunking import chunk_documents
    print("✅ Data loaders imports working")
except Exception as e:
    print(f"❌ Data import failed: {e}")
    sys.exit(1)

# Test 5: Utils imports
try:
    from rag_csv.utils.metrics import OllamaRunMetrics
    from rag_csv.utils.nDCGTopK import nDCGTopK
    from rag_csv.utils.RecallTopK import RecallTopK
    print("✅ Utils imports working")
except Exception as e:
    print(f"❌ Utils import failed: {e}")
    sys.exit(1)

# Test 6: Generator imports
try:
    from rag_csv.generator.kb import KBGenerator
    from rag_csv.generator.tickets import TicketGenerator
    print("✅ Generator imports working")
except Exception as e:
    print(f"❌ Generator import failed: {e}")
    sys.exit(1)

# Test 7: Benchmark imports
try:
    from rag_csv.benchmark import run_benchmark, visualize_results
    print("✅ Benchmark imports working")
except Exception as e:
    print(f"❌ Benchmark import failed: {e}")
    sys.exit(1)

# Test 8: Ingest imports
try:
    from rag_csv.ingest import ingest_incidents, ingest_kb, setup_collections
    print("✅ Ingest imports working")
except Exception as e:
    print(f"❌ Ingest import failed: {e}")
    sys.exit(1)

print("\n=== Alle Tests bestanden! ===")
print("\nNeue Projektstruktur aktiv:")
print("  src/rag_csv/")
print("    ├── core/        (embeddings, retrieval, vectorstore)")
print("    ├── data/        (loaders, chunking)")
print("    ├── config/      (settings, logging, text)")
print("    ├── utils/       (metrics)")
print("    ├── generator/   (kb, tickets, evaluation)")
print("    ├── benchmark/   (runner, visualize)")
print("    └── ingest/      (incidents, kb, setup)")
print("\n  tests/            (alle Test-Dateien)")
print("  scripts/          (CLI-Scripts)")
