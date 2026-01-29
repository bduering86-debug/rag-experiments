#!/usr/bin/env python3
"""
Migrations-Übersicht: alte → neue Projektstruktur
"""

print("""
╔════════════════════════════════════════════════════════════════════╗
║           RAG CSV - PROJEKTSTRUKTUR MIGRATION ✅                   ║
╚════════════════════════════════════════════════════════════════════╝

NEUE STRUKTUR:
==============

📦 src/rag_csv/              # Hauptpackage (pip-installierbar)
   ├── core/                 # Embeddings, Retrieval, Vectorstore
   ├── data/                 # Loaders, Chunking
   ├── config/               # Settings, Logging, Text Utils
   ├── utils/                # Metrics (nDCG, Recall)
   ├── generator/            # KB & Ticket Generator, Eval
   ├── benchmark/            # Runner, Visualization
   ├── ingest/               # Incidents, KB, Setup
   └── cli.py                # CLI Entry Point

📋 tests/                    # Unit & Integration Tests
   ├── test_embeddings.py
   ├── test_loaders.py
   ├── test_retrieval.py
   ├── test_generator.py
   └── test_rq.py

🔧 scripts/                  # Standalone Scripts
   ├── query_demo.py
   └── eval_models.py

📄 pyproject.toml            # Modern Python Project
📄 setup.py                  # Klassischer Installer
📄 PROJEKTSTRUKTUR.md        # Dokumentation
📄 test_new_structure.py     # Validierungs-Test


MIGRATIONS-ÜBERSICHT:
====================

Alte Pfade                         → Neue Pfade
─────────────────────────────────────────────────────────────────

bin/config.py                      → src/rag_csv/config/settings.py
bin/logging_utils.py               → src/rag_csv/config/logging.py
bin/text_utils.py                  → src/rag_csv/config/text.py
bin/metrics_utils.py               → src/rag_csv/utils/metrics.py

app/embeddings.py                  → src/rag_csv/core/embeddings.py
app/retrieval.py                   → src/rag_csv/core/retrieval.py
app/vectorstore.py                 → src/rag_csv/core/vectorstore.py
app/loaders.py                     → src/rag_csv/data/loaders.py
app/chunking.py                    → src/rag_csv/data/chunking.py
app/ingest_incidents.py            → src/rag_csv/ingest/incidents.py
app/ingest_kb.py                   → src/rag_csv/ingest/kb.py
app/setup_collections.py           → src/rag_csv/ingest/setup.py
app/eval_models.py                 → src/rag_csv/generator/evaluation.py
app/query_demo.py                  → scripts/query_demo.py
app/test_*.py                      → tests/test_*.py

generator/kb_generator.py          → src/rag_csv/generator/kb.py
generator/ticketgenerator.py       → src/rag_csv/generator/tickets.py
generator/generator_test.py        → tests/test_generator.py

benchmark/benchmark.py             → src/rag_csv/benchmark/runner.py
benchmark/visual_benchmark.py      → src/rag_csv/benchmark/visualize.py

metrics/nDCGTopK.py                → src/rag_csv/utils/nDCGTopK.py
metrics/RecallTopK.py              → src/rag_csv/utils/RecallTopK.py
metrics/test_rq.py                 → tests/test_rq.py


NEU: IMPORT PFADE:
==================

ALT (nicht mehr gültig):          NEU (verwenden):
─────────────────────────────────────────────────────────

from app.embeddings import         from rag_csv.core.embeddings import
from app.loaders import            from rag_csv.data.loaders import
from bin.config import             from rag_csv.config.settings import
from generator.kb_generator        from rag_csv.generator.kb import
from benchmark.benchmark import    from rag_csv.benchmark import
from metrics.nDCGTopK import        from rag_csv.utils import nDCGTopK


QUICK START:
===========

1. Installation:
   $ pip install -e .              # Editierbar
   $ pip install -e ".[dev]"       # Mit Test-Dependencies

2. Imports in Code:
   from rag_csv.core import search_collection
   from rag_csv.config.settings import QdrantConfig
   from rag_csv.generator import KBGenerator

3. CLI Commands:
   $ rag-query "search text"
   $ rag-ingest incidents
   $ rag-benchmark

4. Tests ausführen:
   $ python -m pytest tests/ -v
   $ python test_new_structure.py  # Validiere Import-Pfade


VALIDATION:
===========

✅ Neue Projektstruktur aktiv
✅ Alle imports funktionieren
✅ Alte Verzeichnisse gelöscht (app/, bin/, benchmark/, etc.)
✅ pyproject.toml + setup.py erstellt
✅ CLI Entry Points konfiguriert
✅ Dokumentation (PROJEKTSTRUKTUR.md) vorhanden


STATUS: MIGRATION ABGESCHLOSSEN ✅
══════════════════════════════════════════════════════════════════════
""")
