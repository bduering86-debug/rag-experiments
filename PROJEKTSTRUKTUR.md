# Projekt-Struktur Dokumentation

## Neue Standard-Python-Projektstruktur

Das Projekt wurde von einer ad-hoc Verzeichnisstruktur zu einem konformen Python-Package reorganisiert.

### Verzeichnisübersicht

```
rag_csv/
├── src/rag_csv/                # Hauptpackage
│   ├── __init__.py             # Package exports
│   ├── cli.py                  # CLI Entry Points
│   ├── core/                   # Kernfunktionalität
│   │   ├── embeddings.py       # Embedding-Wrapper für Ollama
│   │   ├── retrieval.py        # Similarity Search & Retrieval
│   │   └── vectorstore.py      # Qdrant Integration
│   ├── data/                   # Datenverarbeitung
│   │   ├── loaders.py          # CSV → LangChain Documents
│   │   └── chunking.py         # Text-Chunking Strategien
│   ├── config/                 # Konfiguration
│   │   ├── settings.py         # Dataclasses (von bin/config.py)
│   │   ├── logging.py          # Logging Setup (von bin/logging_utils.py)
│   │   └── text.py             # Textverarbeitung (von bin/text_utils.py)
│   ├── utils/                  # Hilfsfunktionen
│   │   ├── metrics.py          # Ollama-Laufzeitmetriken (von bin/metrics_utils.py)
│   │   ├── nDCGTopK.py         # nDCG@K Metriken (von metrics/)
│   │   └── RecallTopK.py       # Recall@K Metriken (von metrics/)
│   ├── ingest/                 # Datenverwaltung
│   │   ├── incidents.py        # Incident Ingestion (von app/ingest_incidents.py)
│   │   ├── kb.py               # KB Ingestion (von app/ingest_kb.py)
│   │   └── setup.py            # Qdrant Collection Setup (von app/setup_collections.py)
│   ├── generator/              # Datengenerierung
│   │   ├── kb.py               # KB-Artikel Generator (von generator/kb_generator.py)
│   │   ├── tickets.py          # Ticket Generator (von generator/ticketgenerator.py)
│   │   └── evaluation.py       # Model Evaluation (von app/eval_models.py)
│   └── benchmark/              # Benchmarking
│       ├── runner.py           # Benchmark Runner (von benchmark/benchmark.py)
│       └── visualize.py        # Ergebnisvisualisierung (von benchmark/visual_benchmark.py)
├── tests/                      # Unit & Integration Tests
│   ├── test_embeddings.py      # Embedding Tests
│   ├── test_loaders.py         # Loader Tests
│   ├── test_retrieval.py       # Retrieval Tests
│   ├── test_generator.py       # Generator Tests (von generator/generator_test.py)
│   ├── test_error_analysis.py  # Error Analysis Tests
│   └── test_rq.py              # Metrik Tests (von metrics/test_rq.py)
├── scripts/                    # Standalone Scripts
│   ├── benchmark_embedding_models.sh
│   ├── local_metrics_server.py
│   ├── plot_experiment.py
│   ├── runs_overview.py
│   └── start_local_metrics_server.sh
├── output/                     # **NEU** Generierte Dateien
│   ├── benchmarks/             # Benchmark-Ergebnisse (CSV)
│   ├── generator/              # Generierte Tickets & KB (CSV)
│   ├── metrics/                # Evaluierungs-Metriken (CSV)
│   ├── logs/                   # Ollama Call Logs (CSV)
│   └── README.md               # Output-Dokumentation
├── docs/                       # Dokumentation
│   └── MIGRATION_SUMMARY.py    # Migrations-Übersicht
├── pyproject.toml              # Python Package Konfiguration (modern)
├── setup.py                    # Klassischer Python Package Installer
├── run_tests.py                # Test-Runner Utility
└── tests/test_new_structure.py # Struktur-Validierung
```

## Migrations-Summary

### Was wurde verschoben?

| Alte Struktur | Neue Struktur | Paket |
|---|---|---|
| `bin/config.py` | `src/rag_csv/config/settings.py` | rag_csv.config |
| `bin/logging_utils.py` | `src/rag_csv/config/logging.py` | rag_csv.config |
| `bin/text_utils.py` | `src/rag_csv/config/text.py` | rag_csv.config |
| `bin/metrics_utils.py` | `src/rag_csv/utils/metrics.py` | rag_csv.utils |
| `app/embeddings.py` | `src/rag_csv/core/embeddings.py` | rag_csv.core |
| `app/retrieval.py` | `src/rag_csv/core/retrieval.py` | rag_csv.core |
| `app/vectorstore.py` | `src/rag_csv/core/vectorstore.py` | rag_csv.core |
| `app/loaders.py` | `src/rag_csv/data/loaders.py` | rag_csv.data |
| `app/chunking.py` | `src/rag_csv/data/chunking.py` | rag_csv.data |
| `app/ingest_incidents.py` | `src/rag_csv/ingest/incidents.py` | rag_csv.ingest |
| `app/ingest_kb.py` | `src/rag_csv/ingest/kb.py` | rag_csv.ingest |
| `app/setup_collections.py` | `src/rag_csv/ingest/setup.py` | rag_csv.ingest |
| `app/eval_models.py` | `src/rag_csv/generator/evaluation.py` | rag_csv.generator |
| `generator/kb_generator.py` | `src/rag_csv/generator/kb.py` | rag_csv.generator |
| `generator/ticketgenerator.py` | `src/rag_csv/generator/tickets.py` | rag_csv.generator |
| `benchmark/benchmark.py` | `src/rag_csv/benchmark/runner.py` | rag_csv.benchmark |
| `benchmark/visual_benchmark.py` | `src/rag_csv/benchmark/visualize.py` | rag_csv.benchmark |
| `metrics/*.py` | `src/rag_csv/utils/` | rag_csv.utils |
| `app/test_*.py` | `tests/` | (standalone) |
| `app/query_demo.py` | `tests/query_demo.py` | (standalone) |

### Alte Verzeichnisse (gelöscht)

- `app/`
- `bin/`
- `generator/`
- `benchmark/`
- `metrics/`


## Neue Import-Pfade

### Alte Imports (funktionieren nicht mehr!)

```python
from app.embeddings import Embeddings
from bin.config import QdrantConfig
from generator.kb_generator import KBGenerator
```

### Neue Imports

```python
from rag_csv.core.embeddings import Embeddings
from rag_csv.config.settings import QdrantConfig
from rag_csv.generator.kb import KBGenerator
```

## Installation & Verwendung

### Entwicklung

```bash
# Im Projektroot:
pip install -e .                    # Editierbare Installation
pip install -e ".[dev,bench]"       # Mit dev & benchmark extras
```

### CLI Befehle

Die CLI kann nach Installation verwendet werden:

```bash
rag-query "Outlook startet nicht mehr" -c incidents -k 5
rag-ingest incidents
rag-ingest kb
rag-ingest all
rag-benchmark
rag-generate kb
rag-generate tickets
```

Alternative Aufrufe über Python-Module:

```bash
python -m rag_csv.ingest.kb
python -m rag_csv.ingest.incidents
```

## Modul-Übersicht

### `rag_csv.core`

Kernfunktionalität für RAG-System:

```python
from rag_csv.core.embeddings import Embeddings  # Ollama Embeddings Wrapper
from rag_csv.core.retrieval import search_collection  # Similarity Search
from rag_csv.core.vectorstore import get_vectorstore  # Qdrant Access
```

### `rag_csv.data`

Datenverarbeitung:

```python
from rag_csv.data.loaders import load_incidents_csv, load_kb_csv
from rag_csv.data.chunking import chunk_documents
```

### `rag_csv.config`

Konfiguration & Utilities:

```python
from rag_csv.config.settings import QdrantConfig, EmbeddingConfig, DataConfig
from rag_csv.config.logging import get_logger, setup_logging
from rag_csv.config.text import safe_parse_level, safe_split
```

### `rag_csv.generator`

Datengenerierung:

```python
from rag_csv.generator.kb import KBGenerator
from rag_csv.generator.tickets import TicketGenerator
from rag_csv.generator.evaluation import benchmark_models
```

### `rag_csv.utils`

Metriken & Hilsfunktionen:

```python
from rag_csv.utils.metrics import OllamaRunMetrics
from rag_csv.utils.nDCGTopK import nDCGTopK
from rag_csv.utils.RecallTopK import RecallTopK
```

## Konfiguration

### pyproject.toml

Moderne Python-Projektdefinition mit:
- Dependencies und Optional-Dependencies
- CLI Entry Points (`rag-*` Commands)
- Tool-Konfiguration (black, isort, mypy, pytest)
- Metadaten (author, version, description)

### setup.py

Klassischer Setup-Installer für Kompatibilität mit älteren pip-Versionen.

## Zukunft: Pip-Installation

Nach Veröffentlichung kann das Paket installiert werden:

```bash
pip install rag-csv
```

Dann können alle Module global importiert werden:

```python
import rag_csv
from rag_csv.core import search_collection
from rag_csv.generator import KBGenerator
```

## Häufige Fehler nach Migration

### ImportError: No module named 'rag_csv'

Lösungen:
1. `pip install -e .` im Projektroot ausführen
2. Oder `sys.path.insert(0, 'src')` vor Imports

### ImportError: Cannot find 'app.embeddings'

✓ Richtig: `from rag_csv.core.embeddings import Embeddings`
✗ Falsch: `from app.embeddings import Embeddings`

### ModuleNotFoundError in Tests

Tests verwenden neue Import-Pfade:
```python
# tests/test_embeddings.py
from rag_csv.core.embeddings import Embeddings
from rag_csv.config.settings import EmbeddingConfig
```

## Best Practices

1. **Imports**: Verwende absolute Imports über `rag_csv` package
2. **Tests**: Alle Tests in `tests/` mit `test_` prefix
3. **Scripts**: Standalone Scripts in `scripts/`
4. **Config**: .env ist single source of truth
5. **Logging**: Verwende `rag_csv.config.logging.get_logger(__name__)`

## Validierung

Zur Validierung der neuen Struktur:

```bash
pytest tests/test_new_structure.py
```

Dies testet alle wichtigen Imports und Konfigurationen.
