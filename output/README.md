# Output Directory Structure

Dieses Verzeichnis enthält alle generierten Dateien des RAG CSV Systems.

## Struktur

```
output/
├── benchmarks/          # Benchmark-Ergebnisse
│   └── *.csv           # Model-Performance Metriken
├── generator/           # Generierte Daten
│   ├── synthetic_incidents_llm_*.csv
│   ├── kb_articles_llm.csv
│   └── synthetic_incidents_with_kb.csv
├── metrics/             # Evaluierungs-Metriken
│   └── *.csv           # nDCG, Recall, etc.
└── logs/                # Ollama Call Logs
    └── ollama_calls.csv
```

## Verwendung

Alle Module im `rag_csv` Package schreiben automatisch in die entsprechenden Unterordner:

### Benchmarks
```python
from rag_csv.generator.evaluation import benchmark_models
benchmark_models()  # → output/benchmarks/ollama_kb_benchmark_results.csv
```

### Generator
```python
from rag_csv.generator.tickets import TicketGenerator
gen = TicketGenerator()
gen.generate()  # → output/generator/synthetic_incidents_llm_<model>.csv
```

### Metrics
Evaluierungs-Metriken werden automatisch in `output/metrics/` gespeichert.

### Logs
Ollama API Call Logs werden in `output/logs/` gespeichert.

## Konfiguration

Output-Pfade können über `.env` angepasst werden:

```bash
OUTPUT_DIR=output                      # Hauptverzeichnis
OUTPUT_CSV_PATH=output/generator       # Generator Output
```

## .gitignore

In diesem Verzeichnis gelten aktuell folgende Ignore-Regeln:
- `output/**/*.json`
- `output/**/*.log`
- `output/**/ollama_calls.csv`
- `output/metrics/`

Ausnahme:
- `!output/README.md`

Hinweis: Generische `*.csv`-Regeln werden durch spezielle Ausnahmen (z. B. `!data/**/*.csv`) überschrieben.
