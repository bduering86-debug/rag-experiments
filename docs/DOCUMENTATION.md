# Project Documentation

## Overview

This repository contains a RAG pipeline with local embeddings, Qdrant vector storage, ingestion utilities, retrieval/evaluation helpers, and experiment tooling.

Core package: `src/rag_csv/`

- `config/`: settings, logging, text helpers
- `core/`: embeddings, retrieval, reranking, vectorstore
- `data/`: CSV loaders and chunking
- `ingest/`: collection setup and data ingestion
- `generator/`: synthetic KB/ticket generation and benchmarks
- `evaluation/`: orchestration logic for answer evaluation
- `utils/`: metrics, scoring, LLM judge/API helpers

## Quick Start (Developer)

```bash
cd /home/bduering/rag_csv
source venv/bin/activate
pip install -e .
```

Ensure required services are running (e.g. Qdrant / local model backend) according to your local setup.

### Setup collections

```bash
python -m rag_csv.ingest.setup --recreate
```

### Ingest data

```bash
python -m rag_csv.ingest.kb
python -m rag_csv.ingest.incidents
```

### Run tests

```bash
pytest tests/test_embeddings.py
pytest tests/test_retrieval.py
```

## CLI Commands

After `pip install -e .`, these console scripts are available:

```bash
rag-query "Outlook startet nicht" -c incidents -k 5
rag-ingest all
rag-benchmark
rag-generate tickets
```

Definitions are in `pyproject.toml` under `[project.scripts]` and implemented in `src/rag_csv/cli.py`.

## Data and Git Behavior

- `*.csv` is globally ignored in `.gitignore`.
- CSV files in `data/` are explicitly re-included via `!data/**/*.csv`.
- Generated artifacts under `output/` remain largely ignored except `output/README.md`.

## Main Entry Points

- `src/rag_csv/cli.py`
- `src/rag_csv/ingest/setup.py`
- `src/rag_csv/ingest/kb.py`
- `src/rag_csv/ingest/incidents.py`
- `src/rag_csv/core/retrieval.py`
- `src/rag_csv/core/vectorstore.py`


