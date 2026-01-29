Project Documentation

Overview

This repository contains a small RAG ingestion pipeline: chunking of text, embedding using a local embedding server, and vector storage in Qdrant. The main components are:

- `bin/config.py`: central configuration classes (QdrantConfig, EmbeddingConfig, DataConfig).
- `app/chunking.py`: text chunking functions.
- `app/embeddings.py`: Embeddings client (batched) talking to HTTP embedding service.
- `app/vectorstore.py`: Qdrant helper functions and a compatibility adapter for qdrant-client.
- `app/setup_collections.py`: create / recreate Qdrant collections.
- `app/ingest_kb.py`, `app/ingest_incidents.py`: ingestion scripts that use `DataConfig` instead of env vars.
- `app/retriever.py`: simple Retriever wrapper exposing `retrieve` and `retrieve_with_scores`.
- `app/test_embeddings.py`: small test harness for embeddings + vectorstore retrieval.

Quick start (developer)

1. Create and activate virtualenv (already present in this repo as `venv`):

```bash
cd /home/bduering/rag_csv
source venv/bin/activate
```

2. Ensure Qdrant and embedding server are running (docker-compose used for Qdrant; embedding server expected at `http://localhost:8080`.)

3. Create collections (optional):

```bash
python -m app.setup_collections --recreate
```

4. Ingest data:

```bash
python -m app.ingest_kb
python -m app.ingest_incidents
```

5. Run tests:

```bash
python -m app.test_embeddings
```

Notes about compatibility adapter

`app/vectorstore.py` contains a small runtime adapter that attaches a `search` method onto `qdrant_client.QdrantClient` instances when the installed `qdrant-client` version exposes `query_points` (or returns different response shapes). This keeps the bundled `langchain_qdrant` wrapper working without forcing an immediate dependency upgrade.

If you prefer pinning dependencies instead of using the adapter, pin `qdrant-client` and `langchain-qdrant` to mutually compatible versions in your environment.

Files to inspect for details

- `app/vectorstore.py`
- `app/retriever.py`
- `app/embeddings.py`
- `app/chunking.py`
- `README.md`

Contact

If you want me to produce a shorter developer README or pin dependencies in `requirements.txt`, tell me and I'll add it.
