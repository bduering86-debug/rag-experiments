# rag-experiments

Developer notes

See the full developer documentation in [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).

Common commands

```bash
source venv/bin/activate
python -m app.setup_collections --recreate   # create qdrant collections
python -m app.ingest_kb
python -m app.ingest_incidents
python -m app.test_embeddings
```

If you want the project to avoid runtime adapters, pin compatible versions of
`qdrant-client` and `langchain-qdrant` in your environment.