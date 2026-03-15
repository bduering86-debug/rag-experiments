# rag-experiments

Developer notes

See the full developer documentation in [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).

Common commands

```bash
source venv/bin/activate
pip install -e .
python -m rag_csv.ingest.setup --recreate
python -m rag_csv.ingest.kb
python -m rag_csv.ingest.incidents
pytest tests/test_embeddings.py

# optional via console scripts
rag-ingest all
rag-query "Outlook startet nicht" -c incidents -k 5
```

If you want the project to avoid runtime adapters, pin compatible versions of
`qdrant-client` and `langchain-qdrant` in your environment.

## Git/Dateien

- CSV-Dateien sind global über `.gitignore` per `*.csv` ausgeschlossen.
- Für das Projekt sind CSVs unter `data/` explizit erlaubt (`!data/**/*.csv`).