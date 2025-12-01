from .loaders import load_kb_csv
from .vectorstore import index_documents
from bin.config import DataConfig


def main():
    cfg = DataConfig()

    print("CSV-Pfad:", cfg.kb_path)
    docs = load_kb_csv(cfg.kb_path)

    print(f"Lade {len(docs)} KB-Dokumente in Qdrant ...")
    index_documents(docs, kind="kb")

    print("⚡ KB-Ingest abgeschlossen.")
