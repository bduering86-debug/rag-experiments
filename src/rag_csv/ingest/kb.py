# app/ingest_kb.py

from rag_csv.config.logging import get_logger
from rag_csv.config.settings import DataConfig
from rag_csv.data.loaders import load_kb_csv
from rag_csv.data.chunking import chunk_documents
from rag_csv.core.vectorstore import recreate_collection, ingest_documents, count_points

logger = get_logger(__name__)

def main():
    config = DataConfig()
    kb_path = config.kb_path

    logger.info("KB CSV: %s", kb_path)

    docs = load_kb_csv(kb_path)
    logger.info("KB Docs geladen: %d", len(docs))

    chunks = chunk_documents(docs, kind="kb")
    logger.info("KB Chunks erzeugt: %d", len(chunks))

    recreate_collection("kb")
    ingest_documents("kb", chunks, batch_size=64)

    pts = count_points("kb")
    logger.info("KB Collection points_count: %d", pts)


if __name__ == "__main__":
    main()
