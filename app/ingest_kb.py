# app/ingest_kb.py

from bin.logging_utils import get_logger
from bin.config import DataConfig
from app.loaders import load_kb_csv
from app.chunking import chunk_documents
from app.vectorstore import recreate_collection, ingest_documents, count_points

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
