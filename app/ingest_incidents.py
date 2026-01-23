# app/ingest_incidents.py

from bin.logging_utils import get_logger
from bin.config import DataConfig
from app.loaders import load_incidents_csv
from app.chunking import chunk_documents
from app.vectorstore import recreate_collection, ingest_documents, count_points

logger = get_logger(__name__)

def main():
    config = DataConfig()
    inc_path = config.incident_path

    logger.info("INC CSV: %s", inc_path)

    docs = load_incidents_csv(inc_path)
    logger.info("INC Docs geladen: %d", len(docs))

    chunks = chunk_documents(docs, kind="incident")
    logger.info("INC Chunks erzeugt: %d", len(chunks))

    recreate_collection("incident")
    ingest_documents("incident", chunks, batch_size=64)

    pts = count_points("incident")
    logger.info("INC Collection points_count: %d", pts)


if __name__ == "__main__":
    main()
