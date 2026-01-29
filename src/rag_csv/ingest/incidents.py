# app/ingest_incidents.py

from rag_csv.config.logging import get_logger
from rag_csv.config.settings import DataConfig
from rag_csv.data.loaders import load_incidents_csv
from rag_csv.data.chunking import chunk_documents
from rag_csv.core.vectorstore import recreate_collection, ingest_documents, count_points

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
