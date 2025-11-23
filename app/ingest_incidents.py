import os
from tqdm import tqdm
from .loaders import load_incidents_csv
from .vectorstore import index_documents

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def main():
    cfg = DataConfig()

    #csv_path = os.path.join(BASE_DIR, "data", "incidents.csv")
    
    print("CSV-Pfad:", cfg.incident_path)
    docs = load_incidents_csv(cfg.incident_path)

    print(f"Lade {len(docs)} Incident-Dokumente in Qdrant ...")

    # Optional: tqdm herum, wenn du willst
    # Hier batcht index_documents intern
    index_documents(docs, kind="incidents")

    print("Incident-Ingest abgeschlossen.")


if __name__ == "__main__":
    main()
