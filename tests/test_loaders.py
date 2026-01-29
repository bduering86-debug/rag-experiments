# bin/test_loaders.py

from pathlib import Path
from rag_csv.config.settings import DataConfig
from rag_csv.data.loaders import load_incidents_csv, load_kb_csv

# Verwende eine DataConfig-Instanz, damit die Pfad-Properties (inkl. DATA_DIR) greifen
data_cfg = DataConfig()
INCIDENT_CSV = data_cfg.incident_path
KB_CSV = data_cfg.kb_path

def test_incidents():
    print("=== TEST: load_incidents_csv ===")

    docs = load_incidents_csv(str(INCIDENT_CSV))

    print(f"Anzahl Incident-Dokumente: {len(docs)}")
    if not docs:
        print("❌ FEHLER: Keine Incident-Dokumente geladen")
        return

    d = docs[0]
    print("\n--- Beispiel Incident (erstes Dokument) ---")
    print("page_content (gekürzt):")
    print(d.page_content[:400], "...\n")

    print("metadata:")
    for k, v in d.metadata.items():
        print(f"  {k}: {v}")

    # einfache Plausibilitätschecks
    assert "Incident" in d.page_content
    assert "ticket_id" in d.metadata
    assert d.metadata["source"] == "incident"

    print("✅ Incident-Loader OK\n")


def test_kb():
    print("=== TEST: load_kb_csv ===")

    docs = load_kb_csv(str(KB_CSV))

    print(f"Anzahl KB-Dokumente: {len(docs)}")
    if not docs:
        print("❌ FEHLER: Keine KB-Dokumente geladen")
        return

    d = docs[0]
    print("\n--- Beispiel KB (erstes Dokument) ---")
    print("page_content (gekürzt):")
    print(d.page_content[:400], "...\n")

    print("metadata:")
    for k, v in d.metadata.items():
        print(f"  {k}: {v}")

    # einfache Plausibilitätschecks
    assert "KB-Artikel" in d.page_content
    assert "kb_id" in d.metadata
    assert d.metadata["source"] == "kb"

    print("✅ KB-Loader OK\n")


def main():
    test_incidents()
    test_kb()


if __name__ == "__main__":
    main()
