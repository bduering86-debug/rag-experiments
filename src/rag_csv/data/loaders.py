from typing import List
import os
import csv

# pandas ist optional; wenn nicht installiert, verwenden wir csv.DictReader als Fallback
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    from langchain_core.documents import Document
except Exception:
    # Minimal fallback Document for environments without langchain_core
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict


def load_incidents_csv(path: str) -> List[Document]:
    docs: List[Document] = []

    if pd is not None:
        df = pd.read_csv(path)
        iterator = (row for _, row in df.iterrows())
    else:
        f = open(path, newline='', encoding='utf-8')
        reader = csv.DictReader(f)
        iterator = reader

    for row in iterator:

        # Felder aus CSV lesen und Variablen zuweisen
        ticket_id = str(row.get("ticket_id", ""))
        title = str(row.get("title", ""))
        desc = str(row.get("description", ""))
        history = str(row.get("conversation_history", ""))

        # Kontext dür LM:
        content = (
            f"Incident {ticket_id}: {title}\n\n"
            f"Beschreibung:\n{desc}\n\n"
            f"Verlauf:\n{history}"
        )

        # Metadten zusammnenstellen'
        metadata = {
            "source": "incident",
            "ticket_id": ticket_id,
            "title": title,
            "status": row.get("status", ""),
            "category": row.get("category", ""),
            "impact": row.get("impact", ""),
            "urgency": row.get("urgency", ""),
            "created_at": row.get("created_at", ""),
            "resolved_at": row.get("resolved_at", ""),
            "gold_kb_id": row.get("gold_kb_id", ""),
            "conversation_history": history,
        }

        # Dokument zusammenstellen und zur Liste hinzufügen
        docs.append(Document(page_content=content, metadata=metadata))

    if pd is None:
        f.close()
    return docs


def load_kb_csv(path: str) -> List[Document]:
    docs: List[Document] = []

    if pd is not None:
        df = pd.read_csv(path)
        iterator = (row for _, row in df.iterrows())
    else:
        f = open(path, newline='', encoding='utf-8')
        reader = csv.DictReader(f)
        iterator = reader

    for row in iterator:
        kb_id = str(row.get("kb_id", ""))
        title = str(row.get("title", ""))
        
        # Verwende kb_fulltext wenn verfügbar, sonst baue Content zusammen
        kb_fulltext = str(row.get("kb_fulltext", ""))
        
        if kb_fulltext and kb_fulltext != "nan" and kb_fulltext.strip():
            # Nutze das strukturierte kb_fulltext Feld direkt
            page_content = kb_fulltext
        else:
            # Fallback: Baue Content aus Einzelfeldern
            problem = str(row.get("problem", ""))
            symptoms = str(row.get("symptoms", ""))
            root_cause = str(row.get("root_cause", ""))
            resolution_steps = str(row.get("resolution_steps", ""))
            validation = str(row.get("validation", ""))

            page_content = (
                f"KB-Artikel {kb_id}: {title}\n\n"
                f"Problem:\n{problem}\n\n"
                f"Symptome:\n{symptoms}\n\n"
                f"Ursache:\n{root_cause}\n\n"
                f"Lösungsschritte:\n{resolution_steps}\n\n"
                f"Validierung:\n{validation}"
            )

        metadata = {
            "source": "kb",
            "kb_id": kb_id,
            "title": title,
            "service": row.get("service", ""),
            "category": row.get("category", ""),
            "tags": row.get("tags", ""),
            "related_ticket_ids": row.get("related_ticket_ids", ""),
        }

        docs.append(Document(page_content=page_content, metadata=metadata))

    if pd is None:
        f.close()
    return docs
