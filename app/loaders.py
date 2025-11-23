from typing import List
import os
import pandas as pd
from langchain_core.documents import Document


def load_incidents_csv(path: str) -> List[Document]:
    df = pd.read_csv(path)
    docs: List[Document] = []

    for _, row in df.iterrows():

        # Felder aus CSV lesen und Variablen zuweisen
        ticket_id = str(row.get("ticket_id", ""))
        title = str(row.get("title", ""))
        desc = str(row.get("description", ""))
        history = str(row.get("history", ""))

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
            "status": row.get("status", ""),
            "category": row.get("category", ""),
            "impact": row.get("impact", ""),
            "urgency": row.get("urgency", ""),
            "created_at": row.get("created_at", ""),
            "resolved_at": row.get("resolved_at", ""),
        }

        # Dokument zusammenstellen und zur Liste hinzufügen
        docs.append(Document(page_content=content, metadata=metadata))

    return docs


def load_kb_csv(path: str) -> List[Document]:
    df = pd.read_csv(path)
    docs: List[Document] = []

    for _, row in df.iterrows():
        kb_id = str(row.get("kb_id", ""))
        title = str(row.get("title", ""))
        summary = str(row.get("summary", ""))
        content = str(row.get("content", ""))

        page_content = (
            f"KB-Artikel {kb_id}: {title}\n\n"
            f"Zusammenfassung:\n{summary}\n\n"
            f"Inhalt:\n{content}"
        )

        metadata = {
            "source": "kb",
            "kb_id": kb_id,
            "service": row.get("service", ""),
            "category": row.get("category", ""),
            "tags": row.get("tags", ""),
        }

        docs.append(Document(page_content=page_content, metadata=metadata))

    return docs
