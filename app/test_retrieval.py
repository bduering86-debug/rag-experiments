# app/test_retrieval.py
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

from bin.logging_utils import get_logger
from app.retrieval import search, SearchHit, hit_to_dict

logger = get_logger(__name__)


def print_hits(hits: List[SearchHit], as_json: bool = False) -> None:
    if as_json:
        payload = [hit_to_dict(h) for h in hits]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    for i, h in enumerate(hits, start=1):
        meta = h.metadata or {}
        src = meta.get("source", h.collection)

        if src == "kb":
            ref = meta.get("kb_id", "")
            title = meta.get("title", "")
        elif src == "incident":
            ref = meta.get("ticket_id", "")
            title = meta.get("title", "")
        else:
            ref = meta.get("kb_id") or meta.get("ticket_id") or ""
            title = meta.get("title", "")

        chunk_idx = meta.get("chunk_index", "")

        print("=" * 80)
        print(f"[{i}] collection={h.collection} source={src} score={h.score:.6f} chunk={chunk_idx}")
        print(f"ref={ref} title={title}".strip())
        print("-" * 80)
        print(h.text)
        print("-" * 80)
        # Optional: ein paar Meta-Felder anzeigen, ohne alles zu spammen
        # Unterschiedliche Felder je nach Quelle
        if src == "kb":
            keys_show = ["service", "category", "tags", "related_ticket_ids"]
        elif src == "incident":
            keys_show = ["status", "category", "impact", "urgency", "created_at", "resolved_at", "gold_kb_id"]
        else:
            keys_show = ["category", "service", "issue_type", "impact", "urgency", "created_at", "kb_fulltext"]
        
        shown = {k: meta.get(k) for k in keys_show if meta.get(k) is not None and meta.get(k) != ""}
        if shown:
            print("meta:", shown)


def save_hits_json(hits: List[SearchHit], out_path: str) -> None:
    payload = [hit_to_dict(h) for h in hits]
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Ergebnisse gespeichert: %s", out_path)


def main():
    p = argparse.ArgumentParser(description="RAG Retrieval Demo (nur Qdrant Search, ohne Ollama)")
    p.add_argument("query", type=str, help="Suchanfrage (Tickettext / Problem / Frage)")
    p.add_argument("--top-k", type=int, default=5, help="Top-K pro Collection")
    p.add_argument("--kb-only", action="store_true", help="Nur KB Collection durchsuchen")
    p.add_argument("--inc-only", action="store_true", help="Nur Incident Collection durchsuchen")
    p.add_argument("--no-merge", action="store_true", help="Nicht zusammenführen/sortieren (separat je Collection)")
    p.add_argument("--preview-chars", type=int, default=500, help="Vorschau-Länge je Treffer")
    p.add_argument("--json", action="store_true", help="Ausgabe als JSON auf STDOUT")
    p.add_argument("--save-json", type=str, default="", help="Ergebnisse zusätzlich als JSON speichern")
    args = p.parse_args()

    use_kb = True
    use_inc = True
    if args.kb_only:
        use_inc = False
    if args.inc_only:
        use_kb = False

    hits = search(
        query=args.query,
        top_k=args.top_k,
        use_kb=use_kb,
        use_incidents=use_inc,
        preview_chars=args.preview_chars,
        merge=(not args.no_merge),
    )

    if not hits:
        print("Keine Treffer gefunden.")
        return

    print_hits(hits, as_json=args.json)

    if args.save_json:
        # optional: Timestamp automatisch ergänzen, falls user nur Verzeichnis/Prefix gibt
        out = args.save_json
        if out.endswith(".json") is False:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = f"{out.rstrip('/')}_retrieval_{ts}.json"
        save_hits_json(hits, out)


if __name__ == "__main__":
    main()
