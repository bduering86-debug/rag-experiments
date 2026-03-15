#!/usr/bin/env python3
"""Direkter Test für Retrieval-Logging"""
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rag_csv.data.load_testcases import load_testcases
from rag_csv.core.retrieval import search

# Testcases laden
df = load_testcases("data/testcaes_quick.csv")
testcases = df.to_dict(orient="records")

print(f"✓ {len(testcases)} Testcases geladen\n")

# Log-Datei erstellen
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path("output/logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"testcases_retrieval_{timestamp}.log"

top_k = 10
success_count = 0

with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"✓ {len(testcases)} Testcases Retrieval-Test\n")
    f.write(f"Spalten: test_case_id, gold_kb_id, found, rank, score, retrieve_k, in_retrieve_k, retrieve_rank\n")
    
    for tc in testcases:
        test_case_id = tc.get('test_case_id', 'N/A')
        gold_kb_id = tc.get('gold_kb_id', '')
        
        # Query zusammenstellen
        query = f"{tc.get('ticket_title', '')} {tc.get('ticket_description', '')}"
        
        print(f"Testing {test_case_id}... ", end='', flush=True)
        
        try:
            # Retrieval durchführen
            hits = search(query, top_k=40, preview_chars=0)
            
            # Gold KB suchen
            found = False
            rank = None
            score = None
            retrieve_rank = None
            
            # In Top-K suchen
            for i, hit in enumerate(hits[:top_k], 1):
                hit_kb_id = hit.metadata.get('kb_id') if hit.metadata else None
                if hit_kb_id == gold_kb_id:
                    found = True
                    rank = i
                    score = hit.score
                    break
            
            # In vollständigem Retrieval-Set suchen
            for i, hit in enumerate(hits, 1):
                hit_kb_id = hit.metadata.get('kb_id') if hit.metadata else None
                if hit_kb_id == gold_kb_id:
                    retrieve_rank = i
                    break
            
            in_retrieve_k = retrieve_rank is not None and retrieve_rank <= 40
            
            if found:
                success_count += 1
            
            # Log-Zeile
            log_line = (f"{test_case_id}: gold={gold_kb_id or 'nan'} | "
                       f"found={found} | "
                       f"rank={rank} | "
                       f"score={score if score else 'None'} | "
                       f"retrieve_k={top_k} | "
                       f"in_retrieve_k={in_retrieve_k} | "
                       f"retrieve_rank={retrieve_rank}\n")
            f.write(log_line)
            print(f"✓ {'found' if found else 'not found'}")
            
        except Exception as e:
            error_line = f"{test_case_id}: ERROR | {str(e)}\n"
            f.write(error_line)
            print(f"✗ ERROR: {e}")
    
    # Summary
    summary = f"\nSummary: found {success_count}/{len(testcases)} testcases with gold in top-{top_k}\n"
    f.write(summary)

print(f"\n✅ Log erstellt: {log_file}")
print(f"   {success_count}/{len(testcases)} Gold-KB in Top-{top_k} gefunden")

# Log anzeigen
print("\n" + "="*80)
with open(log_file, 'r', encoding='utf-8') as f:
    print(f.read())
