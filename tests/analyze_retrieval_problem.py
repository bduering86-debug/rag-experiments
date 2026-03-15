#!/usr/bin/env python3
"""Analyse: Warum ist Gold KB am schlechtesten?"""

import sys
sys.path.insert(0, 'src')

from rag_csv.data.load_testcases import load_testcases, TESTCASES_FILE
from rag_csv.core.retrieval import search

# Testfall laden
df = load_testcases(TESTCASES_FILE)
testcase = df.iloc[0].to_dict()

print("="*80)
print("QUERY ANALYSE")
print("="*80)
print(f"\nTest Case: {testcase['test_case_id']}")
print(f"Titel: {testcase['ticket_title']}")
print(f"\nQUERY (Ticket Description):")
print("-"*80)
print(testcase['ticket_description'][:500])
print("..." if len(testcase['ticket_description']) > 500 else "")

print(f"\n{'='*80}")
print("GOLD KB ARTIKEL")
print("="*80)
print(f"KB ID: {testcase['gold_kb_id']}")
print(f"\nGOLD KB Content:")
print("-"*80)
gold_kb_text = testcase.get('gold_kb_fulltext', 'N/A')
print(gold_kb_text[:800])
print("..." if len(gold_kb_text) > 800 else "")

# Retrieval mit Score-Details
print(f"\n{'='*80}")
print("TOP-3 vs. GOLD KB - VERGLEICH")
print("="*80)

hits = search(
    query=testcase['ticket_description'],
    top_k=30,
    use_kb=True,
    use_incidents=False,
    merge=True
)

print("\nTop-3 Dokumente (beste Matches):")
for i in range(min(3, len(hits))):
    print(f"\n#{i+1} - Score: {hits[i].score:.4f}")
    print(f"KB ID: {hits[i].metadata.get('kb_id')}")
    print(f"Text: {hits[i].text[:200]}...")

# Finde Gold KB
gold_kb_id = testcase['gold_kb_id']
for i, hit in enumerate(hits, 1):
    if hit.metadata.get('kb_id') == gold_kb_id:
        print(f"\n{'='*80}")
        print(f"GOLD KB - Position {i}/30 - Score: {hit.score:.4f}")
        print("="*80)
        print(f"Text: {hit.text[:400]}...")
        break

print(f"\n{'='*80}")
print("PROBLEM-ANALYSE")
print("="*80)
print("""
Mögliche Gründe warum Gold KB schlechten Score hat:

1. CHUNKING: Der relevante Teil des Gold KB ist nicht im gechunkten Text
2. QUERY-KB MISMATCH: Query beschreibt Problem, KB beschreibt Lösung
3. EMBEDDING-PROBLEM: Modell erfasst semantische Ähnlichkeit nicht
4. KONTEXT FEHLT: Gold KB braucht mehr Kontext zum Verständnis

Lösungen:
→ Chunks größer machen oder überlappend
→ Query mit technischen Begriffen aus KB anreichern
→ Hybrid Search (BM25 + Dense)
→ Cross-Encoder Reranking
""")
