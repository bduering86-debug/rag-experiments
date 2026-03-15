#!/usr/bin/env python3
"""
Test Cross-Encoder Reranking.

Zeigt Verbesserung durch Reranking im Vergleich zu purem Embedding-Retrieval.
"""

import sys
sys.path.insert(0, 'src')

from rag_csv.data.load_testcases import load_testcases, TESTCASES_FILE
from rag_csv.core.retrieval import search
from rag_csv.utils.nDCGTopK import nDCGTopK
from rag_csv.utils.RecallTopK import RecallTopK

# Testfall laden
df = load_testcases(TESTCASES_FILE)
testcase = df.iloc[0].to_dict()

print("="*80)
print("CROSS-ENCODER RERANKING TEST")
print("="*80)
print(f"\nTestfall: {testcase['test_case_id']}")
print(f"Gold KB: {testcase['gold_kb_id']}")

# Test 1: OHNE Reranking
print(f"\n{'='*80}")
print("TEST 1: OHNE RERANKING (nur Embeddings)")
print("="*80)

hits_no_rerank = search(
    query=testcase['ticket_description'],
    top_k=10,
    use_kb=True,
    use_incidents=False,
    merge=True,
    rerank=False  # Explizit deaktiviert
)

print(f"\n✓ {len(hits_no_rerank)} Dokumente abgerufen")

# IDs extrahieren
retrieved_ids_no_rerank = [
    hit.metadata.get("kb_id") or hit.metadata.get("ticket_id", "")
    for hit in hits_no_rerank
]

# Finde Gold KB
gold_kb_id = testcase['gold_kb_id']
try:
    position_no_rerank = retrieved_ids_no_rerank.index(gold_kb_id) + 1
    print(f"✓ Gold KB Position: {position_no_rerank}/10")
except ValueError:
    position_no_rerank = None
    print("✗ Gold KB NICHT in Top-10")

# Metriken
relevant_ids = [gold_kb_id]
ndcg_metric = nDCGTopK(k=10)
recall_metric = RecallTopK(k=10)

ndcg_no_rerank = ndcg_metric.compute(retrieved_ids_no_rerank, relevant_ids)
recall_no_rerank = recall_metric.compute(retrieved_ids_no_rerank, relevant_ids)

print(f"\nMetriken OHNE Reranking:")
print(f"  nDCG@10: {ndcg_no_rerank:.4f}")
print(f"  Recall@10: {recall_no_rerank:.4f}")

# Test 2: MIT Reranking
print(f"\n{'='*80}")
print("TEST 2: MIT RERANKING (Cross-Encoder)")
print("="*80)

try:
    hits_rerank = search(
        query=testcase['ticket_description'],
        top_k=10,
        use_kb=True,
        use_incidents=False,
        merge=True,
        rerank=True  # Aktiviert!
    )
    
    print(f"\n✓ {len(hits_rerank)} Dokumente abgerufen (nach Reranking)")
    
    # IDs extrahieren
    retrieved_ids_rerank = [
        hit.metadata.get("kb_id") or hit.metadata.get("ticket_id", "")
        for hit in hits_rerank
    ]
    
    # Finde Gold KB
    try:
        position_rerank = retrieved_ids_rerank.index(gold_kb_id) + 1
        print(f"✓ Gold KB Position: {position_rerank}/10")
        
        # Zeige Top-5 mit Gold KB markiert
        print(f"\nTop-5 nach Reranking:")
        for i in range(min(5, len(hits_rerank))):
            kb_id = retrieved_ids_rerank[i]
            score = hits_rerank[i].score
            marker = " ← GOLD KB!" if kb_id == gold_kb_id else ""
            print(f"  {i+1}. {kb_id} (Score: {score:.4f}){marker}")
            
    except ValueError:
        position_rerank = None
        print("✗ Gold KB NICHT in Top-10")
    
    # Metriken
    ndcg_rerank = ndcg_metric.compute(retrieved_ids_rerank, relevant_ids)
    recall_rerank = recall_metric.compute(retrieved_ids_rerank, relevant_ids)
    
    print(f"\nMetriken MIT Reranking:")
    print(f"  nDCG@10: {ndcg_rerank:.4f}")
    print(f"  Recall@10: {recall_rerank:.4f}")
    
    # Vergleich
    print(f"\n{'='*80}")
    print("VERGLEICH")
    print("="*80)
    
    if position_no_rerank and position_rerank:
        improvement_pos = position_no_rerank - position_rerank
        print(f"\nPosition: {position_no_rerank} → {position_rerank} ({improvement_pos:+d})")
    
    improvement_ndcg = ndcg_rerank - ndcg_no_rerank
    improvement_recall = recall_rerank - recall_no_rerank
    
    print(f"nDCG@10:  {ndcg_no_rerank:.4f} → {ndcg_rerank:.4f} ({improvement_ndcg:+.4f})")
    print(f"Recall@10: {recall_no_rerank:.4f} → {recall_rerank:.4f} ({improvement_recall:+.4f})")
    
    if improvement_ndcg > 0:
        print(f"\n✓ Verbesserung: +{improvement_ndcg/ndcg_no_rerank*100 if ndcg_no_rerank > 0 else float('inf'):.1f}% nDCG")
    
except Exception as e:
    print(f"\n✗ Reranking fehlgeschlagen: {e}")
    print("\nHinweis: Installiere sentence-transformers:")
    print("  pip install sentence-transformers")

print(f"\n{'='*80}")
