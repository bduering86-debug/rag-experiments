#!/usr/bin/env python3
"""Test Retrieval mit verschiedenen TOP_K Werten."""

import sys
sys.path.insert(0, 'src')

from rag_csv.data.load_testcases import load_testcases, TESTCASES_FILE
from rag_csv.core.retrieval import search
from rag_csv.utils.nDCGTopK import nDCGTopK
from rag_csv.utils.RecallTopK import RecallTopK

# Testfall laden
df = load_testcases(TESTCASES_FILE)
testcase = df.iloc[0].to_dict()

print("="*70)
print(f"Testfall: {testcase['test_case_id']}")
print(f"Titel: {testcase['ticket_title'][:50]}...")
print(f"Gold KB: {testcase['gold_kb_id']}")
print("="*70)

# Teste mit verschiedenen TOP_K Werten
for top_k in [20, 30]:
    print(f"\n{'='*70}")
    print(f"TEST MIT TOP_K={top_k}")
    print(f"{'='*70}")
    
    # Retrieval
    hits = search(
        query=testcase['ticket_description'],
        top_k=top_k,
        use_kb=True,
        use_incidents=False,
        merge=True
    )
    
    print(f"\n✓ {len(hits)} Dokumente abgerufen")
    
    # IDs extrahieren
    retrieved_ids = [
        hit.metadata.get("kb_id") or hit.metadata.get("ticket_id", "")
        for hit in hits
    ]
    
    # Finde Gold KB Position
    gold_kb_id = testcase['gold_kb_id']
    try:
        gold_position = retrieved_ids.index(gold_kb_id) + 1
        print(f"✓ Gold KB gefunden auf Position: {gold_position}/{len(hits)}")
        
        # Zeige Top-5 mit Markierung
        print(f"\nTop-5 Dokumente:")
        for i in range(min(5, len(hits))):
            kb_id = retrieved_ids[i]
            score = hits[i].score
            marker = " ← GOLD KB!" if kb_id == gold_kb_id else ""
            print(f"  {i+1}. {kb_id} (Score: {score:.4f}){marker}")
        
        # Zeige Gold KB wenn nicht in Top-5
        if gold_position > 5:
            print(f"\n  ...")
            print(f"  {gold_position}. {gold_kb_id} (Score: {hits[gold_position-1].score:.4f}) ← GOLD KB!")
        
    except ValueError:
        print(f"✗ Gold KB NICHT in Top-{top_k} gefunden!")
        gold_position = None
    
    # Metriken berechnen
    print(f"\nMetriken:")
    relevant_ids = [gold_kb_id]
    
    # nDCG und Recall für verschiedene K-Werte
    for k in [3, 5, 10, top_k]:
        if k > top_k:
            continue
            
        ndcg_metric = nDCGTopK(k=k)
        recall_metric = RecallTopK(k=k)
        
        ndcg = ndcg_metric.compute(retrieved_ids, relevant_ids)
        recall = recall_metric.compute(retrieved_ids, relevant_ids)
        
        in_topk = gold_kb_id in retrieved_ids[:k]
        status = "✓" if in_topk else "✗"
        
        print(f"  {status} nDCG@{k:2d}: {ndcg:.4f} | Recall@{k:2d}: {recall:.4f} | Gold in Top-{k}: {in_topk}")

print(f"\n{'='*70}")
print("ZUSAMMENFASSUNG")
print(f"{'='*70}")
print(f"Gold KB muss höher ranken für bessere Metriken!")
print(f"Nächste Schritte: Reranking oder Query-Optimierung")
