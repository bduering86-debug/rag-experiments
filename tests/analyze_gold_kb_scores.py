#!/usr/bin/env python3
"""
Analysiert warum Gold KB schlechte Similarity-Scores bekommt.
Zeigt alle Retrieval-Scores und vergleicht Gold KB mit Top-Retrievals.
"""

import sys
sys.path.insert(0, 'src')

from rag_csv.core.retrieval import search
from rag_csv.data.load_testcases import load_testcases, TESTCASES_FILE

# Testcase laden
df = load_testcases(TESTCASES_FILE)
testcase = df.iloc[0].to_dict()

print("=" * 80)
print("=== TESTCASE DETAILS ===")
print("=" * 80)
print(f"Test Case ID: {testcase['test_case_id']}")
print(f"Titel: {testcase['ticket_title']}")
print(f"Beschreibung: {testcase['ticket_description'][:200]}...")
print(f"Gold KB ID: {testcase['gold_kb_id']}")
print(f"\nGold KB Fulltext (erste 300 Zeichen):")
print(testcase['gold_kb_fulltext'][:300])
print("...")

# Query zusammenbauen (wie im Orchestrator)
query = f"Titel: {testcase['ticket_title']}\nBeschreibung: {testcase['ticket_description']}"

print("\n" + "=" * 80)
print("=== RETRIEVAL mit K=30 (um Gold KB zu finden) ===")
print("=" * 80)

# Retrieval mit K=30 ohne Reranking
hits = search(
    query=query,
    top_k=30,
    use_kb=True,
    use_incidents=False,
    rerank=False  # Kein Reranking für diese Analyse
)

print(f"\n✓ {len(hits)} Dokumente abgerufen\n")

# Finde Gold KB Position
gold_id = testcase['gold_kb_id']
gold_position = None
gold_score = None

for i, hit in enumerate(hits, 1):
    kb_id = hit.metadata.get('kb_id', 'UNKNOWN')
    if kb_id == gold_id:
        gold_position = i
        gold_score = hit.score
        break

print("=" * 80)
print("=== TOP 10 RETRIEVALS ===")
print("=" * 80)

for i, hit in enumerate(hits[:10], 1):
    kb_id = hit.metadata.get('kb_id', 'UNKNOWN')
    is_gold = "  ⭐ GOLD KB" if kb_id == gold_id else ""
    print(f"{i:2d}. Score: {hit.score:.6f} | ID: {kb_id}{is_gold}")
    # Erste 150 Zeichen des Texts
    text_preview = hit.text[:150].replace('\n', ' ')
    print(f"    Text: {text_preview}...")
    print()

if gold_position and gold_position > 10:
    print(f"\n{'=' * 80}")
    print(f"=== GOLD KB gefunden an Position {gold_position} ===")
    print(f"{'=' * 80}")
    gold_hit = hits[gold_position - 1]
    print(f"Score: {gold_hit.score:.6f}")
    print(f"Text: {gold_hit.text[:300]}...")
    print()

print("=" * 80)
print("=== SCORE ANALYSE ===")
print("=" * 80)

if gold_position:
    print(f"✓ Gold KB gefunden: Position {gold_position}/30")
    print(f"  Gold Score: {gold_score:.6f}")
    print(f"  Top-1 Score: {hits[0].score:.6f}")
    print(f"  Score-Differenz: {abs(hits[0].score - gold_score):.6f}")
    print(f"  Relative Differenz: {(abs(hits[0].score - gold_score) / hits[0].score * 100):.1f}%")
else:
    print("✗ Gold KB NICHT in Top-30 gefunden!")

# Histogram der Scores
print(f"\n{'=' * 80}")
print("=== SCORE DISTRIBUTION (Top-30) ===")
print(f"{'=' * 80}")

score_ranges = [
    (0.0, 0.3, "Sehr niedrig"),
    (0.3, 0.5, "Niedrig"),
    (0.5, 0.7, "Mittel"),
    (0.7, 0.9, "Hoch"),
    (0.9, 1.0, "Sehr hoch")
]

for min_s, max_s, label in score_ranges:
    count = sum(1 for hit in hits if min_s <= hit.score < max_s)
    bar = "█" * count
    print(f"{label:12s} [{min_s:.1f}-{max_s:.1f}): {count:2d} {bar}")

print(f"\n{'=' * 80}")
print("=== EMBEDDING ANALYSE ===")
print(f"{'=' * 80}")

# Zeige Embedding-Modell Info
from rag_csv.config.settings import EmbeddingConfig
emb_config = EmbeddingConfig()
print(f"Embedding Model: {emb_config.model}")
print(f"Embedding Dimension: {emb_config.dim}")
print(f"Embedding Server: {emb_config.base_url}")

print(f"\n{'=' * 80}")
print("=== PROBLEM-DIAGNOSE ===")
print(f"{'=' * 80}")

if not gold_position:
    print("❌ KRITISCH: Gold KB nicht in Top-30!")
    print("   → Embedding Model erfasst Semantik nicht richtig")
    print("   → Empfehlung: Anderes Embedding-Modell testen")
elif gold_position > 20:
    print(f"⚠️  SCHLECHT: Gold KB an Position {gold_position} (sollte Top-3 sein)")
    print(f"   → Score zu niedrig: {gold_score:.6f}")
    print("   → Mögliche Ursachen:")
    print("     1. Embedding Model nicht optimal für deutschen Text")
    print("     2. Query-Formulierung suboptimal")
    print("     3. KB-Artikel-Text-Chunking problematisch")
    print("     4. Distance Metric (Cosine) evtl. nicht ideal")
elif gold_position > 10:
    print(f"⚠️  SUBOPTIMAL: Gold KB an Position {gold_position} (sollte Top-3 sein)")
    print("   → Cross-Encoder Reranking würde helfen!")
else:
    print(f"✓ AKZEPTABEL: Gold KB in Top-10 (Position {gold_position})")
    if gold_position > 3:
        print("  → Könnte mit Reranking auf Top-3 verbessert werden")

print(f"\n{'=' * 80}")
print("=== EMPFEHLUNGEN ===")
print(f"{'=' * 80}")
print("1. Cross-Encoder Reranking aktivieren (fetcht 30, rerankt Top-10)")
print("   → pip install sentence-transformers")
print("   → USE_RERANKING=true in .env")
print()
print("2. Alternative Embedding Models testen:")
print("   → paraphrase-multilingual-mpnet-base-v2 (besser für Deutsch)")
print("   → e5-large-v2 (neueres Modell)")
print()
print("3. Query Expansion:")
print("   → Synonyme hinzufügen")
print("   → Mit LLM umformulieren")
print()
print("4. Hybrid Search:")
print("   → BM25 + Vector Search kombinieren")
