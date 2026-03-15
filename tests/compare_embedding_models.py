#!/usr/bin/env python3
"""
Vergleicht verschiedene Embedding-Modelle für das Gold KB Retrieval-Problem.
Testet welches Modell die beste Position für Gold KB erreicht.
"""

import sys
import os
sys.path.insert(0, 'src')

from rag_csv.data.load_testcases import load_testcases, TESTCASES_FILE
from rag_csv.core.embeddings import Embeddings
from rag_csv.config.settings import EmbeddingConfig, QdrantConfig

# Test-Modelle (auf Ollama verfügbar)
TEST_MODELS = [
    ("bge-m3", 1024),              # Aktuell
    ("nomic-embed-text", 768),     # Top für Retrieval
    ("mxbai-embed-large", 1024),   # Sehr gut für semantische Suche
    ("all-minilm", 384),           # Schnell, kompakt
]

print("=" * 80)
print("=== EMBEDDING MODEL COMPARISON ===")
print("=" * 80)
print("\nTestet welches Modell die beste Position für Gold KB erreicht")
print("Testcase: TC-P-01 (Monitor schwarz nach Neustart)\n")

# Lade Testcase
df = load_testcases(TESTCASES_FILE)
testcase = df.iloc[0].to_dict()

query = f"Titel: {testcase['ticket_title']}\nBeschreibung: {testcase['ticket_description']}"
gold_id = testcase['gold_kb_id']

print(f"Query: {testcase['ticket_title']}")
print(f"Gold KB: {gold_id}")
print(f"Gold Text: {testcase['gold_kb_fulltext'][:150]}...")

# Verfügbare Modelle auf Server prüfen
print("\n" + "=" * 80)
print("=== SCHRITT 1: Verfügbare Modelle auf Ollama prüfen ===")
print("=" * 80)

import requests

try:
    emb_config = EmbeddingConfig()
    response = requests.get(f"{emb_config.base_url.replace('/api/embed', '/api/tags')}", timeout=5)
    available_models = {m['name'].replace(':latest', '') for m in response.json().get('models', [])}
    
    print(f"\n✓ Verbindung zu {emb_config.base_url}")
    print(f"✓ {len(available_models)} Modelle verfügbar")
    
    # Filtere nur verfügbare Test-Modelle
    available_test_models = [(name, dim) for name, dim in TEST_MODELS if name in available_models]
    
    if not available_test_models:
        print("\n⚠️  Keines der Test-Modelle ist installiert!")
        print("\nInstalliere Modelle mit:")
        for model, _ in TEST_MODELS:
            if model not in available_models:
                print(f"  ollama pull {model}")
        sys.exit(1)
    
    print(f"\n✓ {len(available_test_models)} Test-Modelle verfügbar:")
    for model, dim in available_test_models:
        print(f"  - {model} ({dim} dim)")
    
except Exception as e:
    print(f"\n✗ Fehler bei Verbindung zu Ollama: {e}")
    print("\nFahre trotzdem fort mit allen Modellen (ggf. Fehler bei nicht installierten)...")
    available_test_models = TEST_MODELS

print("\n" + "=" * 80)
print("=== SCHRITT 2: Modelle testen ===")
print("=" * 80)

results = []

from qdrant_client import QdrantClient
from rag_csv.config.settings import QdrantConfig

qdrant_config = QdrantConfig()
client = QdrantClient(url=qdrant_config.url)

for model_name, model_dim in available_test_models:
    print(f"\n{'─' * 80}")
    print(f"Testing: {model_name} ({model_dim} dim)")
    print(f"{'─' * 80}")
    
    try:
        # Temporär Embedding Model wechseln
        os.environ["EMBEDDING_MODEL"] = model_name
        os.environ["EMBEDDING_DIM"] = str(model_dim)
        
        # Erstelle Embeddings für Query
        emb_config = EmbeddingConfig()
        embedder = Embeddings(config=emb_config)
        
        print(f"  Embedding Query...")
        query_embedding = embedder.embed_query(query)
        
        if len(query_embedding) != model_dim:
            print(f"  ⚠️  Warnung: Erwartete {model_dim} dim, bekam {len(query_embedding)}")
        
        # Search in Qdrant (mit neuem Embedding)
        print(f"  Suche in Qdrant (K=30)...")
        
        search_result = client.search(
            collection_name=qdrant_config.kb_collection,
            query_vector=query_embedding,
            limit=30
        )
        
        # Finde Gold KB Position
        gold_position = None
        gold_score = None
        
        for i, hit in enumerate(search_result, 1):
            kb_id = hit.payload.get('kb_id', 'UNKNOWN')
            if kb_id == gold_id:
                gold_position = i
                gold_score = hit.score
                break
        
        if gold_position:
            print(f"  ✓ Gold KB gefunden: Position {gold_position}/30")
            print(f"    Score: {gold_score:.6f}")
        else:
            print(f"  ✗ Gold KB NICHT in Top-30!")
            gold_position = 31  # Für Vergleich
            gold_score = 0.0
        
        # Top-3 anzeigen
        print(f"\n  Top-3 Results:")
        for i, hit in enumerate(search_result[:3], 1):
            kb_id = hit.payload.get('kb_id', 'UNKNOWN')
            is_gold = " ⭐" if kb_id == gold_id else ""
            print(f"    {i}. {kb_id} (Score: {hit.score:.6f}){is_gold}")
        
        results.append({
            'model': model_name,
            'dim': model_dim,
            'position': gold_position,
            'score': gold_score,
            'top1_score': search_result[0].score if search_result else 0,
            'found': gold_position <= 30
        })
        
    except Exception as e:
        print(f"  ✗ Fehler: {e}")
        results.append({
            'model': model_name,
            'dim': model_dim,
            'position': 99,
            'score': 0,
            'top1_score': 0,
            'found': False
        })

print("\n" + "=" * 80)
print("=== ERGEBNISSE ===")
print("=" * 80)

# Sortiere nach Position (beste zuerst)
results.sort(key=lambda x: x['position'])

print(f"\n{'Model':<25} {'Dim':<6} {'Gold Position':<15} {'Score':<12} {'Status'}")
print("─" * 80)

for r in results:
    status = "✓ Gefunden" if r['found'] else "✗ Nicht gefunden"
    pos_str = f"{r['position']}/30" if r['position'] <= 30 else "Nicht in Top-30"
    score_str = f"{r['score']:.6f}" if r['score'] > 0 else "N/A"
    
    print(f"{r['model']:<25} {r['dim']:<6} {pos_str:<15} {score_str:<12} {status}")

print("\n" + "=" * 80)
print("=== EMPFEHLUNG ===")
print("=" * 80)

if results and results[0]['found']:
    best = results[0]
    print(f"\n🏆 BESTES MODELL: {best['model']}")
    print(f"   Position: {best['position']}/30")
    print(f"   Score: {best['score']:.6f}")
    
    if best['position'] <= 3:
        print(f"\n✓ EXZELLENT! Gold KB in Top-3 - Kein Reranking nötig!")
    elif best['position'] <= 10:
        print(f"\n✓ GUT! Gold KB in Top-10 - Mit Reranking noch besser")
    else:
        print(f"\n⚠️  MITTEL: Gold KB gefunden, aber nicht in Top-10")
        print(f"   → Reranking wird empfohlen")
    
    print(f"\n💡 Um dieses Modell zu nutzen:")
    print(f"   1. In .env setzen: EMBEDDING_MODEL={best['model']}")
    print(f"   2. In .env setzen: EMBEDDING_DIM={best['dim']}")
    print(f"   3. Qdrant Collection neu-embedden:")
    print(f"      python src/rag_csv/ingest/kb.py")
else:
    print("\n⚠️  Kein Modell konnte Gold KB in Top-30 finden!")
    print("   → Daten müssen mit besserem Modell neu-embedded werden")

print("\n" + "=" * 80)
