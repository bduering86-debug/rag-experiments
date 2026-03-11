#!/bin/bash
# Test verschiedene Embedding-Modelle durch Re-Embedding der KB

set -e

MODELS=(
    "nomic-embed-text:768"
    "mxbai-embed-large:1024"
    "all-minilm:384"
)

echo "================================================================================"
echo "=== EMBEDDING MODEL BENCHMARK ==="
echo "================================================================================"
echo ""
echo "Testet verschiedene Modelle durch komplettes Re-Embedding der KB Collection"
echo ""

# Backup der aktuellen .env
cp .env .env.backup
echo "✓ Backup von .env erstellt"

for model_config in "${MODELS[@]}"; do
    model_name=$(echo $model_config | cut -d: -f1)
    model_dim=$(echo $model_config | cut -d: -f2)
    
    echo ""
    echo "================================================================================"
    echo "=== Testing: $model_name ($model_dim dim) ==="
    echo "================================================================================"
    
    # Update .env
    sed -i "s/^EMBEDDING_MODEL=.*/EMBEDDING_MODEL=$model_name/" .env
    sed -i "s/^EMBEDDING_DIM=.*/EMBEDDING_DIM=$model_dim/" .env
    
    echo "✓ .env aktualisiert: $model_name ($model_dim dim)"
    
    # Qdrant Collection löschen
    echo "  1. Lösche alte Collection..."
    python3 -c "
from qdrant_client import QdrantClient
client = QdrantClient(url='http://localhost:6333')
try:
    client.delete_collection('knowledgebase')
    print('     ✓ Collection gelöscht')
except:
    print('     ℹ Collection existierte nicht')
"
    
    # KB neu-embedden
    echo "  2. Embedde KB mit $model_name..."
    cd src/rag_csv/ingest
    python kb.py > /dev/null 2>&1
    cd ../../../
    echo "     ✓ KB embedded"
    
    # Teste Retrieval
    echo "  3. Teste Retrieval..."
    result=$(python tests/analyze_gold_kb_scores.py 2>&1 | grep -A 5 "=== SCORE ANALYSE ===" | grep "Gold KB gefunden" || echo "Nicht gefunden")
    
    if [[ $result == *"Position"* ]]; then
        position=$(echo $result | grep -oP 'Position \K[0-9]+')
        score=$(echo $result | grep -oP 'Score: \K[0-9.]+')
        echo "     ✓ Gold KB Position: $position/30 (Score: $score)"
        
        # Speichere Ergebnis
        echo "$model_name|$model_dim|$position|$score" >> /tmp/embedding_benchmark_results.txt
    else
        echo "     ✗ Gold KB nicht in Top-30"
        echo "$model_name|$model_dim|31|0.0" >> /tmp/embedding_benchmark_results.txt
    fi
done

echo ""
echo "================================================================================"
echo "=== ERGEBNISSE ==="
echo "================================================================================"
echo ""
printf "%-25s %-8s %-15s %-12s\n" "Model" "Dim" "Gold Position" "Score"
echo "--------------------------------------------------------------------------------"

sort -t'|' -k3 -n /tmp/embedding_benchmark_results.txt | while IFS='|' read -r model dim pos score; do
    if [ "$pos" -le 30 ]; then
        printf "%-25s %-8s %-15s %-12s\n" "$model" "$dim" "$pos/30" "$score"
    else
        printf "%-25s %-8s %-15s %-12s\n" "$model" "$dim" "Nicht gefunden" "N/A"
    fi
done

# Bestes Modell finden
best_line=$(sort -t'|' -k3 -n /tmp/embedding_benchmark_results.txt | head -1)
best_model=$(echo $best_line | cut -d'|' -f1)
best_dim=$(echo $best_line | cut -d'|' -f2)
best_pos=$(echo $best_line | cut -d'|' -f3)
best_score=$(echo $best_line | cut -d'|' -f4)

echo ""
echo "================================================================================"
echo "=== BESTES MODELL ==="
echo "================================================================================"
echo ""
echo "🏆 $best_model ($best_dim dim)"
echo "   Gold Position: $best_pos/30"
echo "   Score: $best_score"
echo ""

if [ "$best_pos" -le 3 ]; then
    echo "✓ EXZELLENT! Gold KB in Top-3"
elif [ "$best_pos" -le 10 ]; then
    echo "✓ GUT! Gold KB in Top-10"
else
    echo "⚠️  MITTEL: Gold KB gefunden, aber nicht optimal"
fi

echo ""
echo "Möchtest du dieses Modell permanent nutzen?"
echo ""
echo "Dann belasse die aktuelle .env, oder stelle wieder her:"
echo "  mv .env.backup .env"
echo ""

# Cleanup
rm -f /tmp/embedding_benchmark_results.txt
