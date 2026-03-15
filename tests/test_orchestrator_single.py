#!/usr/bin/env python3
"""Quick Test für RAG Answer Orchestrator - nur 1 Testcase."""

import sys
sys.path.insert(0, 'src')  #

from rag_csv.config.settings import DataConfig, OllamaConfig, EvaluationConfig
from rag_csv.evaluation.rag_answer_orchestrator import RAGAnswerOrchestrator
from rag_csv.data.load_testcases import load_testcases, TESTCASES_FILE

# Evaluation Config laden
_eval_config = EvaluationConfig()

# Orchestrator erstellen
orchestrator = RAGAnswerOrchestrator(
    top_k=_eval_config.top_k,
    runs_per_testcase=1,
    output_dir="../output/evaluation_test"
)

# Nur ersten Testfall laden
df = load_testcases(TESTCASES_FILE)
testcase = df.iloc[0].to_dict()

print(f"Teste Testfall: {testcase['test_case_id']}")
print(f"Titel: {testcase['ticket_title']}")
print(f"Gold KB: {testcase['gold_kb_id']}\n")

# Evaluiere nur diesen einen Testfall
result = orchestrator.evaluate_testcase(
    testcase=testcase,
    profile="gpu",  # Manuell auf GPU-Profil setzen. GPU-Profil nutzen, da schnellere Testergebnisse ermöglicht werden.
    model="llama3.1:8b-instruct-q4_K_M",
    run=1
)

print("\n=== Ergebnis ===")
print(f"Test Case ID: {result['test_case_id']}")
print(f"Profile: {result['profile']}")
print(f"Model: {result['model']}")
print(f"Retrieved: {result['retrieved_count']} docs")
print(f"nDCG@{orchestrator.top_k}: {result['ndcg@k']:.4f}")
print(f"Recall@{orchestrator.top_k}: {result['recall@k']:.4f}")
print(f"Gold in Top-K: {result['gold_in_topk']}")

print(f"\n=== Latenz ===")
print(f"Gesamt: {result['total_latency']:.2f}s")
print(f"Retrieval: {result['retrieval_duration']:.2f}s")
print(f"LLM: {result['llm_duration']:.2f}s")

print(f"\n=== Token-Statistiken ===")
print(f"Prompt Tokens: {result['prompt_tokens']}")
print(f"Generated Tokens: {result['generated_tokens']}")
print(f"Total Tokens: {result['total_tokens']}")
print(f"Tokens/Sekunde: {result['tokens_per_second']:.1f}")

# LLM Judge Metriken (wenn aktiviert)
if 'judge_quality_score' in result:
    print(f"\n=== LLM Judge Bewertung ===")
    print(f"Faithfulness (F): {result.get('judge_faithfulness', 'N/A')}")
    print(f"Relevance (R): {result.get('judge_relevance', 'N/A')}")
    print(f"Completeness (C): {result.get('judge_completeness', 'N/A')}")
    print(f"Fluency (L): {result.get('judge_fluency', 'N/A')}")
    print(f"Quality Score (normalized): {result['judge_quality_score']:.4f}")
    
    # Begründung anzeigen (falls vorhanden)
    if 'judge_response' in result and result['judge_response']:
        print(f"\nBegründung:")
        # Nur ersten Teil der Begründung
        judge_resp = result['judge_response'][:300]
        print(f"{judge_resp}..." if len(result['judge_response']) > 300 else judge_resp)

print(f"\nAntwort:\n{result['response'][:500]}...")
