#!/usr/bin/env python3
"""
Test-Script für das neue Retrieval-Logging-Format.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rag_csv.data.load_testcases import load_testcases
from rag_csv.evaluation.rag_answer_orchestrator import RAGAnswerOrchestrator

if __name__ == "__main__":
    print("🚀 Teste Retrieval-Logging...")
    
    # Orchestrator erstellen
    orchestrator = RAGAnswerOrchestrator(top_k=10, runs_per_testcase=1)
    
    # Testcases laden
    df = load_testcases("data/testcaes_quick.csv")
    testcases = df.to_dict(orient="records")
    
    print(f"✓ {len(testcases)} Testcases geladen\n")
    
    # Retrieval-Logging ausführen
    log_file = orchestrator.log_testcase_details(testcases)
    
    print(f"\n✅ Log erstellt: {log_file}")
    
    # Log-Datei anzeigen
    print("\n" + "="*80)
    print("LOG-INHALT:")
    print("="*80)
    with open(log_file, 'r', encoding='utf-8') as f:
        print(f.read())
