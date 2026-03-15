#!/usr/bin/env python3
"""Test für LLM-as-a-Judge System."""

import sys
sys.path.insert(0, 'src')

from rag_csv.utils.llm_judge import LLMJudge
from rag_csv.utils.answer_quality import AnswerQualityCalculator, QualityScores

print("=== Test 1: Answer Quality Calculator ===\n")

# Test mit beispielhaften Scores
calculator = AnswerQualityCalculator()

test_scores = QualityScores(
    faithfulness=4.0,
    relevance=5.0,
    completeness=3.0,
    fluency=4.5
)

result = calculator.calculate(test_scores)

print(f"Faithfulness: {result['faithfulness']}")
print(f"Relevance: {result['relevance']}")
print(f"Completeness: {result['completeness']}")
print(f"Fluency: {result['fluency']}")
print(f"\nRaw Score: {result['raw_score']:.4f}")
print(f"Normalized Score: {result['normalized_score']:.4f}")
print(f"Interpretation: {calculator.get_interpretation(result['raw_score'])}")

print("\n" + "="*60)
print("=== Test 2: LLM Judge (benötigt API Key in .env) ===\n")

# Prüfe ob API Key gesetzt ist
import os
api_key = os.getenv("LLM_JUDGE_API_KEY")

if not api_key:
    print("⚠️  LLM_JUDGE_API_KEY nicht in .env gesetzt!")
    print("   Überspringe LLM Judge Test.")
    print("\nUm den Test zu aktivieren:")
    print("1. Füge LLM_JUDGE_API_KEY in .env hinzu")
    print("2. Setze USE_LLM_JUDGE=true")
    sys.exit(0)

print("✓ API Key gefunden, teste LLM Judge...\n")

try:
    judge = LLMJudge()
    
    # Beispiel Ticket
    ticket_description = """
    Nach dem letzten Windows-Update können mehrere Mitarbeiter nicht mehr auf 
    den Netzwerkdrucker im 2. Stock zugreifen. Die Fehlermeldung lautet 
    "Der Drucker konnte nicht gefunden werden". Der Drucker ist im Netzwerk 
    sichtbar und reagiert auf Ping.
    """
    
    # Beispiel Kontext
    context = """
    [KB] Problem: Druckerzugriff nach Windows-Update fehlgeschlagen
    Lösung: 1. Öffnen Sie die Systemsteuerung -> Geräte und Drucker
    2. Entfernen Sie den Drucker
    3. Fügen Sie ihn neu hinzu über "Drucker hinzufügen"
    4. Wählen Sie den Netzwerkdrucker aus der Liste
    5. Installieren Sie ggf. die Treiber neu
    """
    
    # Beispiel Antwort (gut)
    good_answer = """
    **Problemanalyse:**
    Das Problem tritt nach einem Windows-Update auf und betrifft mehrere Benutzer, 
    was auf ein systemweites Problem hindeutet.
    
    **Lösungsschritte:**
    1. Öffnen Sie die Systemsteuerung und navigieren Sie zu "Geräte und Drucker"
    2. Entfernen Sie den betroffenen Netzwerkdrucker
    3. Klicken Sie auf "Drucker hinzufügen"
    4. Wählen Sie den Netzwerkdrucker aus der automatischen Erkennung
    5. Falls erforderlich, installieren Sie die Druckertreiber neu
    
    Falls das Problem weiterhin besteht, prüfen Sie bitte die Netzwerkverbindung 
    und kontaktieren Sie den IT-Support.
    """
    
    result = judge.evaluate(
        ticket_description=ticket_description,
        context=context,
        generated_answer=good_answer
    )
    
    if result["success"]:
        print("✓ LLM Judge Evaluation erfolgreich!\n")
        print(f"Faithfulness: {result['faithfulness']}")
        print(f"Relevance: {result['relevance']}")
        print(f"Completeness: {result['completeness']}")
        print(f"Fluency: {result['fluency']}")
        print(f"\nRaw Score: {result['raw_score']:.4f}")
        print(f"Normalized Score: {result['normalized_score']:.4f}")
        print(f"\nJudge Response:\n{result['judge_response'][:500]}...")
    else:
        print(f"✗ LLM Judge Fehler: {result['error']}")
        
except Exception as e:
    print(f"✗ Fehler beim Test: {e}")
    import traceback
    traceback.print_exc()
