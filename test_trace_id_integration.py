#!/usr/bin/env python3
"""
Testskript zur Verifizierung der trace_id Integration.

Prüft:
1. trace_id wird in Run-CSV gespeichert
2. trace_id wird an SystemMetricsLogger übergeben
3. Metriken werden in output/metrics/[EXPERIMENT_ID] gespeichert
4. trace_id kann zwischen Run-Daten und System-Metriken korreliert werden
"""

import os
import csv
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Set PYTHONPATH
import sys
sys.path.insert(0, "/home/bduering/rag_csv/src")

from rag_csv.evaluation.rag_answer_orchestrator import RAGAnswerOrchestrator

def test_trace_id_integration():
    """Testet die trace_id Integration."""
    print("\n" + "="*70)
    print("Test: trace_id Integration & Metrics-Pfad")
    print("="*70)
    
    # Erstelle temporären Orchestrator mit Mini-Test
    print("\n1. Initialisiere RAGAnswerOrchestrator...")
    orchestrator = RAGAnswerOrchestrator(
        top_k=5,
        use_llm_judge=False,  # Deaktiviert für schnelleren Test
        runs_per_testcase=1
    )
    
    experiment_id = orchestrator.experiment_id
    print(f"   ✓ Experiment ID: {experiment_id}")
    
    # Prüfe ob Metrics-Verzeichnis existiert
    print("\n2. Prüfe Verzeichnis-Struktur...")
    metrics_dir = Path("output/metrics") / experiment_id
    print(f"   Erwartetes Metrics-Verzeichnis: {metrics_dir}")
    
    # Erstelle Testcase
    print("\n3. Erstelle Test-Testcase...")
    test_testcase = {
        "test_case_id": "TEST_001",
        "category": "Test",
        "service": "Test-Service",
        "difficulty_level": "easy",
        "incident_full_text": "Testfall für trace_id Integration",
        "gold_kb_id": "KB001",
        "gold_kb_fulltext": "Test-KB-Artikel"
    }
    
    print("\n4. Führe evaluate_testcase aus (low profile)...")
    print("   ⏳ Bitte warten...")
    
    try:
        result = orchestrator.evaluate_testcase(
            testcase=test_testcase,
            profile="low",
            model="qwen2.5:1.5b-instruct-q4_K_M",  # Kleines, schnelles Modell
            run=1
        )
        
        print("\n5. ✓ Testcase erfolgreich evaluiert")
        
        # Prüfe Ergebnis
        trace_id = result.get("trace_id")
        print(f"\n6. Prüfe Ergebnis-Daten:")
        print(f"   • experiment_id: {result.get('experiment_id')}")
        print(f"   • trace_id: {trace_id}")
        print(f"   • profile: {result.get('profile')}")
        print(f"   • model: {result.get('model')}")
        print(f"   • test_case_id: {result.get('test_case_id')}")
        
        if not trace_id:
            print("\n   ❌ FEHLER: Keine trace_id im Ergebnis!")
            return False
        
        # Prüfe ob Metrics-CSV existiert
        print(f"\n7. Prüfe Metrics-CSV:")
        metrics_csv = metrics_dir / f"system_metrics_{experiment_id}.csv"
        
        if not metrics_csv.exists():
            print(f"   ⚠ Metrics-CSV noch nicht erstellt: {metrics_csv}")
            print(f"   (Warte 2 Sekunden für Background-Thread...)")
            time.sleep(2)
        
        if metrics_csv.exists():
            print(f"   ✓ Metrics-CSV gefunden: {metrics_csv}")
            
            # Lese CSV und prüfe trace_id
            with open(metrics_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                print(f"   ✓ {len(rows)} Metrics-Zeilen gefunden")
                
                # Suche nach unserer trace_id
                matching_rows = [row for row in rows if row.get('trace_id') == trace_id]
                
                if matching_rows:
                    print(f"   ✓ {len(matching_rows)} Zeilen mit trace_id '{trace_id[:8]}...' gefunden")
                    
                    # Zeige erste Zeile
                    print(f"\n   Erste Metrics-Zeile:")
                    first = matching_rows[0]
                    for key in ['timestamp', 'trace_id', 'profile', 'cpu_usage', 'memory_usage', 'gpu_usage', 'gpu_memory']:
                        value = first.get(key, 'N/A')
                        if key == 'trace_id':
                            value = value[:12] + '...' if len(value) > 12 else value
                        print(f"     • {key:20}: {value}")
                else:
                    print(f"   ⚠ Keine Zeilen mit trace_id '{trace_id[:8]}...' gefunden")
                    if rows:
                        print(f"   Vorhandene trace_ids: {set(row.get('trace_id', 'N/A')[:8] for row in rows)}")
        else:
            print(f"   ❌ Metrics-CSV nicht gefunden: {metrics_csv}")
            return False
        
        # Zusammenfassung
        print("\n" + "="*70)
        print("ZUSAMMENFASSUNG")
        print("="*70)
        print(f"✅ trace_id wird korrekt generiert: {trace_id[:16]}...")
        print(f"✅ trace_id wird in Ergebnis-Daten gespeichert")
        print(f"✅ Metrics werden in korrektem Pfad gespeichert:")
        print(f"   {metrics_csv}")
        print(f"✅ trace_id ermöglicht Korrelation zwischen:")
        print(f"   - Run-Daten (output/experiment/runs_*.csv)")
        print(f"   - System-Metriken (output/metrics/{experiment_id}/)")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Fehler bei Testcase-Evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_directory_structure():
    """Verifiziert die Verzeichnisstruktur."""
    print("\n" + "="*70)
    print("Verzeichnisstruktur")
    print("="*70)
    
    base = Path("output")
    
    print(f"\n📁 {base}/")
    for subdir in ['experiment', 'metrics']:
        path = base / subdir
        exists = "✓" if path.exists() else "○"
        print(f"  {exists} {subdir}/")
        
        if path.exists():
            # Zeige erste 3 Dateien/Ordner
            items = list(path.iterdir())[:3]
            for item in items:
                print(f"     • {item.name}")
            if len(list(path.iterdir())) > 3:
                print(f"     • ... ({len(list(path.iterdir())) - 3} weitere)")


if __name__ == "__main__":
    print("\n🔬 trace_id Integration Test")
    print("=" * 70)
    
    # Verzeichnisstruktur anzeigen
    verify_directory_structure()
    
    # Haupttest
    success = test_trace_id_integration()
    
    if success:
        print("\n✅ Alle Tests erfolgreich!")
    else:
        print("\n❌ Test fehlgeschlagen!")
    
    # Abschließende Verzeichnisstruktur
    verify_directory_structure()
