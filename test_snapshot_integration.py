#!/usr/bin/env python3
"""
Snapshot-Integration-Test ohne RAG-Orchestrator Abhängigkeiten
Testet ob Pre/Continuous/Post Snapshots korrekt erfasst werden
"""

import sys
sys.path.insert(0, '/home/bduering/rag_csv/src')

from rag_csv.utils.system_metrics_logger import SystemMetricsLogger
import time
from pathlib import Path
import csv

def test_dual_metrics_simulation():
    """Simuliert RAG-Evaluation mit zwei Metrics-Loggern (LLM + Embedding)"""
    
    experiment_id = 'test_snapshot_dual'
    trace_id = 'test-trace-dual-001'
    metrics_dir = Path('output/metrics') / experiment_id
    
    print('╔════════════════════════════════════════════════════════╗')
    print('║   Dual Metrics Logger Test (LLM + Embedding)         ║')
    print('╚════════════════════════════════════════════════════════╝')
    print()
    
    # Simuliere RAG Orchestrator Metriken (Remote LLM)
    print('1. LLM Metrics Logger (Remote Ollama - local profile für Test)')
    llm_logger = SystemMetricsLogger(
        experiment_id=experiment_id,
        output_dir=str(metrics_dir),
        profile='local',  # Für Test verwenden wir local statt remote
        file_prefix='system_metrics'
    )
    
    # PRE-Snapshot für LLM
    print('   ├─ PRE-Snapshot (Baseline vor Testfall)...')
    llm_logger.capture_snapshot(trace_id, snapshot_type='pre')
    
    # Start LLM Logging
    print('   ├─ Continuous Logging starten...')
    llm_logger.start_logging(trace_id)
    
    # Simuliere Embedding-Metriken (Lokal)
    print()
    print('2. Embedding Metrics Logger (Lokal)')
    embedding_logger = SystemMetricsLogger(
        experiment_id=experiment_id,
        output_dir=str(metrics_dir),
        profile='local',
        file_prefix='embedding_metrics'
    )
    
    # PRE-Snapshot für Embedding
    print('   ├─ PRE-Snapshot (Baseline vor Retrieval)...')
    embedding_logger.capture_snapshot(trace_id, snapshot_type='pre')
    
    # Start Embedding Logging
    print('   ├─ Continuous Logging starten...')
    embedding_logger.start_logging(trace_id)
    
    # Simuliere Retrieval-Zeit
    print('   ├─ Simuliere Retrieval (1.5s)...')
    time.sleep(1.5)
    
    # Stop Embedding Logging
    embedding_logger.stop_logging()
    
    # POST-Snapshot für Embedding
    print('   └─ POST-Snapshot (Baseline nach Retrieval)...')
    embedding_logger.capture_snapshot(trace_id, snapshot_type='post')
    
    # Simuliere LLM-Generierung
    print()
    print('3. Simuliere LLM-Generierung (2.0s)...')
    time.sleep(2.0)
    
    # Stop LLM Logging
    llm_logger.stop_logging()
    
    # POST-Snapshot für LLM
    print('   └─ POST-Snapshot (Baseline nach Testfall)...')
    llm_logger.capture_snapshot(trace_id, snapshot_type='post')
    
    # Analysiere Ergebnisse
    print()
    print('═══════════════════════════════════════════════════════')
    print('Ergebnisse:')
    print('═══════════════════════════════════════════════════════')
    
    csv_files = list(metrics_dir.glob('*.csv'))
    
    for csv_file in sorted(csv_files):
        with open(csv_file, 'r') as f:
            rows = list(csv.DictReader(f))
        
        pre = sum(1 for r in rows if r.get('snapshot_type') == 'pre')
        cont = sum(1 for r in rows if r.get('snapshot_type') == 'continuous')
        post = sum(1 for r in rows if r.get('snapshot_type') == 'post')
        
        print()
        print(f'{csv_file.name}:')
        print(f'  Pre-Snapshots (Baseline):    {pre}')
        print(f'  Continuous Metriken:          {cont}')
        print(f'  Post-Snapshots (Baseline):   {post}')
        print(f'  ────────────────────────────')
        print(f'  Total Einträge:              {len(rows)}')
        
        # Zeige Zeitverteilung
        if rows:
            first_ts = rows[0]['timestamp'].split('T')[1][:8]
            last_ts = rows[-1]['timestamp'].split('T')[1][:8]
            print(f'  Zeitspanne:                  {first_ts} → {last_ts}')
    
    print()
    print('═══════════════════════════════════════════════════════')
    print('✓ Integration Test erfolgreich!')
    print('═══════════════════════════════════════════════════════')

if __name__ == '__main__':
    test_dual_metrics_simulation()
