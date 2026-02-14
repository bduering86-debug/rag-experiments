#!/usr/bin/env python3
"""
Aggregations-Skript für RAG-Evaluation Metriken.

Erstellt zwei Arten von Aggregationen:
1. Pro Testcase: Mittelwerte über alle Runs eines Testcases
2. Pro Modell: Mittelwerte über alle Testcases eines Modells (gesamt)

Zuordnung erfolgt über experiment_id.
"""

import csv
import statistics
from pathlib import Path
from collections import defaultdict
import sys

import sys

def safe_float(value):
    """Konvertiert Wert sicher zu float."""
    try:
        return float(value) if value and value != '' else None
    except (ValueError, TypeError):
        return None

def safe_mean(values):
    """Berechnet Mittelwert, ignoriert None."""
    valid = [v for v in values if v is not None]
    return statistics.mean(valid) if valid else None

def safe_stdev(values):
    """Berechnet Standardabweichung, ignoriert None."""
    valid = [v for v in values if v is not None]
    return statistics.stdev(valid) if len(valid) > 1 else 0.0

def safe_median(values):
    """Berechnet Median, ignoriert None."""
    valid = [v for v in values if v is not None]
    return statistics.median(valid) if valid else None


def aggregate_by_testcase(runs_data: list, experiment_id: str) -> list:
    """
    Aggregiert Metriken pro Testcase (über alle Runs).
    
    Args:
        runs_data: Liste von Dictionaries mit Run-Daten
        experiment_id: Experiment ID
        
    Returns:
        Liste von Dictionaries mit aggregierten Werten
    """
    # Gruppiere nach (experiment_id, profile, model, test_case_id)
    groups = defaultdict(list)
    
    for row in runs_data:
        key = (
            row['experiment_id'],
            row['profile'],
            row['model'],
            row['test_case_id']
        )
        groups[key].append(row)
    
    # Aggregiere jede Gruppe
    results = []
    for (exp_id, profile, model, test_id), rows in groups.items():
        # Sammle Werte
        recall_values = [safe_float(r['recall@k']) for r in rows]
        ndcg_values = [safe_float(r['ndcg@k']) for r in rows]
        latency_values = [safe_float(r['latency_ms']) for r in rows]
        wall_time_values = [safe_float(r['total_wall_time_ms']) for r in rows]
        tokens_s_values = [safe_float(r['tokens_per_s']) for r in rows]
        prompt_tokens = [safe_float(r['prompt_tokens']) for r in rows]
        completion_tokens = [safe_float(r['completion_tokens']) for r in rows]
        errors = [int(r.get('error_flag', 0)) for r in rows]
        
        # Judge optional
        judge_values = [safe_float(r.get('llm_judge_score')) for r in rows if 'llm_judge_score' in r]
        
        result = {
            'experiment_id': exp_id,
            'profile': profile,
            'model': model,
            'test_case_id': test_id,
            'n_runs': len(rows),
            'recall_mean': safe_mean(recall_values),
            'recall_std': safe_stdev(recall_values),
            'recall_min': min(r for r in recall_values if r is not None) if recall_values else None,
            'recall_max': max(r for r in recall_values if r is not None) if recall_values else None,
            'ndcg_mean': safe_mean(ndcg_values),
            'ndcg_std': safe_stdev(ndcg_values),
            'ndcg_min': min(n for n in ndcg_values if n is not None) if ndcg_values else None,
            'ndcg_max': max(n for n in ndcg_values if n is not None) if ndcg_values else None,
            'latency_mean_ms': safe_mean(latency_values),
            'latency_median_ms': safe_median(latency_values),
            'latency_std_ms': safe_stdev(latency_values),
            'wall_time_mean_ms': safe_mean(wall_time_values),
            'wall_time_median_ms': safe_median(wall_time_values),
            'tokens_s_mean': safe_mean(tokens_s_values),
            'tokens_s_median': safe_median(tokens_s_values),
            'prompt_tokens_mean': safe_mean(prompt_tokens),
            'completion_tokens_mean': safe_mean(completion_tokens),
            'error_count': sum(errors)
        }
        
        if judge_values:
            result['judge_mean'] = safe_mean(judge_values)
            result['judge_std'] = safe_stdev(judge_values)
        
        results.append(result)
    
    return results


def aggregate_by_model(runs_data: list, experiment_id: str) -> list:
    """
    Aggregiert Metriken pro Modell (über alle Testcases).
    
    Args:
        runs_data: Liste von Dictionaries mit Run-Daten
        experiment_id: Experiment ID
        
    Returns:
        Liste von Dictionaries mit aggregierten Werten
    """
    # Gruppiere nach (experiment_id, profile, model)
    groups = defaultdict(list)
    
    for row in runs_data:
        key = (
            row['experiment_id'],
            row['profile'],
            row['model']
        )
        groups[key].append(row)
    
    # Aggregiere jede Gruppe
    results = []
    for (exp_id, profile, model), rows in groups.items():
        # Sammle Werte
        recall_values = [safe_float(r['recall@k']) for r in rows]
        ndcg_values = [safe_float(r['ndcg@k']) for r in rows]
        latency_values = [safe_float(r['latency_ms']) for r in rows]
        wall_time_values = [safe_float(r['total_wall_time_ms']) for r in rows]
        tokens_s_values = [safe_float(r['tokens_per_s']) for r in rows]
        prompt_tokens = [safe_float(r['prompt_tokens']) for r in rows]
        completion_tokens = [safe_float(r['completion_tokens']) for r in rows]
        errors = [int(r.get('error_flag', 0)) for r in rows]
        
        # Judge optional
        judge_values = [safe_float(r.get('llm_judge_score')) for r in rows if 'llm_judge_score' in r]
        
        result = {
            'experiment_id': exp_id,
            'profile': profile,
            'model': model,
            'total_runs': len(rows),
            'recall_mean': safe_mean(recall_values),
            'recall_std': safe_stdev(recall_values),
            'recall_min': min(r for r in recall_values if r is not None) if recall_values else None,
            'recall_max': max(r for r in recall_values if r is not None) if recall_values else None,
            'ndcg_mean': safe_mean(ndcg_values),
            'ndcg_std': safe_stdev(ndcg_values),
            'ndcg_min': min(n for n in ndcg_values if n is not None) if ndcg_values else None,
            'ndcg_max': max(n for n in ndcg_values if n is not None) if ndcg_values else None,
            'latency_mean_ms': safe_mean(latency_values),
            'latency_median_ms': safe_median(latency_values),
            'latency_std_ms': safe_stdev(latency_values),
            'latency_min_ms': min(l for l in latency_values if l is not None) if latency_values else None,
            'latency_max_ms': max(l for l in latency_values if l is not None) if latency_values else None,
            'wall_time_mean_ms': safe_mean(wall_time_values),
            'wall_time_median_ms': safe_median(wall_time_values),
            'tokens_s_mean': safe_mean(tokens_s_values),
            'tokens_s_median': safe_median(tokens_s_values),
            'prompt_tokens_mean': safe_mean(prompt_tokens),
            'completion_tokens_mean': safe_mean(completion_tokens),
            'error_count': sum(errors)
        }
        
        if judge_values:
            result['judge_mean'] = safe_mean(judge_values)
            result['judge_std'] = safe_stdev(judge_values)
            result['judge_min'] = min(j for j in judge_values if j is not None) if judge_values else None
            result['judge_max'] = max(j for j in judge_values if j is not None) if judge_values else None
        
        results.append(result)
    
    return results


def process_runs_file(runs_file: Path, output_dir: Path):
    """Verarbeitet eine Runs-Datei und erstellt Aggregationen."""
    print(f"\n📊 Verarbeite: {runs_file.name}")
    
    try:
        with open(runs_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            runs_data = list(reader)
        print(f"   ✓ {len(runs_data)} Runs geladen")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        return
    
    if not runs_data:
        return
    
    experiment_id = runs_data[0]['experiment_id']
    print(f"   📋 Experiment ID: {experiment_id}")
    
    # Testcase-Aggregation
    print(f"\n   1️⃣ Aggregiere pro Testcase...")
    testcase_agg = aggregate_by_testcase(runs_data, experiment_id)
    
    testcase_file = output_dir / f"case_agg_{experiment_id}.csv"
    if testcase_agg:
        with open(testcase_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(testcase_agg[0].keys()))
            writer.writeheader()
            writer.writerows(testcase_agg)
        print(f"   ✓ Gespeichert: {testcase_file.name} ({len(testcase_agg)} Testcases)")
    
    # Modell-Aggregation
    print(f"\n   2️⃣ Aggregiere pro Modell...")
    model_agg = aggregate_by_model(runs_data, experiment_id)
    
    model_file = output_dir / f"model_agg_{experiment_id}.csv"
    if model_agg:
        with open(model_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(model_agg[0].keys()))
            writer.writeheader()
            writer.writerows(model_agg)
        print(f"   ✓ Gespeichert: {model_file.name} ({len(model_agg)} Modelle)")
        
        print(f"\n   📈 Modell-Übersicht:")
        for row in model_agg:
            r = row['recall_mean']
            n = row['ndcg_mean']
            r_str = f"{r:.4f}" if r is not None else "N/A"
            n_str = f"{n:.4f}" if n is not None else "N/A"
            print(f"     • [{row['profile']:4}] {row['model']:40} → R@k: {r_str}, nDCG@k: {n_str}")


def main():
    print("\n" + "="*70)
    print("RAG Evaluation - Metrics Aggregation")
    print("="*70)
    
    experiment_dir = Path("output/experiment")
    runs_files = sorted(experiment_dir.glob("runs_*.csv"))
    
    if not runs_files:
        print("❌ Keine runs_*.csv Dateien gefunden")
        return 1
    
    print(f"\n📁 Gefunden: {len(runs_files)} Runs-Dateien")
    
    for runs_file in runs_files:
        try:
            process_runs_file(runs_file, experiment_dir)
        except Exception as e:
            print(f"\n❌ Fehler bei {runs_file.name}: {e}")
            continue
    
    print("\n" + "="*70)
    print("✅ Aggregation abgeschlossen")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
