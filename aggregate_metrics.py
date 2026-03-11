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


def _increase_csv_field_limit() -> None:
    """Erhöht das CSV-Feldlimit robust für große Prompt/Response-Spalten."""
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size = max_size // 10

def safe_float(value):
    """Konvertiert Wert sicher zu float."""
    try:
        if value is None or value == "":
            return None
        return float(value)
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

def safe_min(values):
    """Berechnet Minimum, ignoriert None."""
    valid = [v for v in values if v is not None]
    return min(valid) if valid else None

def safe_max(values):
    """Berechnet Maximum, ignoriert None."""
    valid = [v for v in values if v is not None]
    return max(valid) if valid else None


def _find_embedding_metrics_file(experiment_id: str) -> Path | None:
    """Findet die Embedding-Metrics-Datei für ein Experiment."""
    metrics_dir = Path("output/metrics") / experiment_id
    if not metrics_dir.exists():
        return None

    exact = metrics_dir / f"embedding_metrics_{experiment_id}.csv"
    if exact.exists():
        return exact

    candidates = sorted(metrics_dir.glob("embedding_metrics_*.csv"))
    return candidates[0] if candidates else None


def load_embedding_metrics_by_trace(experiment_id: str) -> dict:
    """
    Lädt Embedding-Metriken und aggregiert sie pro trace_id.

    Returns:
        dict: trace_id -> aggregierte Embedding-Metriken
    """
    metrics_file = _find_embedding_metrics_file(experiment_id)
    if not metrics_file:
        return {}

    by_trace = defaultdict(list)
    with open(metrics_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trace_id = (row.get("trace_id") or "").strip()
            if trace_id:
                by_trace[trace_id].append(row)

    trace_metrics = {}
    for trace_id, rows in by_trace.items():
        ts_values = [safe_float(r.get("ts_epoch")) for r in rows]
        cpu_values = [safe_float(r.get("cpu_usage")) for r in rows]
        mem_values = [safe_float(r.get("memory_usage")) for r in rows]
        proc_cpu_values = [safe_float(r.get("ollama_proc_cpu_percent")) for r in rows]
        proc_rss_values = [safe_float(r.get("ollama_proc_rss_mb")) for r in rows]

        ts_min = safe_min(ts_values)
        ts_max = safe_max(ts_values)
        duration_ms = (ts_max - ts_min) * 1000 if ts_min is not None and ts_max is not None else None

        trace_metrics[trace_id] = {
            "embedding_metric_samples": len(rows),
            "embedding_duration_ms": duration_ms,
            "embedding_cpu_mean": safe_mean(cpu_values),
            "embedding_cpu_max": safe_max(cpu_values),
            "embedding_memory_mean": safe_mean(mem_values),
            "embedding_memory_max": safe_max(mem_values),
            "embedding_proc_cpu_mean": safe_mean(proc_cpu_values),
            "embedding_proc_cpu_max": safe_max(proc_cpu_values),
            "embedding_proc_rss_mb_mean": safe_mean(proc_rss_values),
            "embedding_proc_rss_mb_max": safe_max(proc_rss_values),
        }

    return trace_metrics


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
        embedding_samples = [safe_float(r.get('embedding_metric_samples')) for r in rows]
        embedding_duration = [safe_float(r.get('embedding_duration_ms')) for r in rows]
        embedding_cpu_mean = [safe_float(r.get('embedding_cpu_mean')) for r in rows]
        embedding_cpu_max = [safe_float(r.get('embedding_cpu_max')) for r in rows]
        embedding_memory_mean = [safe_float(r.get('embedding_memory_mean')) for r in rows]
        embedding_memory_max = [safe_float(r.get('embedding_memory_max')) for r in rows]
        embedding_proc_cpu_mean = [safe_float(r.get('embedding_proc_cpu_mean')) for r in rows]
        embedding_proc_cpu_max = [safe_float(r.get('embedding_proc_cpu_max')) for r in rows]
        embedding_proc_rss_mean = [safe_float(r.get('embedding_proc_rss_mb_mean')) for r in rows]
        embedding_proc_rss_max = [safe_float(r.get('embedding_proc_rss_mb_max')) for r in rows]
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
            'recall_min': safe_min(recall_values),
            'recall_max': safe_max(recall_values),
            'ndcg_mean': safe_mean(ndcg_values),
            'ndcg_std': safe_stdev(ndcg_values),
            'ndcg_min': safe_min(ndcg_values),
            'ndcg_max': safe_max(ndcg_values),
            'latency_mean_ms': safe_mean(latency_values),
            'latency_median_ms': safe_median(latency_values),
            'latency_std_ms': safe_stdev(latency_values),
            'wall_time_mean_ms': safe_mean(wall_time_values),
            'wall_time_median_ms': safe_median(wall_time_values),
            'tokens_s_mean': safe_mean(tokens_s_values),
            'tokens_s_median': safe_median(tokens_s_values),
            'prompt_tokens_mean': safe_mean(prompt_tokens),
            'completion_tokens_mean': safe_mean(completion_tokens),
            'embedding_samples_mean': safe_mean(embedding_samples),
            'embedding_duration_mean_ms': safe_mean(embedding_duration),
            'embedding_duration_median_ms': safe_median(embedding_duration),
            'embedding_cpu_mean': safe_mean(embedding_cpu_mean),
            'embedding_cpu_peak_mean': safe_mean(embedding_cpu_max),
            'embedding_memory_mean': safe_mean(embedding_memory_mean),
            'embedding_memory_peak_mean': safe_mean(embedding_memory_max),
            'embedding_proc_cpu_mean': safe_mean(embedding_proc_cpu_mean),
            'embedding_proc_cpu_peak_mean': safe_mean(embedding_proc_cpu_max),
            'embedding_proc_rss_mb_mean': safe_mean(embedding_proc_rss_mean),
            'embedding_proc_rss_mb_peak_mean': safe_mean(embedding_proc_rss_max),
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
        embedding_samples = [safe_float(r.get('embedding_metric_samples')) for r in rows]
        embedding_duration = [safe_float(r.get('embedding_duration_ms')) for r in rows]
        embedding_cpu_mean = [safe_float(r.get('embedding_cpu_mean')) for r in rows]
        embedding_cpu_max = [safe_float(r.get('embedding_cpu_max')) for r in rows]
        embedding_memory_mean = [safe_float(r.get('embedding_memory_mean')) for r in rows]
        embedding_memory_max = [safe_float(r.get('embedding_memory_max')) for r in rows]
        embedding_proc_cpu_mean = [safe_float(r.get('embedding_proc_cpu_mean')) for r in rows]
        embedding_proc_cpu_max = [safe_float(r.get('embedding_proc_cpu_max')) for r in rows]
        embedding_proc_rss_mean = [safe_float(r.get('embedding_proc_rss_mb_mean')) for r in rows]
        embedding_proc_rss_max = [safe_float(r.get('embedding_proc_rss_mb_max')) for r in rows]
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
            'recall_min': safe_min(recall_values),
            'recall_max': safe_max(recall_values),
            'ndcg_mean': safe_mean(ndcg_values),
            'ndcg_std': safe_stdev(ndcg_values),
            'ndcg_min': safe_min(ndcg_values),
            'ndcg_max': safe_max(ndcg_values),
            'latency_mean_ms': safe_mean(latency_values),
            'latency_median_ms': safe_median(latency_values),
            'latency_std_ms': safe_stdev(latency_values),
            'latency_min_ms': safe_min(latency_values),
            'latency_max_ms': safe_max(latency_values),
            'wall_time_mean_ms': safe_mean(wall_time_values),
            'wall_time_median_ms': safe_median(wall_time_values),
            'tokens_s_mean': safe_mean(tokens_s_values),
            'tokens_s_median': safe_median(tokens_s_values),
            'prompt_tokens_mean': safe_mean(prompt_tokens),
            'completion_tokens_mean': safe_mean(completion_tokens),
            'embedding_samples_mean': safe_mean(embedding_samples),
            'embedding_duration_mean_ms': safe_mean(embedding_duration),
            'embedding_duration_median_ms': safe_median(embedding_duration),
            'embedding_duration_min_ms': safe_min(embedding_duration),
            'embedding_duration_max_ms': safe_max(embedding_duration),
            'embedding_cpu_mean': safe_mean(embedding_cpu_mean),
            'embedding_cpu_peak_mean': safe_mean(embedding_cpu_max),
            'embedding_memory_mean': safe_mean(embedding_memory_mean),
            'embedding_memory_peak_mean': safe_mean(embedding_memory_max),
            'embedding_proc_cpu_mean': safe_mean(embedding_proc_cpu_mean),
            'embedding_proc_cpu_peak_mean': safe_mean(embedding_proc_cpu_max),
            'embedding_proc_rss_mb_mean': safe_mean(embedding_proc_rss_mean),
            'embedding_proc_rss_mb_peak_mean': safe_mean(embedding_proc_rss_max),
            'error_count': sum(errors)
        }
        
        if judge_values:
            result['judge_mean'] = safe_mean(judge_values)
            result['judge_std'] = safe_stdev(judge_values)
            result['judge_min'] = safe_min(judge_values)
            result['judge_max'] = safe_max(judge_values)
        
        results.append(result)
    
    return results


def process_runs_file(runs_file: Path, output_dir: Path):
    """Verarbeitet eine Runs-Datei und erstellt Aggregationen."""
    print(f"\n📊 Verarbeite: {runs_file.name}")
    
    try:
        _increase_csv_field_limit()
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

    embedding_by_trace = load_embedding_metrics_by_trace(experiment_id)
    if embedding_by_trace:
        for row in runs_data:
            row.update(embedding_by_trace.get(row.get("trace_id"), {}))
        print(f"   ✓ Embedding-Metriken korreliert: {len(embedding_by_trace)} trace_ids")
    else:
        print("   ℹ️ Keine Embedding-Metriken gefunden (ohne Embedding-Aggregation)")
    
    def _write_variant(variant_runs: list, variant_output_dir: Path, label: str) -> None:
        if not variant_runs:
            print(f"   ℹ️ Variante '{label}': keine Daten, übersprungen")
            return

        variant_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n   1️⃣ [{label}] Aggregiere pro Testcase...")
        testcase_agg = aggregate_by_testcase(variant_runs, experiment_id)
        testcase_file = variant_output_dir / f"case_agg_{experiment_id}.csv"
        if testcase_agg:
            with open(testcase_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(testcase_agg[0].keys()))
                writer.writeheader()
                writer.writerows(testcase_agg)
            print(f"   ✓ [{label}] Gespeichert: {testcase_file} ({len(testcase_agg)} Testcases)")

        print(f"\n   2️⃣ [{label}] Aggregiere pro Modell...")
        model_agg = aggregate_by_model(variant_runs, experiment_id)
        model_file = variant_output_dir / f"model_agg_{experiment_id}.csv"
        if model_agg:
            with open(model_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(model_agg[0].keys()))
                writer.writeheader()
                writer.writerows(model_agg)
            print(f"   ✓ [{label}] Gespeichert: {model_file} ({len(model_agg)} Modelle)")

            print(f"\n   📈 [{label}] Modell-Übersicht:")
            for row in model_agg:
                r = row['recall_mean']
                n = row['ndcg_mean']
                r_str = f"{r:.4f}" if r is not None else "N/A"
                n_str = f"{n:.4f}" if n is not None else "N/A"
                print(f"     • [{row['profile']:4}] {row['model']:40} → R@k: {r_str}, nDCG@k: {n_str}")

    # Legacy-Ausgabe bleibt bestehen (Kompatibilität zu bestehender Pipeline).
    _write_variant(runs_data, output_dir, "legacy-default")

    has_gpu_profile = any(str(row.get("profile", "")).strip().lower() == "gpu" for row in runs_data)
    if has_gpu_profile:
        # Neue, getrennte Ausgaben mit/ohne GPU unter output/<EXPERIMENT-ID>-neu/{default,no-gpu}.
        experiment_root = output_dir.parent / f"{experiment_id}-neu"
        default_dir = experiment_root / "default"
        no_gpu_dir = experiment_root / "no-gpu"

        _write_variant(runs_data, default_dir, "default")

        no_gpu_runs = [
            row for row in runs_data
            if str(row.get("profile", "")).strip().lower() != "gpu"
        ]
        _write_variant(no_gpu_runs, no_gpu_dir, "no-gpu")
    else:
        print("   ℹ️ Kein GPU-Profil enthalten - keine Variante default/no-gpu erzeugt")


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
