#!/usr/bin/env python3
"""
Thread/Profile Test for Ollama endpoints.

Runs the same prompt across low/mid/high profiles with fixed thread settings,
repeats each run 3 times, logs detailed results, captures CPU usage from metrics
endpoints during generation, and writes aggregated means.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import requests

# Allow imports from src/ when script is executed from project root.
ROOT = Path(__file__).resolve().parent


PROMPT = "Warum ist der Himmel Blau?"
BASELINE_MODELS = [
    "llama3.1:8b-instruct-q4_K_M",
    "granite3.1-dense:8b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q4_K_M",
]
RUNS_PER_CONFIG = 3
OUTPUT_DIR = ROOT / "output" / "profile-thread-test"
TIMEOUT_SECONDS = 300
DETERMINISTIC_OPTIONS: Dict[str, Any] = {
    "temperature": 0.0,
    "seed": 42,
    "top_p": 1.0,
    "top_k": 1,
    "repeat_penalty": 1.0,
}

PROFILE_THREADS: Dict[str, List[int]] = {
    "low": [4],
    "mid": [6, 7, 8],
    "high": [4, 8, 12, 14, 16],
}


@dataclass
class MetricsSample:
    timestamp: str
    snapshot_type: str
    profile: str
    thread_count: int
    run_index: int
    cpu_system_percent: Optional[float]
    ollama_proc_cpu_percent: Optional[float]
    ram_system_percent: Optional[float]
    host: Optional[str]
    raw: str


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _profile_url(profile: str) -> str:
    base = _env("OLLAMA_URL")
    if profile == "low":
        return _env("OLLAMA_URL_LOW_PROFILE") or base
    if profile == "mid":
        return _env("OLLAMA_URL_MID_PROFILE") or base
    if profile == "high":
        return _env("OLLAMA_URL_HIGH_PROFILE") or base
    return base


def _profile_metrics_endpoint(profile: str) -> str:
    if profile == "low":
        return _env("OLLAMA_URL_LOW_METRICS_SERVICE_ENDPOINT")
    if profile == "mid":
        return _env("OLLAMA_URL_MID_METRICS_SERVICE_ENDPOINT")
    if profile == "high":
        return _env("OLLAMA_URL_HIGH_METRICS_SERVICE_ENDPOINT")
    return ""


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _fetch_metrics(endpoint: str) -> Dict[str, Any]:
    if not endpoint:
        return {"error": "NO_ENDPOINT"}
    try:
        response = requests.get(endpoint, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _sample_metrics_during_call(
    endpoint: str,
    interval: float,
    stop_event: threading.Event,
    profile: str,
    thread_count: int,
    run_index: int,
    sink: List[MetricsSample],
) -> None:
    while not stop_event.is_set():
        _capture_metrics_sample(
            endpoint=endpoint,
            profile=profile,
            thread_count=thread_count,
            run_index=run_index,
            snapshot_type="continuous",
            sink=sink,
        )
        stop_event.wait(interval)


def _capture_metrics_sample(
    endpoint: str,
    profile: str,
    thread_count: int,
    run_index: int,
    snapshot_type: str,
    sink: List[MetricsSample],
) -> None:
    payload = _fetch_metrics(endpoint)
    sink.append(
        MetricsSample(
            timestamp=datetime.now().isoformat(),
            snapshot_type=snapshot_type,
            profile=profile,
            thread_count=thread_count,
            run_index=run_index,
            cpu_system_percent=_safe_float(payload.get("cpu_system_percent")),
            ollama_proc_cpu_percent=_safe_float(payload.get("ollama_proc_cpu_percent")),
            ram_system_percent=_safe_float(payload.get("ram_system_percent")),
            host=payload.get("host"),
            raw=json.dumps(payload, ensure_ascii=True),
        )
    )


def _call_ollama(url: str, model: str, prompt: str, thread_count: int) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_thread": thread_count,
                    **DETERMINISTIC_OPTIONS,
                },
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        latency_s = time.perf_counter() - started
        return {"ok": True, "data": data, "latency_s": latency_s, "error": ""}
    except Exception as exc:  # noqa: BLE001
        latency_s = time.perf_counter() - started
        return {"ok": False, "data": {}, "latency_s": latency_s, "error": str(exc)}


def _tokens_per_second(ollama_json: Dict[str, Any]) -> Optional[float]:
    eval_count = ollama_json.get("eval_count")
    eval_duration = ollama_json.get("eval_duration")
    if eval_count is None or eval_duration in (None, 0):
        return None
    try:
        return float(eval_count) / (float(eval_duration) / 1e9)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return mean(valid) if valid else None


def _min_max(values: List[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    return (min(values), max(values))


def _norm(value: Optional[float], vmin: float, vmax: float, higher_better: bool) -> Optional[float]:
    if value is None:
        return None
    if vmax == vmin:
        return 1.0
    scaled = (value - vmin) / (vmax - vmin)
    if higher_better:
        return scaled
    return 1.0 - scaled


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_env_fallback(env_path: Path) -> None:
    """Load .env values if python-dotenv is unavailable in the runtime."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.split("#", 1)[0].strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _write_outputs(run_rows: List[Dict[str, Any]], metric_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    grouped: Dict[tuple[str, str, int], List[Dict[str, Any]]] = {}
    for row in run_rows:
        key = (str(row["model"]), str(row["profile"]), int(row["threads"]))
        grouped.setdefault(key, []).append(row)

    for (model, profile, thread_count), rows in sorted(grouped.items()):
        latency_values = [_safe_float(r.get("latency_s")) for r in rows]
        tps_values = [_safe_float(r.get("tokens_per_second")) for r in rows]
        cpu_values = [_safe_float(r.get("ollama_proc_cpu_percent_mean_during_run")) for r in rows]
        ram_values = [_safe_float(r.get("ram_system_percent_mean_during_run")) for r in rows]
        success_rate = sum(1 for r in rows if r.get("success")) / len(rows)

        summary_rows.append(
            {
                "model": model,
                "profile": profile,
                "threads": thread_count,
                "runs": len(rows),
                "success_rate": success_rate,
                "latency_s_mean": _mean_or_none(latency_values),
                "tokens_per_second_mean": _mean_or_none(tps_values),
                "ollama_proc_cpu_percent_mean": _mean_or_none(cpu_values),
                "ram_system_percent_mean": _mean_or_none(ram_values),
            }
        )

    latency_pool = [r["latency_s_mean"] for r in summary_rows if r.get("latency_s_mean") is not None]
    tps_pool = [r["tokens_per_second_mean"] for r in summary_rows if r.get("tokens_per_second_mean") is not None]
    lat_min, lat_max = _min_max([float(v) for v in latency_pool])
    tps_min, tps_max = _min_max([float(v) for v in tps_pool])
    has_tps = len(tps_pool) > 0

    ranking_rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        latency_score = _norm(_safe_float(row.get("latency_s_mean")), lat_min, lat_max, higher_better=False)
        tps_score = _norm(_safe_float(row.get("tokens_per_second_mean")), tps_min, tps_max, higher_better=True)

        if not has_tps and latency_score is not None:
            composite = latency_score
        elif latency_score is None or tps_score is None:
            composite = None
        else:
            composite = 0.6 * tps_score + 0.4 * latency_score

        ranking_rows.append(
            {
                "model": row["model"],
                "profile": row["profile"],
                "threads": row["threads"],
                "runs": row["runs"],
                "success_rate": row["success_rate"],
                "latency_s_mean": row["latency_s_mean"],
                "tokens_per_second_mean": row["tokens_per_second_mean"],
                "ollama_proc_cpu_percent_mean": row["ollama_proc_cpu_percent_mean"],
                "ram_system_percent_mean": row["ram_system_percent_mean"],
                "latency_score_0_1": latency_score,
                "throughput_score_0_1": tps_score,
                "composite_score": composite,
            }
        )

    ranking_rows.sort(
        key=lambda r: (r["composite_score"] is not None, r["composite_score"]),
        reverse=True,
    )
    for idx, row in enumerate(ranking_rows, start=1):
        row["rank"] = idx

    _write_csv(
        OUTPUT_DIR / "runs.csv",
        run_rows,
        [
            "timestamp",
            "model",
            "profile",
            "threads",
            "run",
            "url",
            "metrics_endpoint",
            "success",
            "error",
            "latency_s",
            "tokens_per_second",
            "eval_count",
            "eval_duration_ns",
            "total_duration_ns",
            "prompt_eval_count",
            "cpu_system_percent_mean_during_run",
            "ollama_proc_cpu_percent_mean_during_run",
            "ram_system_percent_mean_during_run",
            "response_text",
        ],
    )

    _write_csv(
        OUTPUT_DIR / "metrics_samples.csv",
        metric_rows,
        [
            "timestamp",
            "snapshot_type",
            "model",
            "profile",
            "threads",
            "run",
            "cpu_system_percent",
            "ollama_proc_cpu_percent",
            "ram_system_percent",
            "host",
            "raw",
        ],
    )

    _write_csv(
        OUTPUT_DIR / "summary.csv",
        summary_rows,
        [
            "model",
            "profile",
            "threads",
            "runs",
            "success_rate",
            "latency_s_mean",
            "tokens_per_second_mean",
            "ollama_proc_cpu_percent_mean",
            "ram_system_percent_mean",
        ],
    )

    _write_csv(
        OUTPUT_DIR / "ranking.csv",
        ranking_rows,
        [
            "rank",
            "model",
            "profile",
            "threads",
            "runs",
            "success_rate",
            "latency_s_mean",
            "tokens_per_second_mean",
            "ollama_proc_cpu_percent_mean",
            "ram_system_percent_mean",
            "latency_score_0_1",
            "throughput_score_0_1",
            "composite_score",
        ],
    )

    md_lines = [
        "# Profile Thread Test Summary",
        "",
        f"- Timestamp: {datetime.now().isoformat()}",
        f"- Prompt: `{PROMPT}`",
        f"- Models: `{', '.join(BASELINE_MODELS)}`",
        f"- Runs per config: {RUNS_PER_CONFIG}",
        "",
        "## Resolved Endpoints",
        "",
        "| Profile | Ollama URL | Metrics Endpoint |",
        "|---|---|---|",
    ]
    for profile in PROFILE_THREADS:
        md_lines.append(
            f"| {profile} | `{_profile_url(profile) or 'MISSING'}` | "
            f"`{_profile_metrics_endpoint(profile) or 'MISSING'}` |"
        )
    md_lines.extend(
        [
            "",
            "## Means per Profile/Threads",
            "",
            "| Model | Profile | Threads | Runs | Success Rate | Latency Mean (s) | Tokens/s Mean | Ollama CPU% Mean | RAM% Mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        md_lines.append(
            f"| {row['model']} | {row['profile']} | {row['threads']} | {row['runs']} | "
            f"{row['success_rate']:.2f} | "
            f"{(row['latency_s_mean'] or 0):.3f} | "
            f"{(row['tokens_per_second_mean'] or 0):.2f} | "
            f"{(row['ollama_proc_cpu_percent_mean'] or 0):.2f} | "
            f"{(row['ram_system_percent_mean'] or 0):.2f} |"
        )

    md_lines.extend(
        [
            "",
            "## Ranking (Throughput + Latency)",
            "",
            (
                "Composite = 60% Throughput-Score + 40% Latency-Score (beide 0..1 normalisiert)."
                if has_tps
                else "Composite = Latency-Score (Fallback, da keine Tokens/s verfügbar)."
            ),
            "",
            "| Rank | Model | Profile | Threads | Composite | TPS Score | Latency Score |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in ranking_rows:
        md_lines.append(
            f"| {row['rank']} | {row['model']} | {row['profile']} | {row['threads']} | "
            f"{(row['composite_score'] or 0):.4f} | "
            f"{(row['throughput_score_0_1'] or 0):.4f} | "
            f"{(row['latency_score_0_1'] or 0):.4f} |"
        )

    (OUTPUT_DIR / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    return ranking_rows


def main() -> None:
    _load_env_fallback(ROOT / ".env")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    removed_files = 0
    for old_file in OUTPUT_DIR.glob("*"):
        if old_file.is_file():
            old_file.unlink()
            removed_files += 1
    metric_interval = float(_env("OLLAMA_METRICS_SERVICE_INTERVAL", "1") or "1")

    run_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []

    print(f"Output: {OUTPUT_DIR}")
    print(f"Cleaned old output files: {removed_files}")
    print(f"Prompt: {PROMPT}")
    print(f"Models: {', '.join(BASELINE_MODELS)}")
    print("Resolved profile endpoints:")
    for profile in PROFILE_THREADS:
        print(
            f" - {profile}: ollama={_profile_url(profile) or 'MISSING'} "
            f"metrics={_profile_metrics_endpoint(profile) or 'MISSING'}"
        )

    for profile, thread_list in PROFILE_THREADS.items():
        base_url = _profile_url(profile)
        metrics_endpoint = _profile_metrics_endpoint(profile)
        print(
            f"[DEBUG] profile={profile} ollama_url={base_url or 'MISSING'} "
            f"metrics_endpoint={metrics_endpoint or 'MISSING'} threads={thread_list}"
        )
        if not base_url:
            print(f"[WARN] Profile '{profile}' hat keine URL, wird übersprungen.")
            continue

        for model in BASELINE_MODELS:
            print(f"[DEBUG] profile={profile} model={model}")
            for thread_count in thread_list:
                for run_index in range(1, RUNS_PER_CONFIG + 1):
                    print(
                        f"Run -> profile={profile} model={model} "
                        f"threads={thread_count} run={run_index}"
                    )

                    samples: List[MetricsSample] = []
                    _capture_metrics_sample(
                        endpoint=metrics_endpoint,
                        profile=profile,
                        thread_count=thread_count,
                        run_index=run_index,
                        snapshot_type="pre",
                        sink=samples,
                    )
                    stop_event = threading.Event()
                    sampler = threading.Thread(
                        target=_sample_metrics_during_call,
                        args=(
                            metrics_endpoint,
                            metric_interval,
                            stop_event,
                            profile,
                            thread_count,
                            run_index,
                            samples,
                        ),
                        daemon=True,
                    )
                    sampler.start()

                    result = _call_ollama(base_url, model, PROMPT, thread_count)

                    stop_event.set()
                    sampler.join(timeout=metric_interval + 2)
                    _capture_metrics_sample(
                        endpoint=metrics_endpoint,
                        profile=profile,
                        thread_count=thread_count,
                        run_index=run_index,
                        snapshot_type="post",
                        sink=samples,
                    )

                    # Ensure at least one sample per run (if request was too fast).
                    if not samples:
                        _capture_metrics_sample(
                            endpoint=metrics_endpoint,
                            profile=profile,
                            thread_count=thread_count,
                            run_index=run_index,
                            snapshot_type="fallback",
                            sink=samples,
                        )

                    sampled_during_generation = [s for s in samples if s.snapshot_type == "continuous"]
                    run_scope_samples = sampled_during_generation if sampled_during_generation else samples

                    for s in samples:
                        metric_rows.append(
                            {
                                "timestamp": s.timestamp,
                                "snapshot_type": s.snapshot_type,
                                "model": model,
                                "profile": s.profile,
                                "threads": s.thread_count,
                                "run": s.run_index,
                                "cpu_system_percent": s.cpu_system_percent,
                                "ollama_proc_cpu_percent": s.ollama_proc_cpu_percent,
                                "ram_system_percent": s.ram_system_percent,
                                "host": s.host,
                                "raw": s.raw,
                            }
                        )

                    ollama_data = result["data"] if result["ok"] else {}
                    tps = _tokens_per_second(ollama_data)
                    mean_cpu_system = _mean_or_none([s.cpu_system_percent for s in run_scope_samples])
                    mean_cpu_proc = _mean_or_none([s.ollama_proc_cpu_percent for s in run_scope_samples])
                    mean_ram_system = _mean_or_none([s.ram_system_percent for s in run_scope_samples])

                    run_rows.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "model": model,
                            "profile": profile,
                            "threads": thread_count,
                            "run": run_index,
                            "url": base_url,
                            "metrics_endpoint": metrics_endpoint,
                            "success": result["ok"],
                            "error": result["error"],
                            "latency_s": result["latency_s"],
                            "tokens_per_second": tps,
                            "eval_count": ollama_data.get("eval_count"),
                            "eval_duration_ns": ollama_data.get("eval_duration"),
                            "total_duration_ns": ollama_data.get("total_duration"),
                            "prompt_eval_count": ollama_data.get("prompt_eval_count"),
                            "cpu_system_percent_mean_during_run": mean_cpu_system,
                            "ollama_proc_cpu_percent_mean_during_run": mean_cpu_proc,
                            "ram_system_percent_mean_during_run": mean_ram_system,
                            "response_text": ollama_data.get("response", "") or "",
                        }
                    )
                    ranking_rows = _write_outputs(run_rows, metric_rows)
                    print(
                        f"[CHECKPOINT] gespeichert nach "
                        f"profile={profile} model={model} threads={thread_count} run={run_index}"
                    )

    ranking_rows = _write_outputs(run_rows, metric_rows)
    best = ranking_rows[0] if ranking_rows else None
    print("Fertig. Dateien geschrieben:")
    print(f" - {OUTPUT_DIR / 'runs.csv'}")
    print(f" - {OUTPUT_DIR / 'metrics_samples.csv'}")
    print(f" - {OUTPUT_DIR / 'summary.csv'}")
    print(f" - {OUTPUT_DIR / 'ranking.csv'}")
    print(f" - {OUTPUT_DIR / 'summary.md'}")
    if best:
        best_score = best.get("composite_score")
        best_score_str = f"{best_score:.4f}" if isinstance(best_score, (float, int)) else "n/a"
        print(
            "Best config: "
            f"model={best['model']} profile={best['profile']} threads={best['threads']} "
            f"composite={best_score_str}"
        )


if __name__ == "__main__":
    main()
