#!/usr/bin/env python3
"""Generate a compact overview for executed RAG runs."""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger("runs_overview")

RUNS_USECOLS = [
    "experiment_id",
    "trace_id",
    "profile",
    "model",
    "test_case_id",
    "repetition",
    "recall@k",
    "ndcg@k",
    "retrieval_score",
    "latency_ms",
    "tokens_per_s",
    "llm_judge_f",
    "llm_judge_r",
    "llm_judge_c",
    "llm_judge_l",
    "llm_judge_score",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "total_duration_s",
    "load_duration_s",
    "prompt_eval_duration_s",
    "eval_duration_s",
    "error_flag",
]

SCORE_COLS = ["profile", "model", "token_score_norm", "latency_score", "retrieval_score", "answer_score"]
METRIC_COLS = [
    "trace_id",
    "snapshot_type",
    "cpu_usage",
    "memory_usage",
    "ram_used_mb",
    "ollama_proc_cpu_percent",
    "ollama_proc_rss_mb",
    "gpu_usage",
    "gpu_memory",
    "gpu_clock_mhz",
    "gpu_mem_clock_mhz",
    "gpu_graphics_clock",
    "gpu_mem_clock",
    "vram_used_mb",
]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create overview tables and markdown summary for runs.")
    parser.add_argument("--base-dir", type=Path, default=Path("output/experiment"))
    parser.add_argument("--runs-file", type=Path, default=None)
    return parser.parse_args()


def _parse_key(path: Path, prefix: str) -> tuple[int, ...] | None:
    m = re.fullmatch(rf"{prefix}_([0-9_]+)\.csv", path.name)
    if not m:
        return None
    try:
        return tuple(int(p) for p in m.group(1).split("_"))
    except ValueError:
        return None


def pick_latest(base_dir: Path, prefix: str) -> Path | None:
    candidates: list[tuple[tuple[int, ...], Path]] = []
    for p in base_dir.glob(f"{prefix}_*.csv"):
        key = _parse_key(p, prefix)
        if key is not None:
            candidates.append((key, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_runs(runs_file: Path) -> pd.DataFrame:
    header = list(pd.read_csv(runs_file, nrows=0).columns)
    usecols = [c for c in RUNS_USECOLS if c in header]
    runs = pd.read_csv(runs_file, usecols=usecols)
    for col in [c for c in RUNS_USECOLS if c not in {"experiment_id", "trace_id", "profile", "model", "test_case_id"}]:
        if col in runs.columns:
            runs[col] = pd.to_numeric(runs[col], errors="coerce")
    if "error_flag" in runs.columns:
        runs["error_flag"] = runs["error_flag"].fillna(0)
    return runs


def load_latest_scores(base_dir: Path) -> pd.DataFrame:
    path = pick_latest(base_dir, "scores")
    if path is None:
        return pd.DataFrame(columns=SCORE_COLS)
    df = pd.read_csv(path)
    cols = [c for c in SCORE_COLS if c in df.columns]
    if not {"profile", "model"}.issubset(cols):
        return pd.DataFrame(columns=SCORE_COLS)
    out = df[cols].copy()
    for col in [c for c in SCORE_COLS if c not in {"profile", "model"} and c in out.columns]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    LOGGER.info("Using scores file: %s", path)
    return out


def _collect_metric_files(base_dir: Path, kind: str) -> list[Path]:
    files = list(base_dir.glob(f"{kind}_*.csv"))
    metrics_root = base_dir.parent / "metrics"
    if metrics_root.exists():
        files.extend(metrics_root.rglob(f"{kind}_*.csv"))
    dedup = {str(p.resolve()): p for p in files}
    return sorted(dedup.values())


def _agg_metrics(files: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file in files:
        try:
            header = list(pd.read_csv(file, nrows=0).columns)
            usecols = [c for c in METRIC_COLS if c in header]
            if "trace_id" not in usecols:
                continue
            frames.append(pd.read_csv(file, usecols=usecols))
        except Exception as exc:
            LOGGER.warning("Skipping metrics file %s (%s)", file, exc)
    if not frames:
        return pd.DataFrame(columns=["trace_id"])

    all_metrics = pd.concat(frames, ignore_index=True)
    if "snapshot_type" in all_metrics.columns:
        all_metrics = all_metrics[all_metrics["snapshot_type"].astype(str).str.lower() == "continuous"].copy()

    for col in [c for c in METRIC_COLS if c not in {"trace_id", "snapshot_type"} and c in all_metrics.columns]:
        all_metrics[col] = pd.to_numeric(all_metrics[col], errors="coerce")

    g = all_metrics.groupby("trace_id", dropna=False)
    out: dict[str, pd.Series] = {}
    for col in [c for c in METRIC_COLS if c not in {"trace_id", "snapshot_type"} and c in all_metrics.columns]:
        out[f"{col}_mean"] = g[col].mean()
        out[f"{col}_p95"] = g[col].quantile(0.95)
    return pd.DataFrame(out).reset_index() if out else pd.DataFrame(columns=["trace_id"])


def load_metric_aggs(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _agg_metrics(_collect_metric_files(base_dir, "system_metrics")), _agg_metrics(
        _collect_metric_files(base_dir, "embedding_metrics")
    )


def p95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def summarize_profile_model(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["profile", "model"], dropna=False)
    summary = g.agg(
        n_runs=("trace_id", "count"),
        n_errors=("error_flag", "sum"),
        latency_mean_s=("latency_ms", lambda s: s.mean() / 1000.0),
        latency_median_s=("latency_ms", lambda s: s.median() / 1000.0),
        latency_p95_s=("latency_ms", lambda s: p95(s) / 1000.0),
        tokens_per_s_mean=("tokens_per_s", "mean"),
        tokens_per_s_median=("tokens_per_s", "median"),
        ndcg_mean=("ndcg@k", "mean"),
        recall_mean=("recall@k", "mean"),
        retrieval_mean=("retrieval_score", "mean"),
        llm_judge_mean=("llm_judge_score", "mean"),
        total_tokens_mean=("total_tokens", "mean"),
    ).reset_index()

    optional_metric_cols = [
        "cpu_usage_mean",
        "cpu_usage_p95",
        "ram_used_mb_mean",
        "ram_used_mb_p95",
        "ollama_proc_cpu_percent_mean",
        "ollama_proc_cpu_percent_p95",
        "ollama_proc_rss_mb_mean",
        "ollama_proc_rss_mb_p95",
        "gpu_usage_mean",
        "gpu_usage_p95",
        "gpu_memory_mean",
        "gpu_memory_p95",
        "vram_used_mb_mean",
        "vram_used_mb_p95",
        "gpu_clock_mhz_mean",
        "gpu_clock_mhz_p95",
        "gpu_graphics_clock_mean",
        "gpu_graphics_clock_p95",
        "gpu_mem_clock_mhz_mean",
        "gpu_mem_clock_mhz_p95",
        "gpu_mem_clock_mean",
        "gpu_mem_clock_p95",
    ]
    for col in optional_metric_cols:
        if col in df.columns:
            summary[col] = g[col].mean().values

    summary["fail_rate_pct"] = (summary["n_errors"] / summary["n_runs"]) * 100
    summary = summary.sort_values(["profile", "llm_judge_mean"], ascending=[True, False])
    return summary


def summarize_testcases(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["test_case_id", "profile", "model"], dropna=False)
    out = g.agg(
        n_runs=("trace_id", "count"),
        n_errors=("error_flag", "sum"),
        latency_median_s=("latency_ms", lambda s: s.median() / 1000.0),
        llm_judge_mean=("llm_judge_score", "mean"),
        ndcg_mean=("ndcg@k", "mean"),
        recall_mean=("recall@k", "mean"),
    ).reset_index()
    out["fail_rate_pct"] = (out["n_errors"] / out["n_runs"]) * 100
    return out.sort_values(["test_case_id", "profile", "model"])


def summarize_global(df: pd.DataFrame) -> pd.DataFrame:
    row = {
        "n_runs": int(len(df)),
        "n_traces": int(df["trace_id"].nunique()) if "trace_id" in df.columns else 0,
        "n_profiles": int(df["profile"].nunique()) if "profile" in df.columns else 0,
        "n_models": int(df["model"].nunique()) if "model" in df.columns else 0,
        "n_test_cases": int(df["test_case_id"].nunique()) if "test_case_id" in df.columns else 0,
        "n_errors": int(df["error_flag"].fillna(0).sum()) if "error_flag" in df.columns else 0,
        "fail_rate_pct": float(df["error_flag"].fillna(0).mean() * 100) if "error_flag" in df.columns else 0.0,
        "latency_mean_s": float(df["latency_ms"].mean() / 1000.0) if "latency_ms" in df.columns else None,
        "latency_p95_s": float(df["latency_ms"].quantile(0.95) / 1000.0) if "latency_ms" in df.columns else None,
        "tokens_per_s_mean": float(df["tokens_per_s"].mean()) if "tokens_per_s" in df.columns else None,
        "llm_judge_mean": float(df["llm_judge_score"].mean()) if "llm_judge_score" in df.columns else None,
    }
    return pd.DataFrame([row])


def write_markdown(
    out_file: Path,
    variant_name: str,
    runs_file: Path,
    global_df: pd.DataFrame,
    profile_model_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    include_gpu_metrics: bool,
) -> None:
    g = global_df.iloc[0].to_dict()
    profiles = sorted(profile_model_df["profile"].dropna().astype(str).unique().tolist()) if not profile_model_df.empty else []
    models = sorted(profile_model_df["model"].dropna().astype(str).unique().tolist()) if not profile_model_df.empty else []
    top_judge = (
        profile_model_df.sort_values("llm_judge_mean", ascending=False).head(5)[["profile", "model", "llm_judge_mean"]]
        if not profile_model_df.empty
        else pd.DataFrame()
    )
    top_latency = (
        profile_model_df.sort_values("latency_median_s", ascending=True)
        .head(5)[["profile", "model", "latency_median_s", "fail_rate_pct"]]
        if not profile_model_df.empty
        else pd.DataFrame()
    )
    metric_overview = pd.DataFrame()
    if not profile_model_df.empty:
        metric_overview = profile_model_df[["profile", "model"]].copy()
        is_gpu = metric_overview["profile"].astype(str).str.lower() == "gpu"

        host_cpu = profile_model_df["cpu_usage_mean"] if "cpu_usage_mean" in profile_model_df else pd.Series([None] * len(metric_overview))
        host_ram = (
            profile_model_df["ram_used_mb_mean"] if "ram_used_mb_mean" in profile_model_df else pd.Series([None] * len(metric_overview))
        )
        gpu_util = profile_model_df["gpu_usage_mean"] if "gpu_usage_mean" in profile_model_df else pd.Series([None] * len(metric_overview))
        gpu_vram = profile_model_df.get("vram_used_mb_mean", profile_model_df.get("gpu_memory_mean", pd.Series([None] * len(metric_overview))))
        gpu_clock = profile_model_df.get(
            "gpu_clock_mhz_mean",
            profile_model_df.get("gpu_graphics_clock_mean", pd.Series([None] * len(metric_overview))),
        )
        proc_cpu = (
            profile_model_df["ollama_proc_cpu_percent_mean"]
            if "ollama_proc_cpu_percent_mean" in profile_model_df
            else pd.Series([None] * len(metric_overview))
        )

        metric_overview["utilization_mean"] = host_cpu
        metric_overview["memory_mean_mb"] = host_ram
        metric_overview["inference_proc_cpu_mean"] = proc_cpu
        metric_overview["metric_basis"] = "cpu/ram"

        if include_gpu_metrics:
            metric_overview.loc[is_gpu, "utilization_mean"] = gpu_util.loc[is_gpu]
            metric_overview.loc[is_gpu, "memory_mean_mb"] = gpu_vram.loc[is_gpu]
            metric_overview.loc[is_gpu, "metric_basis"] = "gpu_util/vram"
            if gpu_clock.notna().any():
                metric_overview["gpu_clock_mean_mhz"] = None
                metric_overview.loc[is_gpu, "gpu_clock_mean_mhz"] = gpu_clock.loc[is_gpu]

        metric_overview = metric_overview.sort_values(["profile", "model"])

    def df_block(df: pd.DataFrame) -> str:
        return "```text\n" + df.to_string(index=False) + "\n```"

    lines: list[str] = []
    lines.append(f"# Run Overview - {variant_name}")
    lines.append("")
    lines.append(f"- Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Runs file: `{runs_file}`")
    lines.append("")
    lines.append("## Global")
    lines.append(f"- Runs: {int(g['n_runs'])}")
    lines.append(f"- Traces: {int(g['n_traces'])}")
    lines.append(f"- Profiles: {int(g['n_profiles'])} ({', '.join(profiles) if profiles else '-'})")
    lines.append(f"- Models: {int(g['n_models'])} ({', '.join(models) if models else '-'})")
    lines.append(f"- Test cases: {int(g['n_test_cases'])}")
    lines.append(f"- Errors: {int(g['n_errors'])} ({g['fail_rate_pct']:.2f}%)")
    lines.append(f"- Latency mean/p95 (s): {g['latency_mean_s']:.3f} / {g['latency_p95_s']:.3f}")
    lines.append(f"- Tokens/s mean: {g['tokens_per_s_mean']:.2f}")
    lines.append(f"- LLM judge mean: {g['llm_judge_mean']:.4f}")
    lines.append("")
    lines.append("## Top-5 by LLM Judge")
    if top_judge.empty:
        lines.append("- No data")
    else:
        lines.append(df_block(top_judge))
    lines.append("")
    lines.append("## Top-5 Fastest (Median Latency in Sekunden)")
    if top_latency.empty:
        lines.append("- No data")
    else:
        lines.append(df_block(top_latency))
    lines.append("")
    lines.append("## Inference System Metrics")
    if metric_overview.empty:
        lines.append("- No system metric data")
    else:
        lines.append(df_block(metric_overview))
    lines.append("")
    lines.append("## Normalized Scores (latest scores file)")
    if scores_df.empty:
        lines.append("- No score data")
    else:
        score_avg = (
            scores_df.groupby(["profile", "model"], dropna=False)[
                [c for c in ["token_score_norm", "latency_score", "retrieval_score", "answer_score"] if c in scores_df]
            ]
            .mean()
            .reset_index()
        )
        lines.append(df_block(score_avg))
    lines.append("")
    lines.append("## Files")
    lines.append("- `overview_global.csv`")
    lines.append("- `overview_by_profile_model.csv`")
    lines.append("- `overview_by_testcase.csv`")
    lines.append("- `overview_failures.csv`")
    lines.append("- `overview_scores_latest.csv`")

    out_file.write_text("\n".join(lines), encoding="utf-8")


def make_output_root(base_dir: Path, runs_file: Path, runs_df: pd.DataFrame) -> Path:
    exp_id = None
    if "experiment_id" in runs_df.columns:
        vals = [v for v in runs_df["experiment_id"].dropna().astype(str).unique() if v and v.lower() != "nan"]
        if len(vals) == 1:
            exp_id = vals[0]
    match = re.search(r"runs_([0-9_]+)\.csv", runs_file.name)
    token = match.group(1) if match else "unknown"
    root = base_dir / "overview" / (exp_id or token)
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_variant(
    variant_name: str,
    runs_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    out_dir: Path,
    runs_file: Path,
    include_gpu_metrics: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    global_df = summarize_global(runs_df)
    profile_model_df = summarize_profile_model(runs_df)
    if not include_gpu_metrics:
        gpu_cols = [c for c in profile_model_df.columns if c.startswith("gpu_") or c.startswith("vram_")]
        if gpu_cols:
            profile_model_df = profile_model_df.drop(columns=gpu_cols)
    testcase_df = summarize_testcases(runs_df)
    failures = runs_df[runs_df["error_flag"].fillna(0) == 1].copy() if "error_flag" in runs_df.columns else pd.DataFrame()

    global_df.to_csv(out_dir / "overview_global.csv", index=False)
    profile_model_df.to_csv(out_dir / "overview_by_profile_model.csv", index=False)
    testcase_df.to_csv(out_dir / "overview_by_testcase.csv", index=False)
    failures.to_csv(out_dir / "overview_failures.csv", index=False)
    scores_df.to_csv(out_dir / "overview_scores_latest.csv", index=False)
    write_markdown(
        out_file=out_dir / "overview_summary.md",
        variant_name=variant_name,
        runs_file=runs_file,
        global_df=global_df,
        profile_model_df=profile_model_df,
        scores_df=scores_df,
        include_gpu_metrics=include_gpu_metrics,
    )
    LOGGER.info("[%s] Wrote overview files to %s", variant_name, out_dir)


def main() -> None:
    setup_logging()
    args = parse_args()
    base_dir: Path = args.base_dir
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    runs_file = args.runs_file if args.runs_file else pick_latest(base_dir, "runs")
    if runs_file is None or not runs_file.exists():
        raise FileNotFoundError(f"No valid runs file found in {base_dir}")
    LOGGER.info("Using runs file: %s", runs_file)

    runs_df = load_runs(runs_file)
    scores_df = load_latest_scores(base_dir)

    system_agg, embedding_agg = load_metric_aggs(base_dir)
    runs_df = runs_df.merge(system_agg, on="trace_id", how="left", suffixes=("", "_sys"))
    runs_df = runs_df.merge(embedding_agg, on="trace_id", how="left", suffixes=("", "_emb"))

    root = make_output_root(base_dir, runs_file, runs_df)
    run_variant(
        "all_profiles",
        runs_df.copy(),
        scores_df.copy(),
        root / "all_profiles",
        runs_file,
        include_gpu_metrics=True,
    )

    runs_no_gpu = runs_df[runs_df["profile"].astype(str).str.lower() != "gpu"].copy() if "profile" in runs_df else runs_df
    scores_no_gpu = (
        scores_df[scores_df["profile"].astype(str).str.lower() != "gpu"].copy() if "profile" in scores_df else scores_df
    )
    run_variant(
        "without_gpu_profile",
        runs_no_gpu,
        scores_no_gpu,
        root / "without_gpu_profile",
        runs_file,
        include_gpu_metrics=False,
    )

    LOGGER.info("Done. Output root: %s", root)


if __name__ == "__main__":
    main()
