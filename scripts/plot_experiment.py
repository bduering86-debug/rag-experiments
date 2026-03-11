#!/usr/bin/env python3
"""Create robust experiment plots for RAG CSV outputs.

Requirements:
- Read runs from output/experiment (latest runs_*.csv by timestamp in file name by default)
- Read all scores_*.csv (already normalized, never recompute)
- Read system_metrics_*.csv and embedding_metrics_*.csv, aggregate mean/p95 per trace_id, join via runs
- Export PNG figures with dpi=300
- Use pandas + matplotlib only
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_csv.utils.latency_score import LatencyScoreCalculator
from rag_csv.utils.token_score import TokenScoreCalculator


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
    "total_duration_s",
    "load_duration_s",
    "prompt_eval_duration_s",
    "eval_duration_s",
    "error_flag",
]

SCORES_COLS = [
    "profile",
    "model",
    "token_score_norm",
    "latency_score",
    "retrieval_score",
    "answer_score",
]

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
    "vram_used_mb",
    "gpu_clock_mhz",
    "gpu_graphics_clock",
    "gpu_mem_clock_mhz",
    "gpu_mem_clock",
]


LOGGER = logging.getLogger("plot_experiment")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RAG experiment plots from CSV files.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("output/experiment"),
        help="Base directory containing runs_*.csv and scores_*.csv",
    )
    parser.add_argument(
        "--runs-file",
        type=Path,
        default=None,
        help="Optional explicit runs CSV path. If omitted, latest runs_*.csv by timestamp in filename is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output root directory for generated plots.",
    )
    parser.add_argument(
        "--plots-dir-name",
        type=str,
        default="Plots",
        help="Subfolder name for plots without GPU profile.",
    )
    parser.add_argument(
        "--plots-gpu-dir-name",
        type=str,
        default="Pots-GPU",
        help="Subfolder name for plots including GPU profile.",
    )
    parser.add_argument(
        "--with-gpu-data-dir",
        type=Path,
        default=None,
        help="Optional directory containing precomputed variant data (trace_joined.csv, scores_*.csv) for WITH-GPU plots.",
    )
    parser.add_argument(
        "--without-gpu-data-dir",
        type=Path,
        default=None,
        help="Optional directory containing precomputed variant data (trace_joined.csv, scores_*.csv) for WITHOUT-GPU plots.",
    )
    return parser.parse_args()


def _parse_runs_timestamp_key(path: Path) -> tuple[int, ...] | None:
    """Return tuple key parsed from runs_<timestamp>.csv, else None.

    Accepts numeric underscore tokens only, e.g. runs_260227_004.csv.
    Excludes names like runs_low.csv or runs_... copy.csv.
    """
    m = re.fullmatch(r"runs_([0-9_]+)\.csv", path.name)
    if not m:
        return None
    token = m.group(1)
    try:
        return tuple(int(part) for part in token.split("_"))
    except ValueError:
        return None


def pick_latest_runs_file(base_dir: Path) -> Path:
    candidates = []
    for p in base_dir.glob("runs_*.csv"):
        key = _parse_runs_timestamp_key(p)
        if key is not None:
            candidates.append((key, p))

    if not candidates:
        raise FileNotFoundError(f"No valid runs_<timestamp>.csv found in {base_dir}")

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def load_runs(runs_file: Path) -> pd.DataFrame:
    header_cols = list(pd.read_csv(runs_file, nrows=0).columns)
    usecols = [c for c in RUNS_USECOLS if c in header_cols]
    missing = sorted(set(RUNS_USECOLS) - set(usecols))
    if missing:
        LOGGER.warning("Missing runs columns (will skip dependent plots): %s", ", ".join(missing))

    runs = pd.read_csv(runs_file, usecols=usecols)
    if "error_flag" in runs.columns:
        runs["error_flag"] = pd.to_numeric(runs["error_flag"], errors="coerce").fillna(0)
    return runs


def get_experiment_id_from_runs(runs_df: pd.DataFrame) -> str | None:
    if "experiment_id" not in runs_df.columns:
        return None

    values = [v for v in runs_df["experiment_id"].dropna().astype(str).unique() if v and v.lower() != "nan"]
    if len(values) != 1:
        if values:
            LOGGER.warning("Expected exactly one experiment_id in runs data, got: %s", ", ".join(values))
        return None
    return values[0]


def load_scores(base_dir: Path, experiment_id: str | None) -> pd.DataFrame:
    if not experiment_id:
        LOGGER.warning("No experiment_id available from runs file; score plots will be skipped")
        return pd.DataFrame(columns=SCORES_COLS)

    score_file = base_dir / f"scores_{experiment_id}.csv"
    if not score_file.exists():
        LOGGER.warning("Scores file for experiment %s not found: %s", experiment_id, score_file)
        return pd.DataFrame(columns=SCORES_COLS)

    try:
        df = pd.read_csv(score_file)
    except Exception as exc:
        LOGGER.warning("Failed to load %s: %s", score_file, exc)
        return pd.DataFrame(columns=SCORES_COLS)

    cols = [c for c in SCORES_COLS if c in df.columns]
    if not {"profile", "model"}.issubset(cols):
        LOGGER.warning("Skipping %s: missing profile/model", score_file.name)
        return pd.DataFrame(columns=SCORES_COLS)

    scores = df[cols].copy()
    for col in ["token_score_norm", "latency_score", "retrieval_score", "answer_score"]:
        if col in scores.columns:
            scores[col] = pd.to_numeric(scores[col], errors="coerce")
    return scores


def recompute_scores_from_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty or not {"profile", "model"}.issubset(runs_df.columns):
        return pd.DataFrame(columns=SCORES_COLS)

    calc_input: list[dict[str, object]] = []
    for row in runs_df.to_dict(orient="records"):
        latency_ms = row.get("latency_ms")
        calc_input.append(
            {
                "profile": row.get("profile"),
                "model": row.get("model"),
                "tokens_per_second": row.get("tokens_per_s"),
                "total_latency": (float(latency_ms) / 1000.0) if pd.notna(latency_ms) else None,
            }
        )

    token_scores = TokenScoreCalculator().calculate_scores(calc_input, group_by_keys=("profile", "model"))
    latency_scores = LatencyScoreCalculator().calculate_scores(calc_input, group_by_keys=("profile", "model"))

    retrieval_values_by_combo: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    answer_values_by_combo: defaultdict[tuple[str, str], list[float]] = defaultdict(list)

    for row in runs_df.to_dict(orient="records"):
        profile = row.get("profile")
        model = row.get("model")
        if pd.isna(profile) or pd.isna(model):
            continue

        combo = (str(profile), str(model))

        retrieval_value = row.get("retrieval_score")
        if pd.notna(retrieval_value):
            retrieval_values_by_combo[combo].append(float(retrieval_value))

        answer_value = row.get("llm_judge_score")
        if pd.notna(answer_value):
            answer_values_by_combo[combo].append(float(answer_value))

    retrieval_scores = {
        combo: sum(values) / len(values)
        for combo, values in retrieval_values_by_combo.items()
        if values
    }
    answer_scores = {
        combo: sum(values) / len(values)
        for combo, values in answer_values_by_combo.items()
        if values
    }

    all_combinations = set(token_scores) | set(latency_scores) | set(retrieval_scores) | set(answer_scores)
    rows = []
    for profile, model in sorted(all_combinations):
        rows.append(
            {
                "profile": profile,
                "model": model,
                "token_score_norm": token_scores.get((profile, model)),
                "latency_score": latency_scores.get((profile, model)),
                "retrieval_score": retrieval_scores.get((profile, model)),
                "answer_score": answer_scores.get((profile, model)),
            }
        )

    scores = pd.DataFrame(rows, columns=SCORES_COLS)
    for col in ["token_score_norm", "latency_score", "retrieval_score", "answer_score"]:
        if col in scores.columns:
            scores[col] = pd.to_numeric(scores[col], errors="coerce")
    return scores


def load_scores_from_data_dir(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("scores_*.csv"))
    if not files:
        LOGGER.warning("No scores_*.csv found in %s", data_dir)
        return pd.DataFrame(columns=SCORES_COLS)
    try:
        df = pd.read_csv(files[-1])
    except Exception as exc:
        LOGGER.warning("Failed to load %s: %s", files[-1], exc)
        return pd.DataFrame(columns=SCORES_COLS)

    cols = [c for c in SCORES_COLS if c in df.columns]
    if not {"profile", "model"}.issubset(cols):
        LOGGER.warning("Scores file %s missing profile/model", files[-1])
        return pd.DataFrame(columns=SCORES_COLS)
    out = df[cols].copy()
    for col in ["token_score_norm", "latency_score", "retrieval_score", "answer_score"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_variant_from_data_dir(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trace_file = data_dir / "trace_joined.csv"
    if not trace_file.exists():
        raise FileNotFoundError(f"trace_joined.csv not found in {data_dir}")

    header_cols = list(pd.read_csv(trace_file, nrows=0).columns)
    run_usecols = [c for c in RUNS_USECOLS if c in header_cols]
    sys_map = {
        "sys_cpu_usage_mean": "cpu_usage_mean",
        "sys_cpu_usage_max": "cpu_usage_p95",
        "sys_ram_used_mb_mean": "ram_used_mb_mean",
        "sys_ram_used_mb_max": "ram_used_mb_p95",
        "sys_ollama_proc_cpu_percent_mean": "ollama_proc_cpu_percent_mean",
        "sys_ollama_proc_cpu_percent_max": "ollama_proc_cpu_percent_p95",
        "sys_ollama_proc_rss_mb_mean": "ollama_proc_rss_mb_mean",
        "sys_ollama_proc_rss_mb_max": "ollama_proc_rss_mb_p95",
        "sys_gpu_usage_mean": "gpu_usage_mean",
        "sys_gpu_usage_max": "gpu_usage_p95",
        "sys_gpu_memory_mean": "gpu_memory_mean",
        "sys_gpu_memory_max": "gpu_memory_p95",
    }
    sys_cols = [c for c in sys_map if c in header_cols]

    df = pd.read_csv(trace_file, usecols=[*run_usecols, *sys_cols])
    ensure_numeric(
        df,
        [
            "latency_ms",
            "tokens_per_s",
            "llm_judge_score",
            "llm_judge_f",
            "llm_judge_r",
            "llm_judge_c",
            "llm_judge_l",
            "load_duration_s",
            "prompt_eval_duration_s",
            "eval_duration_s",
            "retrieval_score",
            "ndcg@k",
            "recall@k",
            "error_flag",
        ],
    )
    if "error_flag" in df.columns:
        df["error_flag"] = pd.to_numeric(df["error_flag"], errors="coerce").fillna(0)

    system_agg = pd.DataFrame(columns=["trace_id"])
    if "trace_id" in df.columns and sys_cols:
        renamed = df[["trace_id", *sys_cols]].rename(columns=sys_map)
        for col in renamed.columns:
            if col != "trace_id":
                renamed[col] = pd.to_numeric(renamed[col], errors="coerce")
        system_agg = renamed.groupby("trace_id", dropna=False).mean(numeric_only=True).reset_index()

    scores = recompute_scores_from_runs(df)
    return df, scores, system_agg


def _collect_metric_files(base_dir: Path, kind_prefix: str) -> list[Path]:
    files = sorted(base_dir.glob(f"{kind_prefix}_*.csv"))

    metrics_root = base_dir.parent / "metrics"
    if metrics_root.exists():
        files.extend(sorted(metrics_root.rglob(f"{kind_prefix}_*.csv")))

    dedup = {str(p.resolve()): p for p in files}
    return sorted(dedup.values())


def _aggregate_metrics(files: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            header_cols = list(pd.read_csv(f, nrows=0).columns)
            usecols = [c for c in METRIC_COLS if c in header_cols]
            if "trace_id" not in usecols:
                continue
            df = pd.read_csv(f, usecols=usecols)
            frames.append(df)
        except Exception as exc:
            LOGGER.warning("Failed to load metrics file %s: %s", f, exc)

    if not frames:
        return pd.DataFrame(columns=["trace_id"])

    all_metrics = pd.concat(frames, ignore_index=True)
    if "snapshot_type" in all_metrics.columns:
        all_metrics = all_metrics[all_metrics["snapshot_type"].astype(str).str.lower() == "continuous"].copy()

    numeric_cols = [c for c in METRIC_COLS if c not in {"trace_id", "snapshot_type"} and c in all_metrics.columns]
    for c in numeric_cols:
        all_metrics[c] = pd.to_numeric(all_metrics[c], errors="coerce")

    grouped = all_metrics.groupby("trace_id", dropna=False)
    agg_data: dict[str, pd.Series] = {}
    for c in numeric_cols:
        agg_data[f"{c}_mean"] = grouped[c].mean()
        agg_data[f"{c}_p95"] = grouped[c].quantile(0.95)

    if not agg_data:
        return pd.DataFrame(columns=["trace_id"])

    agg_df = pd.DataFrame(agg_data).reset_index()
    return agg_df


def load_and_aggregate_metrics(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    system_files = _collect_metric_files(base_dir, "system_metrics")
    embedding_files = _collect_metric_files(base_dir, "embedding_metrics")

    if not system_files:
        LOGGER.warning("No system_metrics_*.csv found under %s or %s", base_dir, base_dir.parent / "metrics")
    if not embedding_files:
        LOGGER.warning(
            "No embedding_metrics_*.csv found under %s or %s", base_dir, base_dir.parent / "metrics"
        )

    system_agg = _aggregate_metrics(system_files)
    embedding_agg = _aggregate_metrics(embedding_files)
    return system_agg, embedding_agg


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def add_profile_model_key(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "profile" in result.columns and "model" in result.columns:
        result["profile_model"] = result["profile"].astype(str) + "|" + result["model"].astype(str)
    return result


def make_output_dir(base_dir: Path, runs_file: Path, runs_df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    exp_id = get_experiment_id_from_runs(runs_df)

    ts_match = re.search(r"runs_([0-9_]+)\.csv", runs_file.name)
    ts_token = ts_match.group(1) if ts_match else "unknown"
    folder = exp_id if exp_id else ts_token

    out_dir = base_dir / "plots" / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_fig(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def set_seconds_axis(ax: plt.Axes, step: float = 100.0) -> None:
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(MultipleLocator(step / 2.0))


def sort_test_case_ids(values: list[str]) -> list[str]:
    def _key(v: str) -> tuple[int, int, str]:
        m = re.fullmatch(r"TC-([A-Z])-([0-9]+)", v or "")
        if not m:
            return (2, 0, v)
        bucket = 0 if m.group(1) == "N" else 1
        return (bucket, int(m.group(2)), v)

    return sorted(values, key=_key)


def plot_box_latency(df: pd.DataFrame, out_dir: Path) -> Path | None:
    req = {"latency_ms", "profile_model"}
    if not req.issubset(df.columns):
        return None
    work = df[["profile_model", "latency_ms"]].dropna().copy()
    if work.empty:
        return None
    work["latency_s"] = work["latency_ms"] / 1000.0

    order = work.groupby("profile_model")["latency_s"].median().sort_values().index.tolist()
    data = [work.loc[work["profile_model"] == k, "latency_s"].values for k in order]

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True)
    ax.set_title("Latency (s) by profile|model")
    ax.set_ylabel("latency_s")
    set_seconds_axis(ax, step=100.0)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", which="both", alpha=0.3)

    out = out_dir / "01_latency_boxplot_log.png"
    save_fig(fig, out)
    return out


def plot_box_tokens(df: pd.DataFrame, out_dir: Path) -> Path | None:
    req = {"tokens_per_s", "profile_model"}
    if not req.issubset(df.columns):
        return None
    work = df[["profile_model", "tokens_per_s"]].dropna()
    if work.empty:
        return None

    order = work.groupby("profile_model")["tokens_per_s"].median().sort_values().index.tolist()
    data = [work.loc[work["profile_model"] == k, "tokens_per_s"].values for k in order]

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True)
    ax.set_title("Tokens/s by profile|model")
    ax.set_ylabel("tokens_per_s")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    out = out_dir / "02_tokens_per_s_boxplot.png"
    save_fig(fig, out)
    return out


def plot_pareto(df: pd.DataFrame, out_dir: Path) -> Path | None:
    req = {"profile", "model", "latency_ms", "llm_judge_score", "tokens_per_s"}
    if not req.issubset(df.columns):
        return None
    grp = (
        df.groupby(["profile", "model"], dropna=False)
        .agg(
            latency_median_s=("latency_ms", lambda s: s.median() / 1000.0),
            judge_mean=("llm_judge_score", "mean"),
            tps_median=("tokens_per_s", "median"),
        )
        .reset_index()
        .dropna()
    )
    if grp.empty:
        return None

    profiles = sorted(grp["profile"].astype(str).unique())
    cmap = plt.get_cmap("tab10")
    color_map = {p: cmap(i % 10) for i, p in enumerate(profiles)}

    fig, ax = plt.subplots(figsize=(13, 8))
    size_base = grp["tps_median"].fillna(0)
    size_scaled = 50 + 350 * (size_base - size_base.min()) / ((size_base.max() - size_base.min()) or 1)

    for _, row in grp.iterrows():
        profile = str(row["profile"])
        model = str(row["model"])
        x = row["latency_median_s"]
        y = row["judge_mean"]
        s = float(size_scaled.loc[_])
        ax.scatter(x, y, s=s, color=color_map[profile], alpha=0.7, edgecolors="black", linewidths=0.5)
        ax.annotate(model, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[p], label=p, markersize=8)
        for p in profiles
    ]
    ax.legend(handles=legend_handles, title="profile", loc="best")
    ax.set_xscale("log")
    ax.set_title("Pareto: median latency(s) vs mean llm_judge_score (bubble=median tokens/s)")
    ax.set_xlabel("median latency_s (log)")
    ax.set_ylabel("mean llm_judge_score")
    ax.grid(alpha=0.3)

    out = out_dir / "03_pareto_latency_answer_tokens.png"
    save_fig(fig, out)
    return out


def plot_latency_breakdown(df: pd.DataFrame, out_dir: Path) -> Path | None:
    req = {"profile_model", "load_duration_s", "prompt_eval_duration_s", "eval_duration_s"}
    if not req.issubset(df.columns):
        return None

    grp = (
        df.groupby("profile_model", dropna=False)[["load_duration_s", "prompt_eval_duration_s", "eval_duration_s"]]
        .median()
        .dropna(how="all")
    )
    if grp.empty:
        return None

    x = range(len(grp))
    fig, ax = plt.subplots(figsize=(15, 8))
    b1 = grp["load_duration_s"].fillna(0)
    b2 = grp["prompt_eval_duration_s"].fillna(0)
    b3 = grp["eval_duration_s"].fillna(0)

    ax.bar(x, b1, label="load_duration_s")
    ax.bar(x, b2, bottom=b1, label="prompt_eval_duration_s")
    ax.bar(x, b3, bottom=b1 + b2, label="eval_duration_s")
    ax.set_xticks(list(x))
    ax.set_xticklabels(grp.index.tolist(), rotation=45, ha="right")
    ax.set_ylabel("seconds (median)")
    ax.set_title("Latency Breakdown by profile|model")
    set_seconds_axis(ax, step=100.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = out_dir / "04_latency_breakdown_stacked.png"
    save_fig(fig, out)
    return out


def draw_heatmap(ax: plt.Axes, table: pd.DataFrame, title: str, cmap: str = "YlGnBu") -> None:
    if table.empty:
        ax.set_title(f"{title} (no data)")
        ax.axis("off")
        return

    data = table.values
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index.tolist())

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text = "" if pd.isna(val) else f"{val:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_retrieval_heatmaps(df: pd.DataFrame, out_dir: Path) -> Path | None:
    req = {"model", "profile", "ndcg@k", "recall@k"}
    if not req.issubset(df.columns):
        return None

    ndcg_tbl = df.pivot_table(index="model", columns="profile", values="ndcg@k", aggfunc="mean")
    recall_tbl = df.pivot_table(index="model", columns="profile", values="recall@k", aggfunc="mean")
    if ndcg_tbl.empty and recall_tbl.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    draw_heatmap(axes[0], ndcg_tbl, "mean(ndcg@k) by model x profile")
    draw_heatmap(axes[1], recall_tbl, "mean(recall@k) by model x profile")

    out = out_dir / "05_retrieval_heatmaps_ndcg_recall.png"
    save_fig(fig, out)
    return out


def plot_retrieval_scatter(df: pd.DataFrame, out_dir: Path) -> Path | None:
    req = {"retrieval_score", "llm_judge_score", "profile", "model"}
    if not req.issubset(df.columns):
        return None

    work = df[["retrieval_score", "llm_judge_score", "profile", "model"]].dropna()
    if work.empty:
        return None

    profiles = sorted(work["profile"].astype(str).unique())
    models = sorted(work["model"].astype(str).unique())
    colors = {p: plt.get_cmap("tab10")(i % 10) for i, p in enumerate(profiles)}
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(13, 8))
    for model in models:
        subset_m = work[work["model"].astype(str) == model]
        for profile in profiles:
            subset = subset_m[subset_m["profile"].astype(str) == profile]
            if subset.empty:
                continue
            ax.scatter(
                subset["retrieval_score"],
                subset["llm_judge_score"],
                color=colors[profile],
                marker=marker_map[model],
                alpha=0.7,
                edgecolors="black",
                linewidths=0.4,
            )

    for profile, color in colors.items():
        ax.scatter([], [], c=[color], marker="o", label=f"profile={profile}")
    for model, marker in marker_map.items():
        ax.scatter([], [], c="black", marker=marker, label=f"model={model}")

    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.set_xlabel("retrieval_score")
    ax.set_ylabel("llm_judge_score")
    ax.set_title("retrieval_score vs llm_judge_score")
    ax.grid(alpha=0.3)

    out = out_dir / "06_retrieval_vs_answer_scatter.png"
    save_fig(fig, out)
    return out


def plot_judge_box(df: pd.DataFrame, out_dir: Path) -> Path | None:
    req = {"profile_model", "llm_judge_score"}
    if not req.issubset(df.columns):
        return None

    work = df[["profile_model", "llm_judge_score"]].dropna()
    if work.empty:
        return None

    order = work.groupby("profile_model")["llm_judge_score"].median().sort_values().index.tolist()
    data = [work.loc[work["profile_model"] == k, "llm_judge_score"].values for k in order]

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True)
    ax.set_title("llm_judge_score by profile|model")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    out = out_dir / "07_answer_quality_boxplot.png"
    save_fig(fig, out)
    return out


def plot_judge_components(df: pd.DataFrame, out_dir: Path) -> Path | None:
    cols = ["llm_judge_f", "llm_judge_r", "llm_judge_c", "llm_judge_l"]
    req = {"profile_model", *cols}
    if not req.issubset(df.columns):
        return None

    grp = df.groupby("profile_model", dropna=False)[cols].mean().dropna(how="all")
    if grp.empty:
        return None

    fig, ax = plt.subplots(figsize=(16, 8))
    width = 0.2
    x = pd.Series(range(len(grp)))

    for idx, c in enumerate(cols):
        ax.bar(x + (idx - 1.5) * width, grp[c].fillna(0), width=width, label=c)

    ax.set_xticks(x)
    ax.set_xticklabels(grp.index.tolist(), rotation=45, ha="right")
    ax.set_title("Mean llm_judge dimensions by profile|model")
    ax.set_ylabel("mean score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = out_dir / "08_answer_dimensions_grouped_bar.png"
    save_fig(fig, out)
    return out


def plot_scores_heatmaps(scores_df: pd.DataFrame, out_dir: Path) -> Path | None:
    if scores_df.empty:
        return None

    needed = ["token_score_norm", "latency_score", "retrieval_score", "answer_score"]
    if not {"model", "profile"}.issubset(scores_df.columns):
        return None

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes_list = axes.flatten()

    for ax, col in zip(axes_list, needed):
        if col not in scores_df.columns:
            ax.axis("off")
            ax.set_title(f"{col} (missing)")
            continue
        tbl = scores_df.pivot_table(index="model", columns="profile", values=col, aggfunc="mean")
        draw_heatmap(ax, tbl, f"{col} by model x profile")

    out = out_dir / "09_scores_heatmaps_2x2.png"
    save_fig(fig, out)
    return out


def plot_scores_grouped(scores_df: pd.DataFrame, out_dir: Path) -> Path | None:
    if scores_df.empty:
        return None

    cols = ["token_score_norm", "latency_score", "retrieval_score", "answer_score"]
    req = {"profile", "model", *cols}
    if not req.issubset(scores_df.columns):
        return None

    grp = scores_df.groupby(["profile", "model"], dropna=False)[cols].mean().reset_index()
    if grp.empty:
        return None

    groups = grp["profile"].astype(str).unique().tolist()
    models = grp["model"].astype(str).unique().tolist()

    fig, axes = plt.subplots(len(groups), 1, figsize=(16, 5 * max(1, len(groups))), squeeze=False)
    for i, profile in enumerate(groups):
        ax = axes[i, 0]
        sub = grp[grp["profile"].astype(str) == profile].set_index("model")[cols]
        x = pd.Series(range(len(sub.index)))
        width = 0.2
        for idx, c in enumerate(cols):
            ax.bar(x + (idx - 1.5) * width, sub[c].fillna(0), width=width, label=c)
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index.tolist(), rotation=45, ha="right")
        ax.set_title(f"Normalized scores by model (profile={profile})")
        ax.set_ylabel("score")
        ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.legend()

    out = out_dir / "10_scores_grouped_bars_by_profile.png"
    save_fig(fig, out)
    return out


def _resolve_util_memory_series(grp: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    profile_lower = grp["profile"].astype(str).str.lower()
    util_mean = grp["cpu_usage_mean"] if "cpu_usage_mean" in grp else pd.Series([0.0] * len(grp))
    util_p95 = grp["cpu_usage_p95"] if "cpu_usage_p95" in grp else pd.Series([0.0] * len(grp))
    mem_mean = grp["ram_used_mb_mean"] if "ram_used_mb_mean" in grp else pd.Series([0.0] * len(grp))
    mem_p95 = grp["ram_used_mb_p95"] if "ram_used_mb_p95" in grp else pd.Series([0.0] * len(grp))

    if "gpu_usage_mean" in grp:
        util_mean = util_mean.where(profile_lower != "gpu", grp["gpu_usage_mean"])
    if "gpu_usage_p95" in grp:
        util_p95 = util_p95.where(profile_lower != "gpu", grp["gpu_usage_p95"])
    if "vram_used_mb_mean" in grp:
        mem_mean = mem_mean.where(profile_lower != "gpu", grp["vram_used_mb_mean"])
    elif "gpu_memory_mean" in grp:
        mem_mean = mem_mean.where(profile_lower != "gpu", grp["gpu_memory_mean"])
    if "vram_used_mb_p95" in grp:
        mem_p95 = mem_p95.where(profile_lower != "gpu", grp["vram_used_mb_p95"])
    elif "gpu_memory_p95" in grp:
        mem_p95 = mem_p95.where(profile_lower != "gpu", grp["gpu_memory_p95"])

    return util_mean, util_p95, mem_mean, mem_p95


def _plot_metric_bars(grp: pd.DataFrame, values_a: pd.Series, values_b: pd.Series, title: str, labels: tuple[str, str], out: Path) -> Path | None:
    if grp.empty:
        return None
    x_labels = (grp["profile"].astype(str) + "|" + grp["model"].astype(str)).tolist()
    x = pd.Series(range(len(x_labels)))
    fig, ax = plt.subplots(figsize=(18, 8))
    width = 0.35
    ax.bar(x - width / 2, values_a.fillna(0), width=width, label=labels[0])
    ax.bar(x + width / 2, values_b.fillna(0), width=width, label=labels[1])
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, out)
    return out


def plot_system_utilization(joined_system: pd.DataFrame, out_dir: Path) -> Path | None:
    if not {"profile", "model"}.issubset(joined_system.columns) or joined_system.empty:
        return None
    sys_cols = [c for c in joined_system.columns if c.endswith("_mean") or c.endswith("_p95")]
    grp = joined_system.groupby(["profile", "model"], dropna=False)[sys_cols].mean().reset_index()
    util_mean, util_p95, _, _ = _resolve_util_memory_series(grp)
    return _plot_metric_bars(
        grp=grp,
        values_a=util_mean,
        values_b=util_p95,
        title="System Metrics: CPU/GPU Utilization by profile|model",
        labels=("util mean", "util p95"),
        out=out_dir / "11_system_utilization_bars.png",
    )


def plot_system_memory(joined_system: pd.DataFrame, out_dir: Path) -> Path | None:
    if not {"profile", "model"}.issubset(joined_system.columns) or joined_system.empty:
        return None
    sys_cols = [c for c in joined_system.columns if c.endswith("_mean") or c.endswith("_p95")]
    grp = joined_system.groupby(["profile", "model"], dropna=False)[sys_cols].mean().reset_index()
    _, _, mem_mean, mem_p95 = _resolve_util_memory_series(grp)
    return _plot_metric_bars(
        grp=grp,
        values_a=mem_mean,
        values_b=mem_p95,
        title="System Metrics: RAM/VRAM by profile|model",
        labels=("memory mean mb", "memory p95 mb"),
        out=out_dir / "12_system_memory_bars.png",
    )


def plot_embedding_utilization(joined_embedding: pd.DataFrame, out_dir: Path) -> Path | None:
    if not {"profile", "model"}.issubset(joined_embedding.columns) or joined_embedding.empty:
        return None
    emb_cols = [c for c in joined_embedding.columns if c.endswith("_mean") or c.endswith("_p95")]
    grp = joined_embedding.groupby(["profile", "model"], dropna=False)[emb_cols].mean().reset_index()
    util_mean, util_p95, _, _ = _resolve_util_memory_series(grp)
    return _plot_metric_bars(
        grp=grp,
        values_a=util_mean,
        values_b=util_p95,
        title="Embedding Metrics: CPU/GPU Utilization by profile|model",
        labels=("util mean", "util p95"),
        out=out_dir / "13_embedding_utilization_bars.png",
    )


def plot_embedding_memory(joined_embedding: pd.DataFrame, out_dir: Path) -> Path | None:
    if not {"profile", "model"}.issubset(joined_embedding.columns) or joined_embedding.empty:
        return None
    emb_cols = [c for c in joined_embedding.columns if c.endswith("_mean") or c.endswith("_p95")]
    grp = joined_embedding.groupby(["profile", "model"], dropna=False)[emb_cols].mean().reset_index()
    _, _, mem_mean, mem_p95 = _resolve_util_memory_series(grp)
    return _plot_metric_bars(
        grp=grp,
        values_a=mem_mean,
        values_b=mem_p95,
        title="Embedding Metrics: RAM/VRAM by profile|model",
        labels=("memory mean mb", "memory p95 mb"),
        out=out_dir / "14_embedding_memory_bars.png",
    )


def plot_error_analysis(df: pd.DataFrame, out_dir: Path) -> Path | None:
    if not {"profile", "model", "error_flag"}.issubset(df.columns):
        return None
    work = df.copy()
    work["error_flag"] = pd.to_numeric(work["error_flag"], errors="coerce").fillna(0)

    grp = work.groupby(["profile", "model"], dropna=False)["error_flag"].mean().reset_index()
    labels = (grp["profile"].astype(str) + "|" + grp["model"].astype(str)).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    x = pd.Series(range(len(labels)))
    axes[0].bar(x, grp["error_flag"] * 100)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_title("Fail rate (error_flag==1) by profile|model")
    axes[0].set_ylabel("fail rate %")
    axes[0].grid(axis="y", alpha=0.3)

    if "test_case_id" in work.columns and work["test_case_id"].notna().any():
        hm = work.pivot_table(index="test_case_id", columns="profile", values="error_flag", aggfunc="mean")
        draw_heatmap(axes[1], hm * 100, "Fail rate % by test_case_id x profile", cmap="OrRd")
    else:
        axes[1].set_title("Fail rate by test_case_id x profile (no data)")
        axes[1].axis("off")

    out = out_dir / "15_error_analysis_failrate.png"
    save_fig(fig, out)
    return out


def plot_metric_overview_all_models(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    out_dir: Path,
    filename: str,
    color: str,
) -> Path | None:
    required = {"test_case_id", metric_col}
    if not required.issubset(df.columns):
        return None

    work = df[["test_case_id", metric_col]].dropna().copy()
    if work.empty:
        return None

    grp = (
        work.groupby("test_case_id", dropna=False)[metric_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "metric_mean", "std": "metric_std", "count": "n"})
    )
    if grp.empty:
        return None

    ordered_ids = sort_test_case_ids(grp["test_case_id"].astype(str).tolist())
    grp["test_case_id"] = grp["test_case_id"].astype(str)
    grp = grp.set_index("test_case_id").reindex(ordered_ids).reset_index()
    grp["metric_std"] = pd.to_numeric(grp["metric_std"], errors="coerce").fillna(0.0)

    fig, ax = plt.subplots(figsize=(18, 8))
    x = pd.Series(range(len(grp)))
    ax.bar(x, grp["metric_mean"], color=color, alpha=0.9)
    ax.errorbar(
        x,
        grp["metric_mean"],
        yerr=grp["metric_std"],
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=2,
        alpha=0.7,
    )

    global_mean = float(work[metric_col].mean())
    ax.axhline(global_mean, color="#d62728", linestyle="--", linewidth=1.2, label=f"global mean={global_mean:.3f}")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(grp["test_case_id"].tolist(), rotation=90, fontsize=8)
    ax.set_ylabel(metric_label)
    ax.set_xlabel("test_case_id")
    ax.set_title(f"{metric_label}: overview across all models/profiles (mean ± std by testcase)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    out = out_dir / filename
    save_fig(fig, out)
    return out


def plot_recall_overview_all_models(df: pd.DataFrame, out_dir: Path) -> Path | None:
    return plot_metric_overview_all_models(
        df=df,
        metric_col="recall@k",
        metric_label="Recall@k",
        out_dir=out_dir,
        filename="16_recall_overview_all_models.png",
        color="#4e79a7",
    )


def plot_ndcg_overview_all_models(df: pd.DataFrame, out_dir: Path) -> Path | None:
    return plot_metric_overview_all_models(
        df=df,
        metric_col="ndcg@k",
        metric_label="nDCG@k",
        out_dir=out_dir,
        filename="17_ndcg_overview_all_models.png",
        color="#59a14f",
    )


def plot_retrieval_score_overview_all_models(df: pd.DataFrame, out_dir: Path) -> Path | None:
    return plot_metric_overview_all_models(
        df=df,
        metric_col="retrieval_score",
        metric_label="Retrieval Score",
        out_dir=out_dir,
        filename="18_retrieval_score_overview_all_models.png",
        color="#f28e2b",
    )


def _aggregate_recall_per_model_case(df: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "test_case_id", "recall@k"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["model", "test_case_id", "recall_case"])

    work = df[["model", "test_case_id", "recall@k"]].dropna().copy()
    if work.empty:
        return pd.DataFrame(columns=["model", "test_case_id", "recall_case"])

    work["recall@k"] = pd.to_numeric(work["recall@k"], errors="coerce")
    work = work.dropna(subset=["recall@k"])
    if work.empty:
        return pd.DataFrame(columns=["model", "test_case_id", "recall_case"])

    # Robust gegen mehrere Wiederholungen je Modell+Testfall: mittlerer Recall pro Testfall.
    case_level = (
        work.groupby(["model", "test_case_id"], dropna=False)["recall@k"]
        .mean()
        .reset_index(name="recall_case")
    )
    return case_level


def plot_recall_model_bar(df: pd.DataFrame, out_dir: Path) -> Path | None:
    case_level = _aggregate_recall_per_model_case(df)
    if case_level.empty:
        return None

    model_agg = (
        case_level.groupby("model", dropna=False)["recall_case"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="recall_at_k_model")
    )
    if model_agg.empty:
        return None

    fig, ax = plt.subplots(figsize=(14, 8))
    x = pd.Series(range(len(model_agg)))
    ax.bar(x, model_agg["recall_at_k_model"], color="#3a86ff")
    ax.set_xticks(x)
    ax.set_xticklabels(model_agg["model"].astype(str).tolist(), rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Recall@k")
    ax.set_xlabel("model")
    ax.set_title("Aggregated Recall@k per model")
    ax.grid(axis="y", alpha=0.3)

    out = out_dir / "19_recall_aggregated_per_model_bar.png"
    save_fig(fig, out)
    return out


def plot_recall_hit_distribution(df: pd.DataFrame, out_dir: Path) -> Path | None:
    case_level = _aggregate_recall_per_model_case(df)
    if case_level.empty:
        return None

    model_counts = (
        case_level.groupby("model", dropna=False)
        .agg(
            total_testcases=("test_case_id", "nunique"),
            hits=("recall_case", "sum"),
        )
        .reset_index()
    )
    if model_counts.empty:
        return None

    model_counts["misses"] = model_counts["total_testcases"] - model_counts["hits"]
    model_counts = model_counts.sort_values("hits", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 8))
    x = pd.Series(range(len(model_counts)))
    ax.bar(x, model_counts["hits"], label="Treffer", color="#2a9d8f")
    ax.bar(x, model_counts["misses"], bottom=model_counts["hits"], label="Nicht Treffer", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(model_counts["model"].astype(str).tolist(), rotation=45, ha="right")
    ax.set_ylabel("Anzahl Testfälle")
    ax.set_xlabel("model")
    ax.set_title("Trefferverteilung pro Modell (Recall@k)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    out = out_dir / "20_recall_hits_misses_stacked_by_model.png"
    save_fig(fig, out)
    return out


def run_plot_suite(
    runs_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    system_agg: pd.DataFrame,
    embedding_agg: pd.DataFrame,
    out_dir: Path,
    variant_label: str,
) -> list[Path]:
    runs_local = add_profile_model_key(runs_df)
    joined_system = runs_local.merge(system_agg, on="trace_id", how="left", suffixes=("", "_sys"))
    joined_embedding = runs_local.merge(embedding_agg, on="trace_id", how="left", suffixes=("", "_emb"))

    plotters = [
        lambda: plot_box_latency(runs_local, out_dir),
        lambda: plot_box_tokens(runs_local, out_dir),
        lambda: plot_pareto(runs_local, out_dir),
        lambda: plot_latency_breakdown(runs_local, out_dir),
        lambda: plot_retrieval_heatmaps(runs_local, out_dir),
        lambda: plot_retrieval_scatter(runs_local, out_dir),
        lambda: plot_judge_box(runs_local, out_dir),
        lambda: plot_judge_components(runs_local, out_dir),
        lambda: plot_scores_heatmaps(scores_df, out_dir),
        lambda: plot_scores_grouped(scores_df, out_dir),
        lambda: plot_system_utilization(joined_system, out_dir),
        lambda: plot_system_memory(joined_system, out_dir),
        lambda: plot_embedding_utilization(joined_embedding, out_dir),
        lambda: plot_embedding_memory(joined_embedding, out_dir),
        lambda: plot_error_analysis(runs_local, out_dir),
        lambda: plot_recall_overview_all_models(runs_local, out_dir),
        lambda: plot_ndcg_overview_all_models(runs_local, out_dir),
        lambda: plot_retrieval_score_overview_all_models(runs_local, out_dir),
        lambda: plot_recall_model_bar(runs_local, out_dir),
        lambda: plot_recall_hit_distribution(runs_local, out_dir),
    ]

    created: list[Path] = []
    for fn in plotters:
        try:
            out = fn()
            if out:
                created.append(out)
        except Exception as exc:
            LOGGER.warning("[%s] Plot failed: %s", variant_label, exc)

    total_runs = len(runs_local)
    total_errors = int(pd.to_numeric(runs_local.get("error_flag", 0), errors="coerce").fillna(0).sum())
    models = sorted(runs_local.get("model", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    profiles = sorted(runs_local.get("profile", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())

    LOGGER.info("[%s] Summary", variant_label)
    LOGGER.info("[%s] - Anzahl Runs: %d", variant_label, total_runs)
    LOGGER.info("[%s] - Anzahl Fehler: %d", variant_label, total_errors)
    LOGGER.info("[%s] - Modelle (%d): %s", variant_label, len(models), ", ".join(models) if models else "-")
    LOGGER.info("[%s] - Profile (%d): %s", variant_label, len(profiles), ", ".join(profiles) if profiles else "-")
    LOGGER.info("[%s] - Plot-Dateien erzeugt: %d", variant_label, len(created))
    for p in created:
        LOGGER.info("[%s]   %s", variant_label, p)

    return created


def filter_metrics_by_trace(metric_df: pd.DataFrame, runs_df: pd.DataFrame) -> pd.DataFrame:
    if metric_df.empty or "trace_id" not in metric_df.columns or "trace_id" not in runs_df.columns:
        return pd.DataFrame(columns=["trace_id"])
    trace_ids = runs_df["trace_id"].dropna().astype(str).unique().tolist()
    return metric_df[metric_df["trace_id"].astype(str).isin(trace_ids)].copy()


def main() -> None:
    setup_logging()
    args = parse_args()
    base_dir: Path = args.base_dir

    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    use_variant_dirs = args.with_gpu_data_dir is not None or args.without_gpu_data_dir is not None
    if use_variant_dirs and not (args.with_gpu_data_dir and args.without_gpu_data_dir):
        raise ValueError("Please provide both --with-gpu-data-dir and --without-gpu-data-dir together.")

    runs_file = args.runs_file if args.runs_file else pick_latest_runs_file(base_dir)
    if not runs_file.exists():
        raise FileNotFoundError(f"runs file not found: {runs_file}")

    LOGGER.info("Using runs file: %s", runs_file)
    runs_ref_df = load_runs(runs_file)

    out_root = make_output_dir(base_dir, runs_file, runs_ref_df, output_dir=args.output_dir)
    LOGGER.info("Output root directory: %s", out_root)

    out_all = out_root / args.plots_gpu_dir_name
    out_all.mkdir(parents=True, exist_ok=True)
    out_no_gpu = out_root / args.plots_dir_name
    out_no_gpu.mkdir(parents=True, exist_ok=True)

    if use_variant_dirs:
        LOGGER.info("Using precomputed variant data dirs.")
        runs_with_gpu, scores_with_gpu, system_with_gpu = load_variant_from_data_dir(args.with_gpu_data_dir)
        runs_without_gpu, scores_without_gpu, system_without_gpu = load_variant_from_data_dir(args.without_gpu_data_dir)
        _, embedding_agg_all = load_and_aggregate_metrics(base_dir)
        embedding_with_gpu = filter_metrics_by_trace(embedding_agg_all, runs_with_gpu)
        embedding_without_gpu = filter_metrics_by_trace(embedding_agg_all, runs_without_gpu)

        run_plot_suite(
            runs_df=runs_with_gpu,
            scores_df=scores_with_gpu,
            system_agg=system_with_gpu,
            embedding_agg=embedding_with_gpu,
            out_dir=out_all,
            variant_label=f"with_gpu ({args.plots_gpu_dir_name})",
        )
        run_plot_suite(
            runs_df=runs_without_gpu,
            scores_df=scores_without_gpu,
            system_agg=system_without_gpu,
            embedding_agg=embedding_without_gpu,
            out_dir=out_no_gpu,
            variant_label=f"without_gpu ({args.plots_dir_name})",
        )
        return

    runs_df = load_runs(runs_file)
    ensure_numeric(
        runs_df,
        [
            "latency_ms",
            "tokens_per_s",
            "llm_judge_score",
            "llm_judge_f",
            "llm_judge_r",
            "llm_judge_c",
            "llm_judge_l",
            "load_duration_s",
            "prompt_eval_duration_s",
            "eval_duration_s",
            "retrieval_score",
            "ndcg@k",
            "recall@k",
        ],
    )

    experiment_id = get_experiment_id_from_runs(runs_df)
    scores_df = recompute_scores_from_runs(runs_df)
    if scores_df.empty:
        scores_df = load_scores(base_dir, experiment_id)
    ensure_numeric(scores_df, ["token_score_norm", "latency_score", "retrieval_score", "answer_score"])
    system_agg, embedding_agg = load_and_aggregate_metrics(base_dir)

    run_plot_suite(
        runs_df=runs_df,
        scores_df=scores_df,
        system_agg=system_agg,
        embedding_agg=embedding_agg,
        out_dir=out_all,
        variant_label=f"with_gpu ({args.plots_gpu_dir_name})",
    )

    if "profile" in runs_df.columns:
        runs_no_gpu = runs_df[runs_df["profile"].astype(str).str.lower() != "gpu"].copy()
    else:
        runs_no_gpu = runs_df.copy()
    if "profile" in scores_df.columns:
        scores_no_gpu = scores_df[scores_df["profile"].astype(str).str.lower() != "gpu"].copy()
    else:
        scores_no_gpu = scores_df.copy()
    run_plot_suite(
        runs_df=runs_no_gpu,
        scores_df=scores_no_gpu,
        system_agg=system_agg,
        embedding_agg=embedding_agg,
        out_dir=out_no_gpu,
        variant_label=f"without_gpu ({args.plots_dir_name})",
    )


if __name__ == "__main__":
    main()
