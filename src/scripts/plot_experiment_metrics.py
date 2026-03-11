#!/usr/bin/env python3
"""Erstellt Diagramme fuer ein ausgewertetes Experiment."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Fehlende Abhaengigkeit: matplotlib. "
        "Installiere es im aktiven Environment, z. B. mit 'pip install matplotlib'."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeuge Diagramme fuer Retrieval, Answer, Latenz und Token/s."
    )
    parser.add_argument("--experiment-id", required=True, help="Experiment-ID, z. B. 20260227_223132")
    parser.add_argument("--output-root", default="output", help="Output-Root (Default: output)")
    parser.add_argument(
        "--exclude-gpu",
        action="store_true",
        help="Schliesst GPU-Profil-Daten aus allen Darstellungen aus.",
    )
    return parser.parse_args()


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _try_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _sort_testcase_ids(ids: list[str]) -> list[str]:
    def _key(value: str) -> tuple[int, int, str]:
        m = re.match(r"^TC-([A-Z])-([0-9]+)$", value or "")
        if not m:
            return (2, 0, value)
        bucket = 0 if m.group(1) == "N" else 1
        return (bucket, int(m.group(2)), value)

    return sorted(ids, key=_key)


def _label_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "profile" in out.columns and "model" in out.columns:
        out["profile_model"] = out["profile"].astype(str) + " | " + out["model"].astype(str)
    elif "model" in out.columns:
        out["profile_model"] = out["model"].astype(str)
    else:
        out["profile_model"] = "n/a"
    return out


def _exclude_gpu_profiles(df: pd.DataFrame, exclude_gpu: bool) -> pd.DataFrame:
    if not exclude_gpu or df.empty or "profile" not in df.columns:
        return df.copy()
    out = df.copy()
    return out[out["profile"].astype(str).str.lower() != "gpu"].copy()


def _ensure_latency_seconds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Primary source from current summary schema.
    if "latency_ms_mean" in out.columns:
        out["latency_s_mean"] = out["latency_ms_mean"] / 1000.0
    elif "latency_mean_ms" in out.columns:
        out["latency_s_mean"] = out["latency_mean_ms"] / 1000.0

    # P95 if available, otherwise fallback to mean to keep plots robust.
    if "latency_ms_p95" in out.columns:
        out["latency_s_p95"] = out["latency_ms_p95"] / 1000.0
    elif "latency_p95_ms" in out.columns:
        out["latency_s_p95"] = out["latency_p95_ms"] / 1000.0
    elif "latency_s_mean" in out.columns:
        out["latency_s_p95"] = out["latency_s_mean"]

    return out


def _extract_gpu_clock_mhz(system_df: pd.DataFrame) -> pd.Series:
    if "response_raw" not in system_df.columns:
        return pd.Series([pd.NA] * len(system_df), index=system_df.index, dtype="float64")

    # response_raw is stored as a dict-like string from logger; extract optional GPU clock keys.
    clock_values: list[float | None] = []
    candidates = (
        "gpu_clock",
        "gpu_clock_mhz",
        "gpu_core_clock",
        "gpu_core_clock_mhz",
        "gpu_sm_clock_mhz",
        "gpu_mem_clock_mhz",
    )

    for raw in system_df["response_raw"].tolist():
        value = None
        if isinstance(raw, str) and raw.startswith("{") and raw.endswith("}"):
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, dict):
                for key in candidates:
                    if key in parsed and parsed.get(key) is not None:
                        try:
                            value = float(parsed.get(key))
                        except (TypeError, ValueError):
                            value = None
                        break
        clock_values.append(value)

    return pd.to_numeric(pd.Series(clock_values, index=system_df.index), errors="coerce")


def validate_system_metrics(system_df: pd.DataFrame, exp_id: str) -> None:
    if system_df.empty:
        print(f"[WARN] [{exp_id}] Keine timeseries_system_metrics fuer Validierung gefunden.")
        return

    cpu_sys = pd.to_numeric(system_df.get("cpu_usage"), errors="coerce")
    cpu_proc = pd.to_numeric(system_df.get("ollama_proc_cpu_percent"), errors="coerce")

    sys_valid = int(cpu_sys.notna().sum())
    proc_valid = int(cpu_proc.notna().sum())

    print(f"[INFO] [{exp_id}] system_metrics CPU Stichproben: cpu_usage={sys_valid}, ollama_proc_cpu_percent={proc_valid}")

    if sys_valid:
        print(
            "[INFO] "
            f"[{exp_id}] cpu_usage: mean={cpu_sys.mean():.2f}, p95={cpu_sys.quantile(0.95):.2f}, max={cpu_sys.max():.2f}"
        )
    if proc_valid:
        print(
            "[INFO] "
            f"[{exp_id}] ollama_proc_cpu_percent: mean={cpu_proc.mean():.2f}, p95={cpu_proc.quantile(0.95):.2f}, max={cpu_proc.max():.2f}"
        )

    if sys_valid and proc_valid and cpu_proc.mean() > (cpu_sys.mean() * 2.0):
        print(
            "[INFO] "
            f"[{exp_id}] Hinweis: cpu_usage ist System-CPU (Host), nicht Inference-Prozess-CPU. "
            "Inference-Last sollte mit ollama_proc_cpu_percent interpretiert werden."
        )


def plot_retrieval_heatmap(testcase_df: pd.DataFrame, out_dir: Path) -> None:
    if testcase_df.empty:
        return
    required = {"test_case_id", "profile_model", "retrieval_score_mean"}
    if not required.issubset(set(testcase_df.columns)):
        return

    pivot = testcase_df.pivot_table(
        index="test_case_id",
        columns="profile_model",
        values="retrieval_score_mean",
        aggfunc="mean",
    )
    ordered_idx = [x for x in _sort_testcase_ids(pivot.index.astype(str).tolist()) if x in pivot.index]
    pivot = pivot.reindex(ordered_idx)

    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title("Retrieval Score Heatmap (Testcase x Profil|Modell)")
    ax.set_xlabel("Profil | Modell")
    ax.set_ylabel("Test Case ID")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("retrieval_score_mean")
    fig.tight_layout()
    fig.savefig(out_dir / "retrieval_heatmap_testcase_profile_model.png", dpi=200)
    plt.close(fig)


def plot_retrieval_histogram(testcase_df: pd.DataFrame, out_dir: Path) -> None:
    if "retrieval_score_mean" not in testcase_df.columns:
        return
    values = testcase_df["retrieval_score_mean"].dropna()
    if values.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=20, color="#2a9d8f", edgecolor="black")
    ax.set_title("Verteilung Retrieval Score (Testcase-Ebene)")
    ax.set_xlabel("retrieval_score_mean")
    ax.set_ylabel("Anzahl")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "retrieval_score_histogram.png", dpi=200)
    plt.close(fig)


def plot_answer_boxplot(trace_df: pd.DataFrame, model_df: pd.DataFrame, out_dir: Path) -> None:
    if not trace_df.empty and {"profile_model", "llm_judge_score"}.issubset(set(trace_df.columns)):
        grouped = []
        labels = []
        for label in sorted(trace_df["profile_model"].dropna().unique().tolist()):
            vals = trace_df.loc[trace_df["profile_model"] == label, "llm_judge_score"].dropna()
            if not vals.empty:
                grouped.append(vals.values)
                labels.append(label)
        if grouped:
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.boxplot(grouped, tick_labels=labels, vert=True, showfliers=False)
            ax.set_title("LLM Judge Score pro Profil|Modell (Run-Ebene)")
            ax.set_ylabel("llm_judge_score")
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / "answer_llm_judge_boxplot.png", dpi=200)
            plt.close(fig)
            return

    if "llm_judge_score_mean" in model_df.columns and "profile_model" in model_df.columns:
        ordered = model_df.sort_values("llm_judge_score_mean", ascending=False)
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.barh(ordered["profile_model"], ordered["llm_judge_score_mean"], color="#457b9d")
        ax.invert_yaxis()
        ax.set_title("LLM Judge Score Mean pro Profil|Modell")
        ax.set_xlabel("llm_judge_score_mean")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "answer_llm_judge_mean_barh.png", dpi=200)
        plt.close(fig)


def plot_latency_with_p95(model_df: pd.DataFrame, out_dir: Path) -> None:
    required = {"profile_model", "latency_s_mean"}
    if not required.issubset(set(model_df.columns)):
        return
    ordered = model_df.sort_values("latency_s_mean", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(16, 8))
    x = range(len(ordered))
    ax.bar(x, ordered["latency_s_mean"], color="#f4a261", label="Mean")
    if "latency_s_p95" in ordered.columns:
        ax.scatter(x, ordered["latency_s_p95"], color="#d62828", s=18, label="P95", zorder=3)
    ax.set_yscale("log")
    ax.set_title("Latenz pro Profil|Modell (Mean + P95, log-Skala)")
    ax.set_ylabel("Latency (s)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ordered["profile_model"], rotation=90, fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_seconds_mean_p95_log.png", dpi=200)
    plt.close(fig)


def plot_tokens_bar(model_df: pd.DataFrame, out_dir: Path) -> None:
    if not {"profile_model", "tokens_per_s_mean"}.issubset(set(model_df.columns)):
        return
    ordered = model_df.sort_values("tokens_per_s_mean", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(16, 8))
    x = range(len(ordered))
    ax.bar(x, ordered["tokens_per_s_mean"], color="#264653")
    ax.set_title("Tokens/s Mean pro Profil|Modell")
    ax.set_ylabel("tokens_per_s_mean")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ordered["profile_model"], rotation=90, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "tokens_per_s_mean_bar.png", dpi=200)
    plt.close(fig)


def plot_pareto(model_df: pd.DataFrame, out_dir: Path) -> None:
    required = {"profile", "profile_model", "latency_s_mean", "llm_judge_score_mean", "tokens_per_s_mean"}
    if not required.issubset(set(model_df.columns)):
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    profiles = sorted(model_df["profile"].dropna().astype(str).unique().tolist())
    palette = ["#1d3557", "#2a9d8f", "#e9c46a", "#e76f51", "#6a4c93"]
    color_map = {p: palette[i % len(palette)] for i, p in enumerate(profiles)}

    size = model_df["tokens_per_s_mean"].clip(lower=0).fillna(0) * 8.0 + 20.0
    for _, row in model_df.iterrows():
        ax.scatter(
            row["latency_s_mean"],
            row["llm_judge_score_mean"],
            s=float(size.loc[_]),
            color=color_map.get(str(row["profile"]), "#666666"),
            alpha=0.8,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.annotate(row["profile_model"], (row["latency_s_mean"], row["llm_judge_score_mean"]), fontsize=7, alpha=0.9)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", label=p, markerfacecolor=color_map[p], markersize=8)
        for p in profiles
    ]
    ax.legend(handles=handles, title="Profile", loc="lower right")
    ax.set_xscale("log")
    ax.set_xlabel("latency_s_mean (log)")
    ax.set_ylabel("llm_judge_score_mean")
    ax.set_title("Pareto-Ansicht: Latenz vs Answer-Score (Bubble = Token/s)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_latency_seconds_vs_answer_bubble_tokens.png", dpi=220)
    plt.close(fig)


def plot_system_metrics(system_df: pd.DataFrame, out_dir: Path, include_gpu_plots: bool) -> None:
    if system_df.empty:
        return

    required_cpu = {"profile", "ollama_proc_cpu_percent"}
    if required_cpu.issubset(set(system_df.columns)):
        cpu_group = (
            system_df.groupby("profile", dropna=False)["ollama_proc_cpu_percent"]
            .agg(["mean", "median", "max"]) 
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        if not cpu_group.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = range(len(cpu_group))
            ax.bar(x, cpu_group["mean"], color="#457b9d", label="Mean")
            ax.scatter(x, cpu_group["max"], color="#d62828", s=22, label="Max", zorder=3)
            ax.set_xticks(list(x))
            ax.set_xticklabels(cpu_group["profile"].astype(str), rotation=30, ha="right")
            ax.set_ylabel("ollama_proc_cpu_percent")
            ax.set_title("Inference CPU-Auslastung je Profil (system_metrics)")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "system_inference_cpu_profile.png", dpi=220)
            plt.close(fig)

    if not include_gpu_plots:
        return

    gpu_candidates = ["gpu_usage", "gpu_memory"]
    if not set(gpu_candidates).intersection(set(system_df.columns)):
        return

    gpu_df = system_df.copy()
    for col in gpu_candidates:
        if col in gpu_df.columns:
            gpu_df[col] = pd.to_numeric(gpu_df[col], errors="coerce")

    gpu_df["gpu_clock_mhz"] = _extract_gpu_clock_mhz(gpu_df)

    has_gpu_signal = False
    if "gpu_usage" in gpu_df.columns and gpu_df["gpu_usage"].notna().any():
        has_gpu_signal = True
    if "gpu_memory" in gpu_df.columns and gpu_df["gpu_memory"].notna().any():
        has_gpu_signal = True
    if "gpu_clock_mhz" in gpu_df.columns and gpu_df["gpu_clock_mhz"].notna().any():
        has_gpu_signal = True
    if not has_gpu_signal:
        return

    agg = gpu_df.groupby("profile", dropna=False).agg(
        gpu_usage_mean=("gpu_usage", "mean") if "gpu_usage" in gpu_df.columns else ("profile", "size"),
        gpu_usage_max=("gpu_usage", "max") if "gpu_usage" in gpu_df.columns else ("profile", "size"),
        gpu_memory_mean=("gpu_memory", "mean") if "gpu_memory" in gpu_df.columns else ("profile", "size"),
        gpu_memory_max=("gpu_memory", "max") if "gpu_memory" in gpu_df.columns else ("profile", "size"),
        gpu_clock_mhz_mean=("gpu_clock_mhz", "mean"),
        gpu_clock_mhz_max=("gpu_clock_mhz", "max"),
    ).reset_index()

    agg = agg.sort_values("profile")

    fig, ax1 = plt.subplots(figsize=(12, 7))
    x = range(len(agg))

    ax1.bar(x, agg["gpu_usage_mean"], color="#2a9d8f", alpha=0.85, label="GPU Usage Mean")
    ax1.scatter(x, agg["gpu_usage_max"], color="#1d3557", s=24, label="GPU Usage Max", zorder=3)
    ax1.set_ylabel("gpu_usage")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(agg["profile"].astype(str), rotation=30, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, agg["gpu_memory_mean"], color="#e76f51", marker="o", linewidth=1.5, label="vRAM Mean")
    ax2.scatter(x, agg["gpu_memory_max"], color="#d62828", s=20, label="vRAM Max", zorder=3)
    ax2.set_ylabel("gpu_memory (vRAM)")

    if agg["gpu_clock_mhz_mean"].notna().any():
        ax2.plot(x, agg["gpu_clock_mhz_mean"], color="#6a4c93", marker="x", linewidth=1.2, label="GPU Clock MHz Mean")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    ax1.set_title("GPU-Auslastung je Profil (GPU Usage + vRAM, optional Clock)")
    ax1.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "system_gpu_usage_vram_profile.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    exp_id = args.experiment_id

    exp_dir = output_root / exp_id
    model_summary = _try_read(exp_dir / "summary_model_profile.csv")
    testcase_summary = _try_read(exp_dir / "summary_testcase.csv")
    trace_joined = _try_read(exp_dir / "trace_joined.csv")
    system_metrics = _try_read(exp_dir / "timeseries_system_metrics.csv")

    if system_metrics is None:
        system_metrics = _try_read(output_root / "metrics" / exp_id / f"system_metrics_{exp_id}.csv")

    # Fallback auf aggregierte Dateien, falls summary_* noch nicht existiert.
    if model_summary is None:
        model_summary = _try_read(output_root / "experiment" / f"model_agg_{exp_id}.csv")
    if testcase_summary is None:
        testcase_summary = _try_read(output_root / "experiment" / f"case_agg_{exp_id}.csv")

    if model_summary is None or testcase_summary is None:
        raise FileNotFoundError(
            "Benötigte Dateien fehlen. Erwartet: output/<exp>/summary_model_profile.csv "
            "und output/<exp>/summary_testcase.csv (oder model_agg/case_agg als Fallback)."
        )

    model_summary = _label_columns(model_summary)
    testcase_summary = _label_columns(testcase_summary)
    trace_joined = _label_columns(trace_joined if trace_joined is not None else pd.DataFrame())
    system_metrics = system_metrics if system_metrics is not None else pd.DataFrame()

    # Optionaler GPU-Ausschluss fuer alle Darstellungen.
    model_summary = _exclude_gpu_profiles(model_summary, args.exclude_gpu)
    testcase_summary = _exclude_gpu_profiles(testcase_summary, args.exclude_gpu)
    trace_joined = _exclude_gpu_profiles(trace_joined, args.exclude_gpu)
    system_metrics = _exclude_gpu_profiles(system_metrics, args.exclude_gpu)

    model_summary = _coerce_numeric(
        model_summary,
        ["retrieval_score_mean", "llm_judge_score_mean", "latency_ms_mean", "latency_ms_p95", "tokens_per_s_mean"],
    )
    model_summary = _ensure_latency_seconds(model_summary)
    testcase_summary = _coerce_numeric(testcase_summary, ["retrieval_score_mean"])
    trace_joined = _coerce_numeric(trace_joined, ["llm_judge_score"])
    system_metrics = _coerce_numeric(
        system_metrics,
        [
            "cpu_usage",
            "memory_usage",
            "ram_used_mb",
            "ram_available_mb",
            "ram_total_mb",
            "ollama_proc_cpu_percent",
            "ollama_proc_rss_mb",
            "gpu_usage",
            "gpu_memory",
            "ts_epoch",
            "trace_elapsed_s",
        ],
    )

    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    validate_system_metrics(system_metrics, exp_id)

    plot_retrieval_heatmap(testcase_summary, plots_dir)
    plot_retrieval_histogram(testcase_summary, plots_dir)
    plot_answer_boxplot(trace_joined, model_summary, plots_dir)
    plot_latency_with_p95(model_summary, plots_dir)
    plot_tokens_bar(model_summary, plots_dir)
    plot_pareto(model_summary, plots_dir)
    plot_system_metrics(system_metrics, plots_dir, include_gpu_plots=not args.exclude_gpu)

    if args.exclude_gpu:
        gpu_plot = plots_dir / "system_gpu_usage_vram_profile.png"
        if gpu_plot.exists():
            gpu_plot.unlink()

    print(f"Plots gespeichert unter: {plots_dir}")


if __name__ == "__main__":
    main()
