#!/usr/bin/env python3
"""Aggregiert Experimentmetriken aus Runs-, Scores- und System-Metrics-CSV.

Beispiel:
    python3 src/scripts/evaluate_experiment_metrics.py --experiment-id 20260215_093928
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_csv.utils.retrieval_score import RetrievalScoreCalculator
from rag_csv.utils.token_score import TokenScoreCalculator
from rag_csv.utils.latency_score import LatencyScoreCalculator


RUNS_GLOB = "runs_*.csv"
SCORES_GLOB = "scores_*.csv"
SYSTEM_GLOB = "system_metrics_*.csv"


@dataclass
class ExperimentFiles:
    runs_files: list[Path]
    scores_file: Path | None
    system_metrics_file: Path | None


class ExperimentFileLocator:
    def __init__(self, project_root: Path, output_root: Path) -> None:
        self.project_root = project_root
        self.output_root = output_root

    def locate(self, experiment_id: str) -> ExperimentFiles:
        runs_dir = self.output_root / "experiment"
        runs_candidates = sorted(runs_dir.glob(RUNS_GLOB))

        runs_files: list[Path] = []
        for path in runs_candidates:
            try:
                sample = pd.read_csv(path, usecols=["experiment_id"], nrows=30000)
            except Exception:
                continue
            if "experiment_id" in sample.columns and (sample["experiment_id"].astype(str) == experiment_id).any():
                runs_files.append(path)

        scores_file = self._locate_scores_file(runs_dir, experiment_id)
        system_metrics_file = self._locate_system_metrics_file(experiment_id)

        return ExperimentFiles(
            runs_files=runs_files,
            scores_file=scores_file,
            system_metrics_file=system_metrics_file,
        )

    def _locate_scores_file(self, runs_dir: Path, experiment_id: str) -> Path | None:
        direct = runs_dir / f"scores_{experiment_id}.csv"
        if direct.exists():
            return direct

        candidates = sorted(runs_dir.glob(SCORES_GLOB))
        for path in candidates:
            if experiment_id in path.name:
                return path
        return None

    def _locate_system_metrics_file(self, experiment_id: str) -> Path | None:
        direct = self.output_root / "metrics" / experiment_id / f"system_metrics_{experiment_id}.csv"
        if direct.exists():
            return direct

        candidates = sorted((self.output_root / "metrics").rglob(SYSTEM_GLOB))
        for path in candidates:
            if experiment_id in path.name or experiment_id in str(path.parent):
                return path
        return None


class ExperimentDataLoader:
    def __init__(self, files: ExperimentFiles, experiment_id: str) -> None:
        self.files = files
        self.experiment_id = experiment_id

    def load_runs(self) -> pd.DataFrame:
        if not self.files.runs_files:
            raise FileNotFoundError(
                f"Keine runs_*.csv mit experiment_id={self.experiment_id} unter output/experiment gefunden."
            )

        chunks = [pd.read_csv(path) for path in self.files.runs_files]
        runs = pd.concat(chunks, ignore_index=True)
        runs = runs[runs["experiment_id"].astype(str) == self.experiment_id].copy()

        key_cols = ["trace_id", "profile", "model", "test_case_id", "repetition"]
        existing_keys = [col for col in key_cols if col in runs.columns]
        if existing_keys:
            runs = runs.drop_duplicates(subset=existing_keys, keep="last")

        return runs.reset_index(drop=True)

    def load_scores(self) -> pd.DataFrame:
        if self.files.scores_file is None:
            return pd.DataFrame(
                columns=["profile", "model", "token_score_norm", "latency_score", "retrieval_score", "answer_score"]
            )

        scores = pd.read_csv(self.files.scores_file)
        # Rückwärtskompatibilität: ältere Exporte hatten die Spalte "token_score".
        if "token_score_norm" not in scores.columns and "token_score" in scores.columns:
            scores["token_score_norm"] = scores["token_score"]
        return scores

    def load_system_metrics(self, trace_ids: Iterable[str]) -> pd.DataFrame:
        if self.files.system_metrics_file is None:
            return pd.DataFrame()

        metrics = pd.read_csv(self.files.system_metrics_file)
        if "trace_id" in metrics.columns:
            metrics = metrics[metrics["trace_id"].isin(set(trace_ids))].copy()
        return metrics.reset_index(drop=True)


class MetricsAggregator:
    RUN_NUMERIC_COLUMNS = [
        "recall@k",
        "ndcg@k",
        "retrieval_score",
        "latency_ms",
        "total_wall_time_ms",
        "tokens_per_s",
        "prompt_tokens_per_s",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "total_duration_s",
        "load_duration_s",
        "prompt_eval_duration_s",
        "eval_duration_s",
        "llm_judge_f",
        "llm_judge_r",
        "llm_judge_c",
        "llm_judge_l",
        "llm_judge_score",
        "error_flag",
    ]

    SYSTEM_NUMERIC_COLUMNS = [
        "cpu_usage",
        "memory_usage",
        "ram_used_mb",
        "ram_available_mb",
        "ram_total_mb",
        "ollama_proc_cpu_percent",
        "ollama_proc_rss_mb",
        "gpu_usage",
        "gpu_memory",
    ]

    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        self.retrieval_interpreter = RetrievalScoreCalculator()

    def normalize_runs(self, runs: pd.DataFrame) -> pd.DataFrame:
        df = runs.copy()
        for col in self.RUN_NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "total_duration_s" in df.columns:
            df["total_duration_ms"] = df["total_duration_s"] * 1000.0
            df["total_duration_min"] = df["total_duration_s"] / 60.0
        if "total_wall_time_ms" in df.columns:
            df["total_wall_time_min"] = df["total_wall_time_ms"] / 60000.0

        return df

    def normalize_system_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        if metrics.empty:
            return metrics.copy()

        df = metrics.copy()
        for col in self.SYSTEM_NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "ts_epoch" in df.columns:
            df["ts_epoch"] = pd.to_numeric(df["ts_epoch"], errors="coerce")
            first_ts = df.groupby("trace_id")["ts_epoch"].transform("min")
            df["trace_elapsed_s"] = df["ts_epoch"] - first_ts

        return df

    def build_trace_system_aggregate(self, metrics: pd.DataFrame) -> pd.DataFrame:
        if metrics.empty:
            return pd.DataFrame(columns=["trace_id"])

        agg_spec: dict[str, list[str]] = {}
        for col in self.SYSTEM_NUMERIC_COLUMNS:
            if col in metrics.columns:
                agg_spec[col] = ["mean", "median", "min", "max", "std"]

        if "ts_epoch" in metrics.columns:
            agg_spec["ts_epoch"] = ["min", "max", "count"]

        grouped = metrics.groupby("trace_id", dropna=False).agg(agg_spec)
        grouped.columns = [f"sys_{col}_{stat}" for col, stat in grouped.columns]
        grouped = grouped.reset_index()

        if "sys_ts_epoch_min" in grouped.columns and "sys_ts_epoch_max" in grouped.columns:
            grouped["sys_trace_duration_s"] = grouped["sys_ts_epoch_max"] - grouped["sys_ts_epoch_min"]
            grouped["sys_trace_duration_min"] = grouped["sys_trace_duration_s"] / 60.0

        return grouped

    def join_trace_level(self, runs: pd.DataFrame, trace_system: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
        joined = runs.copy()
        if not trace_system.empty:
            joined = joined.merge(trace_system, on="trace_id", how="left")

        if not scores.empty and "model" in scores.columns:
            scores_for_join = scores.copy()
            rename_map = {}
            if "retrieval_score" in scores_for_join.columns:
                rename_map["retrieval_score"] = "retrieval_score_cfg"
            if "answer_score" in scores_for_join.columns:
                rename_map["answer_score"] = "answer_score_cfg"
            if rename_map:
                scores_for_join = scores_for_join.rename(columns=rename_map)

            if "profile" in scores.columns and "profile" in joined.columns:
                joined = joined.merge(scores_for_join, on=["profile", "model"], how="left")
            else:
                joined = joined.merge(scores_for_join, on="model", how="left")

        return joined

    def summarize_model_profile(self, joined_trace: pd.DataFrame) -> pd.DataFrame:
        grouping = ["experiment_id", "profile", "model"]
        summary = self._summarize(joined_trace, grouping)
        return self._add_interpretations(summary)

    def summarize_testcase(self, joined_trace: pd.DataFrame) -> pd.DataFrame:
        grouping = ["experiment_id", "profile", "model", "test_case_id"]
        summary = self._summarize(joined_trace, grouping)
        return self._add_interpretations(summary)

    def build_kpi_long(self, summary: pd.DataFrame, id_columns: list[str]) -> pd.DataFrame:
        metric_cols = [col for col in summary.columns if col not in id_columns]
        if not metric_cols:
            return pd.DataFrame(columns=id_columns + ["metric", "value"])

        long_df = summary.melt(
            id_vars=id_columns,
            value_vars=metric_cols,
            var_name="metric",
            value_name="value",
        )
        return long_df

    def _summarize(self, frame: pd.DataFrame, grouping: list[str]) -> pd.DataFrame:
        available_grouping = [col for col in grouping if col in frame.columns]
        if not available_grouping:
            return pd.DataFrame()

        summary = frame.groupby(available_grouping, dropna=False).size().reset_index(name="run_count")

        stats_columns = [
            "recall@k",
            "ndcg@k",
            "retrieval_score",
            "latency_ms",
            "total_wall_time_ms",
            "total_wall_time_min",
            "total_duration_s",
            "total_duration_ms",
            "total_duration_min",
            "tokens_per_s",
            "prompt_tokens_per_s",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "load_duration_s",
            "prompt_eval_duration_s",
            "eval_duration_s",
            "llm_judge_f",
            "llm_judge_r",
            "llm_judge_c",
            "llm_judge_l",
            "llm_judge_score",
            "error_flag",
            "token_score_norm",
            "token_score",
            "latency_score",
            "retrieval_score_cfg",
            "answer_score_cfg",
            "sys_trace_duration_s",
            "sys_cpu_usage_mean",
            "sys_memory_usage_mean",
            "sys_ram_used_mb_mean",
            "sys_ram_available_mb_mean",
            "sys_ollama_proc_cpu_percent_mean",
            "sys_ollama_proc_rss_mb_mean",
            "sys_gpu_usage_mean",
            "sys_gpu_memory_mean",
        ]

        for col in stats_columns:
            if col not in frame.columns:
                continue

            grouped = frame.groupby(available_grouping, dropna=False)[col]
            tmp = grouped.agg(["mean", "median", "min", "max", "std", "count"]).reset_index()
            tmp = tmp.rename(
                columns={
                    "mean": f"{col}_mean",
                    "median": f"{col}_median",
                    "min": f"{col}_min",
                    "max": f"{col}_max",
                    "std": f"{col}_std",
                    "count": f"{col}_count",
                }
            )

            p95 = grouped.quantile(0.95).reset_index(name=f"{col}_p95")
            tmp = tmp.merge(p95, on=available_grouping, how="left")
            summary = summary.merge(tmp, on=available_grouping, how="left")

            if col in {"total_duration_min", "total_wall_time_min", "total_duration_s", "total_wall_time_ms"}:
                col_sum = grouped.sum(min_count=1).reset_index(name=f"{col}_sum")
                summary = summary.merge(col_sum, on=available_grouping, how="left")

        return summary

    def _add_interpretations(self, summary: pd.DataFrame) -> pd.DataFrame:
        if summary.empty:
            return summary

        out = summary.copy()
        recall_col = "recall@k_mean"
        ndcg_col = "ndcg@k_mean"
        retrieval_col = "retrieval_score_mean"

        if recall_col in out.columns and ndcg_col in out.columns:
            out["retrieval_interpretation"] = out.apply(
                lambda row: self.retrieval_interpreter.get_interpretation(
                    self._safe_float(row.get(recall_col)),
                    self._safe_float(row.get(ndcg_col)),
                ),
                axis=1,
            )
        else:
            out["retrieval_interpretation"] = "Keine Daten verfügbar"

        if recall_col in out.columns:
            out["recall_interpretation"] = out[recall_col].apply(self._recall_text)
        else:
            out["recall_interpretation"] = "Keine Daten verfügbar"

        if retrieval_col in out.columns:
            out["retrieval_score_interpretation"] = out[retrieval_col].apply(self._retrieval_score_text)
        else:
            out["retrieval_score_interpretation"] = "Keine Daten verfügbar"

        judge_col = "llm_judge_score_mean"
        if judge_col in out.columns:
            out["llm_judge_interpretation"] = out[judge_col].apply(self._judge_text)
        else:
            out["llm_judge_interpretation"] = "Keine Daten verfügbar"

        out["overall_assessment_text"] = out.apply(self._overall_assessment_text, axis=1)

        return out

    @staticmethod
    def _safe_float(value: object) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _recall_text(value: object) -> str:
        val = MetricsAggregator._safe_float(value)
        if val is None:
            return "Keine Daten verfügbar"
        if val >= 0.9:
            return "Sehr hoher Recall"
        if val >= 0.7:
            return "Guter Recall"
        if val >= 0.4:
            return "Mittlerer Recall"
        if val >= 0.2:
            return "Niedriger Recall"
        return "Sehr niedriger Recall"

    @staticmethod
    def _retrieval_score_text(value: object) -> str:
        val = MetricsAggregator._safe_float(value)
        if val is None:
            return "Keine Daten verfügbar"
        if val >= 0.85:
            return "Sehr starke Retrieval-Qualitaet"
        if val >= 0.65:
            return "Gute Retrieval-Qualitaet"
        if val >= 0.4:
            return "Mittlere Retrieval-Qualitaet"
        if val >= 0.2:
            return "Schwache Retrieval-Qualitaet"
        return "Sehr schwache Retrieval-Qualitaet"

    @staticmethod
    def _judge_text(value: object) -> str:
        val = MetricsAggregator._safe_float(value)
        if val is None:
            return "Keine Judge-Daten verfügbar"
        if val >= 0.85:
            return "Sehr hohe Antwortqualitaet laut LLM Judge"
        if val >= 0.7:
            return "Gute Antwortqualitaet laut LLM Judge"
        if val >= 0.5:
            return "Mittlere Antwortqualitaet laut LLM Judge"
        if val >= 0.3:
            return "Niedrige Antwortqualitaet laut LLM Judge"
        return "Sehr niedrige Antwortqualitaet laut LLM Judge"

    def _overall_assessment_text(self, row: pd.Series) -> str:
        retrieval = self._safe_float(row.get("retrieval_score_cfg_mean"))
        if retrieval is None:
            retrieval = self._safe_float(row.get("retrieval_score_mean"))

        judge = self._safe_float(row.get("answer_score_cfg_mean"))
        if judge is None:
            judge = self._safe_float(row.get("llm_judge_score_mean"))

        latency_score = self._safe_float(row.get("latency_score_mean"))
        token_score = self._safe_float(row.get("token_score_norm_mean"))
        if token_score is None:
            token_score = self._safe_float(row.get("token_score_mean"))

        parts: list[str] = []
        if retrieval is not None:
            parts.append(f"Retrieval={retrieval:.3f}")
        if judge is not None:
            parts.append(f"Judge={judge:.3f}")
        if latency_score is not None:
            parts.append(f"Latency-Score={latency_score:.3f}")
        if token_score is not None:
            parts.append(f"Token-Score={token_score:.3f}")

        score_components = [v for v in [retrieval, judge, latency_score, token_score] if v is not None]
        if not score_components:
            quality = "Keine Gesamtbewertung möglich"
        else:
            avg = sum(score_components) / len(score_components)
            if avg >= 0.8:
                quality = "Gesamtbewertung: sehr stark"
            elif avg >= 0.65:
                quality = "Gesamtbewertung: gut"
            elif avg >= 0.45:
                quality = "Gesamtbewertung: mittel"
            elif avg >= 0.25:
                quality = "Gesamtbewertung: schwach"
            else:
                quality = "Gesamtbewertung: sehr schwach"

        if not parts:
            return quality
        return f"{quality} ({', '.join(parts)})"


class MetricsExporter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        summary_model_profile: pd.DataFrame,
        summary_testcase: pd.DataFrame,
        trace_joined: pd.DataFrame,
        timeseries_system_metrics: pd.DataFrame,
        summary_long: pd.DataFrame,
        testcase_long: pd.DataFrame,
    ) -> dict[str, Path]:
        paths = {
            "summary_model_profile": self.output_dir / "summary_model_profile.csv",
            "summary_testcase": self.output_dir / "summary_testcase.csv",
            "trace_joined": self.output_dir / "trace_joined.csv",
            "timeseries_system_metrics": self.output_dir / "timeseries_system_metrics.csv",
            "summary_model_profile_long": self.output_dir / "summary_model_profile_long.csv",
            "summary_testcase_long": self.output_dir / "summary_testcase_long.csv",
            "summary_markdown": self.output_dir / "summary.md",
        }

        summary_model_profile.to_csv(paths["summary_model_profile"], index=False)
        summary_testcase.to_csv(paths["summary_testcase"], index=False)
        trace_joined.to_csv(paths["trace_joined"], index=False)
        timeseries_system_metrics.to_csv(paths["timeseries_system_metrics"], index=False)
        summary_long.to_csv(paths["summary_model_profile_long"], index=False)
        testcase_long.to_csv(paths["summary_testcase_long"], index=False)
        self._write_markdown_summary(
            paths["summary_markdown"],
            summary_model_profile,
            summary_testcase,
            trace_joined,
            timeseries_system_metrics,
        )

        return paths

    def _write_markdown_summary(
        self,
        target: Path,
        summary_model_profile: pd.DataFrame,
        summary_testcase: pd.DataFrame,
        trace_joined: pd.DataFrame,
        timeseries_system_metrics: pd.DataFrame,
    ) -> None:
        model_cols = [
            "experiment_id",
            "profile",
            "model",
            "run_count",
            "latency_ms_mean",
            "latency_ms_p95",
            "tokens_per_s_mean",
            "tokens_per_s_p95",
            "prompt_tokens_per_s_mean",
            "prompt_tokens_per_s_p95",
            "prompt_tokens_mean",
            "completion_tokens_mean",
            "total_tokens_mean",
            "total_wall_time_ms_mean",
            "total_wall_time_min_mean",
            "total_wall_time_min_sum",
            "total_duration_s_mean",
            "total_duration_min_mean",
            "total_duration_min_sum",
            "sys_cpu_usage_mean_mean",
            "sys_memory_usage_mean_mean",
            "sys_ollama_proc_rss_mb_mean_mean",
            "sys_gpu_usage_mean_mean",
            "sys_gpu_memory_mean_mean",
            "token_score_mean",
            "latency_score_mean",
            "retrieval_score_cfg_mean",
            "answer_score_cfg_mean",
            "recall@k_mean",
            "retrieval_score_mean",
            "llm_judge_score_mean",
            "retrieval_interpretation",
            "recall_interpretation",
            "retrieval_score_interpretation",
            "llm_judge_interpretation",
            "overall_assessment_text",
        ]
        testcase_cols = [
            "experiment_id",
            "profile",
            "model",
            "test_case_id",
            "run_count",
            "latency_ms_mean",
            "tokens_per_s_mean",
            "prompt_tokens_per_s_mean",
            "prompt_tokens_mean",
            "completion_tokens_mean",
            "total_tokens_mean",
            "total_duration_s_mean",
            "total_duration_min_mean",
            "total_duration_min_sum",
            "total_wall_time_min_sum",
            "recall@k_mean",
            "ndcg@k_mean",
            "retrieval_score_mean",
            "retrieval_score_cfg_mean",
            "llm_judge_score_mean",
            "answer_score_cfg_mean",
            "retrieval_interpretation",
            "recall_interpretation",
            "retrieval_score_interpretation",
            "llm_judge_interpretation",
            "overall_assessment_text",
        ]

        model_table = self._table_for_markdown(summary_model_profile, model_cols)
        testcase_table = self._table_for_markdown(summary_testcase, testcase_cols)
        narrative = self._build_profile_model_text(summary_model_profile)
        total_runtime_min = self._safe_series_sum(trace_joined, "total_duration_min")
        total_wall_time_min = self._safe_series_sum(trace_joined, "total_wall_time_min")
        traces = int(trace_joined["trace_id"].nunique()) if "trace_id" in trace_joined.columns else 0
        runs = len(trace_joined.index)
        sys_samples = len(timeseries_system_metrics.index)

        content = [
            "# Experiment Summary",
            "",
            "## Overview",
            f"- Runs: {runs}",
            f"- Trace IDs: {traces}",
            f"- System metric samples: {sys_samples}",
            f"- Aggregierte Laufzeit (total_duration): {total_runtime_min:.4f} min",
            f"- Aggregierte Laufzeit (total_wall_time): {total_wall_time_min:.4f} min",
            "",
            "## Textuelle Zusammenfassung (Profil x Modell)",
            narrative,
            "",
            "## Model x Profile",
            model_table,
            "",
            "## Test Case x Model x Profile",
            testcase_table,
            "",
        ]
        target.write_text("\n".join(content), encoding="utf-8")

    def _table_for_markdown(self, frame: pd.DataFrame, preferred_cols: list[str], max_rows: int = 30) -> str:
        if frame.empty:
            return "_Keine Daten vorhanden._"

        cols = [col for col in preferred_cols if col in frame.columns]
        if not cols:
            return "_Keine passenden Spalten gefunden._"

        subset = frame[cols].copy().head(max_rows)
        for col in subset.columns:
            if pd.api.types.is_float_dtype(subset[col]):
                subset[col] = subset[col].round(4)

        return self._df_to_markdown(subset)

    def _build_profile_model_text(self, summary_model_profile: pd.DataFrame) -> str:
        if summary_model_profile.empty:
            return "_Keine Daten vorhanden._"

        lines: list[str] = []
        sort_cols = [col for col in ["profile", "model"] if col in summary_model_profile.columns]
        frame = summary_model_profile.sort_values(sort_cols) if sort_cols else summary_model_profile

        for _, row in frame.iterrows():
            profile = row.get("profile", "n/a")
            model = row.get("model", "n/a")
            run_count = self._fmt(row.get("run_count"))
            latency = self._fmt(row.get("latency_ms_mean"), 2)
            tps = self._fmt(row.get("tokens_per_s_mean"), 4)
            prompt_tps = self._fmt(row.get("prompt_tokens_per_s_mean"), 4)
            recall = self._fmt(row.get("recall@k_mean"), 4)
            retrieval = self._fmt(
                row.get("retrieval_score_cfg_mean")
                if row.get("retrieval_score_cfg_mean") is not None
                else row.get("retrieval_score_mean"),
                4,
            )
            judge = self._fmt(
                row.get("answer_score_cfg_mean")
                if row.get("answer_score_cfg_mean") is not None
                else row.get("llm_judge_score_mean"),
                4,
            )
            duration_sum = self._fmt(row.get("total_duration_min_sum"), 4)
            wall_sum = self._fmt(row.get("total_wall_time_min_sum"), 4)
            gpu_usage = self._fmt(row.get("sys_gpu_usage_mean_mean"), 2)
            gpu_memory = self._fmt(row.get("sys_gpu_memory_mean_mean"), 2)
            retrieval_text = row.get("retrieval_interpretation", "Keine Daten verfügbar")
            recall_text = row.get("recall_interpretation", "Keine Daten verfügbar")
            retrieval_score_text = row.get("retrieval_score_interpretation", "Keine Daten verfügbar")
            judge_text = row.get("llm_judge_interpretation", "Keine Daten verfügbar")
            overall_text = row.get("overall_assessment_text", "Keine Gesamtbewertung möglich")

            lines.append(f"### Profil `{profile}` / Modell `{model}`")
            lines.append(f"Runs: {run_count}")
            lines.append(f"Performance: Latenz-Mean {latency} ms, Tokens/s-Mean {tps}, Prompt-Tokens/s-Mean {prompt_tps}")
            lines.append(
                f"Retrieval: Recall-Mean {recall} ({recall_text}), Retrieval-Score-Mean {retrieval} "
                f"({retrieval_score_text}), Interpretation: {retrieval_text}"
            )
            lines.append(f"LLM Judge: Score-Mean {judge}, Bewertung: {judge_text}")
            lines.append(f"System: GPU-Usage-Mean {gpu_usage}, GPU-Memory-Mean {gpu_memory}")
            lines.append(f"Laufzeit: Gesamtlaufzeit {duration_sum} min, Walltime-Summe {wall_sum} min")
            lines.append(f"Gesamtbewertung: {overall_text}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _safe_series_sum(frame: pd.DataFrame, column: str) -> float:
        if column not in frame.columns:
            return 0.0
        return float(pd.to_numeric(frame[column], errors="coerce").sum())

    @staticmethod
    def _fmt(value: object, digits: int = 0) -> str:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if pd.isna(val):
            return "n/a"
        return f"{val:.{digits}f}"

    def _df_to_markdown(self, frame: pd.DataFrame) -> str:
        headers = [str(col) for col in frame.columns]
        sep = ["---"] * len(headers)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for _, row in frame.iterrows():
            values = []
            for value in row.tolist():
                if pd.isna(value):
                    values.append("")
                else:
                    text = str(value).replace("|", "\\|")
                    values.append(text)
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)


class EvaluationPipeline:
    def __init__(
        self,
        project_root: Path,
        output_root: Path,
        export_dir: Path,
        *,
        exclude_gpu: bool = False,
        recompute_scores_from_runs: bool = False,
        flat_export: bool = False,
        save_scores_to_export: bool = False,
    ) -> None:
        self.project_root = project_root
        self.output_root = output_root
        self.export_dir = export_dir
        self.exclude_gpu = exclude_gpu
        self.recompute_scores_from_runs = recompute_scores_from_runs
        self.flat_export = flat_export
        self.save_scores_to_export = save_scores_to_export

    @staticmethod
    def _filter_gpu_profile(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "profile" not in df.columns:
            return df
        out = df.copy()
        return out[out["profile"].astype(str).str.lower() != "gpu"].copy().reset_index(drop=True)

    @staticmethod
    def _recompute_scores_from_runs(runs: pd.DataFrame) -> pd.DataFrame:
        if runs.empty:
            return pd.DataFrame(
                columns=["profile", "model", "token_score_norm", "latency_score", "retrieval_score", "answer_score"]
            )

        work = runs.copy()
        work["tokens_per_s"] = pd.to_numeric(work.get("tokens_per_s"), errors="coerce")
        work["latency_ms"] = pd.to_numeric(work.get("latency_ms"), errors="coerce")
        work["retrieval_score"] = pd.to_numeric(work.get("retrieval_score"), errors="coerce")
        work["llm_judge_score"] = pd.to_numeric(work.get("llm_judge_score"), errors="coerce")

        prepared_results: list[dict[str, object]] = []
        for _, row in work.iterrows():
            profile = row.get("profile")
            model = row.get("model")
            if pd.isna(profile) or pd.isna(model):
                continue

            latency_ms = row.get("latency_ms")
            latency_s = None
            if latency_ms is not None and not pd.isna(latency_ms):
                latency_s = float(latency_ms) / 1000.0

            prepared_results.append(
                {
                    "profile": str(profile),
                    "model": str(model),
                    "tokens_per_second": None if pd.isna(row.get("tokens_per_s")) else float(row.get("tokens_per_s")),
                    "total_latency": latency_s,
                    "retrieval_score": None if pd.isna(row.get("retrieval_score")) else float(row.get("retrieval_score")),
                    "judge_normalized_score": None
                    if pd.isna(row.get("llm_judge_score"))
                    else float(row.get("llm_judge_score")),
                }
            )

        token_scores = TokenScoreCalculator().calculate_scores(prepared_results, group_by_keys=("profile", "model"))
        latency_scores = LatencyScoreCalculator().calculate_scores(prepared_results, group_by_keys=("profile", "model"))

        retrieval_values_by_combo: dict[tuple[str, str], list[float]] = defaultdict(list)
        answer_values_by_combo: dict[tuple[str, str], list[float]] = defaultdict(list)
        for result in prepared_results:
            combo = (str(result["profile"]), str(result["model"]))
            retrieval_value = result.get("retrieval_score")
            if retrieval_value is not None:
                retrieval_values_by_combo[combo].append(float(retrieval_value))
            answer_value = result.get("judge_normalized_score")
            if answer_value is not None:
                answer_values_by_combo[combo].append(float(answer_value))

        retrieval_scores = {
            combo: statistics.mean(values)
            for combo, values in retrieval_values_by_combo.items()
            if values
        }
        answer_scores = {
            combo: statistics.mean(values)
            for combo, values in answer_values_by_combo.items()
            if values
        }

        all_combinations = set(token_scores.keys()) | set(latency_scores.keys()) | set(retrieval_scores.keys()) | set(answer_scores.keys())

        records = []
        for profile, model in sorted(all_combinations):
            records.append(
                {
                    "profile": profile,
                    "model": model,
                    "token_score_norm": token_scores.get((profile, model)),
                    "latency_score": latency_scores.get((profile, model)),
                    "retrieval_score": retrieval_scores.get((profile, model)),
                    "answer_score": answer_scores.get((profile, model)),
                }
            )
        return pd.DataFrame.from_records(records)

    def run(self, experiment_id: str) -> dict[str, Path]:
        locator = ExperimentFileLocator(self.project_root, self.output_root)
        files = locator.locate(experiment_id)

        loader = ExperimentDataLoader(files, experiment_id)
        aggregator = MetricsAggregator(experiment_id)

        runs = loader.load_runs()
        if self.exclude_gpu:
            runs = self._filter_gpu_profile(runs)
        runs = aggregator.normalize_runs(runs)

        if self.recompute_scores_from_runs:
            scores = self._recompute_scores_from_runs(runs)
        else:
            scores = loader.load_scores()

        if self.exclude_gpu and not scores.empty:
            scores = self._filter_gpu_profile(scores)

        for col in ["token_score_norm", "token_score", "latency_score", "retrieval_score", "answer_score"]:
            if col in scores.columns:
                scores[col] = pd.to_numeric(scores[col], errors="coerce")

        if self.save_scores_to_export:
            target_dir = self.export_dir if self.flat_export else (self.export_dir / experiment_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            scores.to_csv(target_dir / f"scores_{experiment_id}.csv", index=False)

        system_metrics = loader.load_system_metrics(runs["trace_id"].dropna().astype(str).unique())
        system_metrics = aggregator.normalize_system_metrics(system_metrics)

        trace_system = aggregator.build_trace_system_aggregate(system_metrics)
        trace_joined = aggregator.join_trace_level(runs, trace_system, scores)

        summary_model_profile = aggregator.summarize_model_profile(trace_joined)
        summary_testcase = aggregator.summarize_testcase(trace_joined)

        summary_long = aggregator.build_kpi_long(
            summary_model_profile,
            ["experiment_id", "profile", "model"],
        )
        testcase_long = aggregator.build_kpi_long(
            summary_testcase,
            ["experiment_id", "profile", "model", "test_case_id"],
        )

        exporter_target = self.export_dir if self.flat_export else (self.export_dir / experiment_id)
        exporter = MetricsExporter(exporter_target)
        return exporter.export(
            summary_model_profile=summary_model_profile,
            summary_testcase=summary_testcase,
            trace_joined=trace_joined,
            timeseries_system_metrics=system_metrics,
            summary_long=summary_long,
            testcase_long=testcase_long,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregiere Metriken pro Experiment (runs + scores + system metrics)."
    )
    parser.add_argument("--experiment-id", required=True, help="Experiment-ID, z. B. 20260215_093928")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Projekt-Root (Default: aktuelles Verzeichnis)",
    )
    parser.add_argument(
        "--output-root",
        default="output",
        help="Root der Eingabedaten (Default: output)",
    )
    parser.add_argument(
        "--export-root",
        default="output",
        help="Zielordner fuer Exporte (Default: output; Ergebnis unter output/<ExperimentID>)",
    )
    parser.add_argument(
        "--exclude-gpu",
        action="store_true",
        help="Schliesst Profil 'gpu' aus Runs, Scores und System-Metriken aus.",
    )
    parser.add_argument(
        "--recompute-scores-from-runs",
        action="store_true",
        help="Berechnet Scores neu aus runs_*.csv statt scores_*.csv zu laden.",
    )
    parser.add_argument(
        "--flat-export",
        action="store_true",
        help="Exportiert direkt in --export-root (ohne Unterordner <ExperimentID>).",
    )
    parser.add_argument(
        "--save-scores",
        action="store_true",
        help="Speichert die verwendeten/reberechneten Scores als scores_<ExperimentID>.csv im Exportordner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    export_root = (project_root / args.export_root).resolve()

    pipeline = EvaluationPipeline(
        project_root,
        output_root,
        export_root,
        exclude_gpu=args.exclude_gpu,
        recompute_scores_from_runs=args.recompute_scores_from_runs,
        flat_export=args.flat_export,
        save_scores_to_export=args.save_scores,
    )
    output_files = pipeline.run(args.experiment_id)

    print("Export abgeschlossen:")
    for name, path in output_files.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
