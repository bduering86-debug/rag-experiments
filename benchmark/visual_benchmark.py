# bin/visual_benchmark.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from .config import BASE_DIR
from .logging_utils import get_logger

logger = get_logger(__name__)

CSV_PATH = BASE_DIR / "logs" / "ollama_calls.csv"
OUT_DIR = BASE_DIR / "reports" / "benchmarks"


def load_data():
    if not CSV_PATH.exists():
        logger.error("Benchmark-CSV nicht gefunden: %s", CSV_PATH)
        return None

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        logger.warning("Benchmark-CSV ist leer: %s", CSV_PATH)
        return None

    return df


def summarize(df: pd.DataFrame):
    logger.info("===== Gesamt-Summary =====")
    logger.info("Anzahl Zeilen gesamt: %d", len(df))

    # nur Zeilen mit Tokens/s (relevant für Ollama)
    df_tokens = df.dropna(subset=["tokens_per_s"])
    if df_tokens.empty:
        logger.warning("Keine Tokens/s-Daten vorhanden.")
        return

    logger.info("=== Durchschnitt Tokens/s pro Modell ===")
    by_model = df_tokens.groupby("model")["tokens_per_s"].mean().sort_values(ascending=False)
    for model, tps in by_model.items():
        logger.info("Modell %-30s Ø Tokens/s = %.2f", model, tps)

    logger.info("=== Durchschnitt wall_s pro Modell ===")
    by_model_wall = df_tokens.groupby("model")["wall_s"].mean().sort_values()
    for model, w in by_model_wall.items():
        logger.info("Modell %-30s Ø wall_s = %.2f", model, w)


def plot_tokens_per_model(df: pd.DataFrame):
    df_tokens = df.dropna(subset=["tokens_per_s"])
    if df_tokens.empty:
        logger.warning("Keine Tokens/s-Daten für Plot.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "tokens_per_s_per_model.png"

    plt.figure()
    # Boxplot: Tokens/s nach Modell
    df_tokens.boxplot(column="tokens_per_s", by="model", rot=45)
    plt.suptitle("")  # default Titel von Pandas entfernen
    plt.title("Tokens/s pro Modell")
    plt.xlabel("Modell")
    plt.ylabel("Tokens/s")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    logger.info("Plot gespeichert: %s", out_file)


def plot_walltime_per_model(df: pd.DataFrame):
    df_tokens = df.dropna(subset=["tokens_per_s"])
    if df_tokens.empty:
        logger.warning("Keine Daten für wall_s-Plot.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "walltime_per_model.png"

    plt.figure()
    df_tokens.boxplot(column="wall_s", by="model", rot=45)
    plt.suptitle("")
    plt.title("Antwortzeit (wall_s) pro Modell")
    plt.xlabel("Modell")
    plt.ylabel("Sekunden")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    logger.info("Plot gespeichert: %s", out_file)


def main():
    df = load_data()
    if df is None:
        return

    summarize(df)
    plot_tokens_per_model(df)
    plot_walltime_per_model(df)


if __name__ == "__main__":
    main()
