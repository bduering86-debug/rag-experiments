#!/usr/bin/env python3
"""
Token Score Calculator für Modell-Performance-Evaluation.

Berechnet normalisierte Token-Scores basierend auf Token/s über alle Testfälle.
"""

from typing import List, Dict, Any

import pandas as pd

from rag_csv.config.logging import get_logger


class TokenScoreCalculator:
    """
    Berechnet einen normalisierten Token-Score für Modelle über alle Testfälle.
    
    Der Score wird nach folgender Formel berechnet:
    TokenScore_norm = (Token/s - p1(Token/s)) / (p99(Token/s) - p1(Token/s))
    
    Dabei werden p1 und p99 standardmäßig über ALLE gültigen Messungen berechnet.
    Anschließend wird jede Messung global normalisiert und auf [0, 1] geclamped.
    Damit ist der Score zwischen unterschiedlichen Modellen direkt vergleichbar.
    Score-Range: [0, 1], wobei höhere Werte = bessere Performance (höhere Token/s)
    """
    
    def __init__(self):
        """Initialisiert den TokenScoreCalculator."""
        self.logger = get_logger(f"{__name__}.TokenScoreCalculator")
    
    def calculate_scores(
        self,
        results: List[Dict[str, Any]],
        group_by_keys: tuple[str, ...] = ("model",),
    ) -> Dict[str | tuple[str, ...], float]:
        """
        Berechnet Token-Scores für alle Modelle.
        
        Args:
            results: Liste der Evaluation-Ergebnisse mit tokens_per_second Werten
            
        Returns:
            Dict[str | tuple[str, ...], float]:
                Dictionary mit group key -> normalized_token_score
        """
        if not results:
            self.logger.warning("Keine Token/s Daten gefunden für Score-Berechnung")
            return {}

        df = pd.DataFrame(results).copy()

        if "model" not in df.columns:
            self.logger.warning("Spalte 'model' fehlt, kann Token-Score-Normierung nicht berechnen")
            return {}

        # tokens_per_second robust in numerische Werte konvertieren.
        df["tokens_per_second"] = pd.to_numeric(df.get("tokens_per_second"), errors="coerce")
        df = df.dropna(subset=["model", "tokens_per_second"]).copy()

        if df.empty:
            self.logger.warning("Keine gültigen Token/s Daten gefunden für Score-Berechnung")
            return {}

        # Gruppierungsspalten validieren und als String normalisieren,
        # damit Dict-Keys stabil reproduzierbar sind.
        missing_group_keys = [key for key in group_by_keys if key not in df.columns]
        if missing_group_keys:
            self.logger.warning("Fehlende Gruppierungsspalten: %s", ", ".join(missing_group_keys))
            return {}

        for key in group_by_keys:
            df = df[df[key].notna()].copy()
            df[key] = df[key].astype(str)

        if df.empty:
            self.logger.warning("Nach Filterung keine Daten für Gruppierung verfügbar")
            return {}

        # Globale Perzentile über alle Modelle berechnen, damit die Skala
        # für den Modellvergleich konsistent bleibt.
        p1_global = float(df["tokens_per_second"].quantile(0.01))
        p99_global = float(df["tokens_per_second"].quantile(0.99))

        # Normierung je Zeile mit Schutz vor Division-by-zero (p99 == p1 -> 0.0).
        denominator = p99_global - p1_global
        df["token_score_norm"] = 0.0
        if denominator != 0:
            df["token_score_norm"] = (df["tokens_per_second"] - p1_global) / denominator

        # Ergebnis strikt auf [0, 1] begrenzen.
        df["token_score_norm"] = df["token_score_norm"].clip(lower=0.0, upper=1.0).fillna(0.0)

        # Für die bestehende Pipeline geben wir pro group_by-Key den Mittelwert zurück.
        # Bei group_by_keys=("profile", "model") ergibt das einen Score je Profil+Modell,
        # basierend auf globaler Normierung über alle Modelle und Systeme.
        grouped_scores = df.groupby(list(group_by_keys), dropna=False)["token_score_norm"].mean()

        normalized_scores: Dict[str | tuple[str, ...], float] = {}
        if len(group_by_keys) == 1:
            key_name = group_by_keys[0]
            normalized_scores = {
                str(group_value): float(score)
                for group_value, score in grouped_scores.items()
                if pd.notna(group_value)
            }
            self.logger.info("Token-Score-Normierung berechnet für %d '%s'-Gruppen", len(normalized_scores), key_name)
        else:
            normalized_scores = {
                tuple(str(part) for part in group_key): float(score)
                for group_key, score in grouped_scores.items()
            }
            self.logger.info("Token-Score-Normierung berechnet für %d Gruppen", len(normalized_scores))

        self.logger.info(
            "Globale Token-Score-Normierung | p1=%.4f tok/s | p99=%.4f tok/s",
            p1_global,
            p99_global,
        )

        return normalized_scores
