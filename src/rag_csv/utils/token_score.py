#!/usr/bin/env python3
"""
Token Score Calculator für Modell-Performance-Evaluation.

Berechnet normalisierte Token-Scores basierend auf Token/s über alle Testfälle.
"""

import statistics
from typing import List, Dict, Any
from collections import defaultdict

from rag_csv.config.logging import get_logger


class TokenScoreCalculator:
    """
    Berechnet einen normalisierten Token-Score für Modelle über alle Testfälle.
    
    Der Score wird nach folgender Formel berechnet:
    TokenScore_norm = (Token/s - p1(Token/s)) / (p99(Token/s) - p1(Token/s))
    
    Dabei werden p1 und p99 über ALLE Einzelmessungen aller Modelle berechnet.
    Die Modell-Mittelwerte werden dann mit diesen globalen Perzentilen normalisiert.
    Score-Range: [0, 1], wobei höhere Werte = bessere Performance (höhere Token/s)
    """
    
    def __init__(self):
        """Initialisiert den TokenScoreCalculator."""
        self.logger = get_logger(f"{__name__}.TokenScoreCalculator")
    
    def calculate_scores(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Berechnet Token-Scores für alle Modelle.
        
        Args:
            results: Liste der Evaluation-Ergebnisse mit tokens_per_second Werten
            
        Returns:
            Dict[str, float]: Dictionary mit model -> normalized_token_score
        """
        # Gruppiere nach Modell und sammle alle Einzelmessungen
        model_tokens = defaultdict(list)
        all_tokens_values = []
        
        for result in results:
            model = result.get("model")
            tokens_per_s = result.get("tokens_per_second")
            
            if model and tokens_per_s is not None:
                model_tokens[model].append(tokens_per_s)
                all_tokens_values.append(tokens_per_s)
        
        # Berechne Mittelwerte pro Modell
        model_avg_tokens = {}
        for model, tokens_list in model_tokens.items():
            model_avg_tokens[model] = statistics.mean(tokens_list)
        
        if not model_avg_tokens:
            self.logger.warning("Keine Token/s Daten gefunden für Score-Berechnung")
            return {}
        
        # Berechne p1 und p99 über ALLE Einzelmessungen (nicht nur Modell-Mittelwerte)
        # Dies gibt eine robustere Normalisierung basierend auf der gesamten Verteilung
        p1 = statistics.quantiles(all_tokens_values, n=100)[0]  # p1
        p99 = statistics.quantiles(all_tokens_values, n=100)[98]  # p99
        
        self.logger.info(f"Token/s - p1: {p1:.2f}, p99: {p99:.2f} (basierend auf {len(all_tokens_values)} Messungen)")
        
        # Vermeide Division durch Null
        if p99 - p1 == 0:
            self.logger.warning("p99 und p1 sind identisch, kann keine normalisierten Scores berechnen")
            return {model: 1.0 for model in model_avg_tokens.keys()}
        
        # Berechne normalisierte Scores
        normalized_scores = {}
        for model, avg_tokens in model_avg_tokens.items():
            normalized_score = (avg_tokens - p1) / (p99 - p1)
            # Clamp auf [0, 1]
            normalized_score = max(0.0, min(1.0, normalized_score))
            normalized_scores[model] = normalized_score
            
            self.logger.info(
                f"Modell: {model} | Avg Token/s: {avg_tokens:.2f} | Normalized Score: {normalized_score:.4f}"
            )
        
        return normalized_scores
