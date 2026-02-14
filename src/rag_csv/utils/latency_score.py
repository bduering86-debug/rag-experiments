#!/usr/bin/env python3
"""
Latency Score Calculator für Modell-Performance-Evaluation.

Berechnet normalisierte Latency-Scores basierend auf Latenz über alle Testfälle.
"""

import statistics
from typing import List, Dict, Any
from collections import defaultdict

from rag_csv.config.logging import get_logger


class LatencyScoreCalculator:
    """
    Berechnet einen normalisierten Latency-Score für Modelle über alle Testfälle.
    
    Der Score wird nach folgender Formel berechnet:
    LatencyScore_norm = 1 - ((Latency - p1(Latency)) / (p99(Latency) - p1(Latency)))
    
    Dabei werden p1 und p99 über ALLE Einzelmessungen aller Modelle berechnet.
    Die Modell-Mittelwerte werden dann mit diesen globalen Perzentilen normalisiert.
    Niedrigere Latenz führt zu höherem Score (daher 1 - ...).
    Score-Range: [0, 1], wobei höhere Werte = bessere Performance (niedrigere Latenz)
    """
    
    def __init__(self):
        """Initialisiert den LatencyScoreCalculator."""
        self.logger = get_logger(f"{__name__}.LatencyScoreCalculator")
    
    def calculate_scores(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Berechnet Latency-Scores für alle Modelle.
        
        Args:
            results: Liste der Evaluation-Ergebnisse mit total_latency Werten
            
        Returns:
            Dict[str, float]: Dictionary mit model -> normalized_latency_score
        """
        # Gruppiere nach Modell und sammle alle Einzelmessungen
        model_latencies = defaultdict(list)
        all_latency_values = []
        
        for result in results:
            model = result.get("model")
            latency = result.get("total_latency")
            
            if model and latency is not None:
                model_latencies[model].append(latency)
                all_latency_values.append(latency)
        
        # Berechne Mittelwerte pro Modell
        model_avg_latencies = {}
        for model, latency_list in model_latencies.items():
            model_avg_latencies[model] = statistics.mean(latency_list)
        
        if not model_avg_latencies:
            self.logger.warning("Keine Latency-Daten gefunden für Score-Berechnung")
            return {}
        
        # Berechne p1 und p99 über ALLE Einzelmessungen (nicht nur Modell-Mittelwerte)
        # Dies gibt eine robustere Normalisierung basierend auf der gesamten Verteilung
        p1 = statistics.quantiles(all_latency_values, n=100)[0]  # p1
        p99 = statistics.quantiles(all_latency_values, n=100)[98]  # p99
        
        self.logger.info(f"Latency - p1: {p1:.4f}s, p99: {p99:.4f}s (basierend auf {len(all_latency_values)} Messungen)")
        
        # Vermeide Division durch Null
        if p99 - p1 == 0:
            self.logger.warning("p99 und p1 sind identisch, kann keine normalisierten Scores berechnen")
            return {model: 1.0 for model in model_avg_latencies.keys()}
        
        # Berechne normalisierte Scores
        normalized_scores = {}
        for model, avg_latency in model_avg_latencies.items():
            normalized_score = 1 - ((avg_latency - p1) / (p99 - p1))
            # Clamp auf [0, 1]
            normalized_score = max(0.0, min(1.0, normalized_score))
            normalized_scores[model] = normalized_score
            
            self.logger.info(
                f"Modell: {model} | Avg Latency: {avg_latency:.4f}s | Normalized Score: {normalized_score:.4f}"
            )
        
        return normalized_scores
