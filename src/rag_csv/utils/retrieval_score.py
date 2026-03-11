#!/usr/bin/env python3
"""
Retrieval Score Calculator - Berechnet kombinierten Retrieval-Score aus Recall und nDCG.

Der Retrieval Score kombiniert Recall@K und nDCG@K zu einer einzelnen Metrik:
    RetrievalScore = (Recall@K + nDCG@K) / 2

Wertebereich: [0.0, 1.0]
- 1.0: Perfekte Retrieval-Performance (Recall=1.0, nDCG=1.0)
- 0.0: Keine relevanten Dokumente gefunden (Recall=0.0, nDCG=0.0)
"""

from typing import Dict, List, Optional


class RetrievalScoreCalculator:
    """
    Berechnet kombinierten Retrieval-Score aus Recall@K und nDCG@K.
    
    Der Score gibt einen Gesamteindruck der Retrieval-Performance:
    - Recall misst, ob relevante Dokumente gefunden wurden
    - nDCG misst, ob relevante Dokumente an den richtigen Positionen sind
    
    Beispiel:
        >>> calculator = RetrievalScoreCalculator()
        >>> score = calculator.calculate(recall=0.8, ndcg=0.9)
        >>> print(score)  # 0.85
    """
    
    def __init__(self):
        """Initialisiert den Retrieval Score Calculator."""
        pass
    
    def calculate(self, recall: Optional[float], ndcg: Optional[float]) -> Optional[float]:
        """
        Berechnet den Retrieval-Score aus Recall und nDCG.
        
        Formula: RetrievalScore = (Recall + nDCG) / 2
        
        Args:
            recall: Recall@K Wert (0.0-1.0) oder None
            ndcg: nDCG@K Wert (0.0-1.0) oder None
            
        Returns:
            float: Retrieval-Score (0.0-1.0) oder None wenn Input None ist
            
        Raises:
            ValueError: Wenn Werte außerhalb [0.0, 1.0] liegen
        """
        # Handle None-Werte
        if recall is None or ndcg is None:
            return None
        
        # Validiere Eingaben
        if not (0.0 <= recall <= 1.0):
            raise ValueError(f"Recall muss zwischen 0.0 und 1.0 liegen, ist aber {recall}")
        if not (0.0 <= ndcg <= 1.0):
            raise ValueError(f"nDCG muss zwischen 0.0 und 1.0 liegen, ist aber {ndcg}")
        
        # Berechne Score
        retrieval_score = (recall + ndcg) / 2.0
        
        return retrieval_score
    
    def calculate_batch(self, results: List[Dict]) -> Dict[str, float]:
        """
        Berechnet Retrieval-Scores für eine Liste von Ergebnissen.
        
        Args:
            results: Liste von Dicts mit 'recall@k' und 'ndcg@k' keys
            
        Returns:
            Dict: Mapping von model -> durchschnittlicher Retrieval-Score
        """
        model_scores = {}
        
        for result in results:
            model = result.get("model")
            if not model:
                continue
            
            recall = result.get("recall@k")
            ndcg = result.get("ndcg@k")
            
            score = self.calculate(recall, ndcg)
            if score is not None:
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(score)
        
        # Durchschnitt berechnen
        avg_scores = {}
        for model, scores in model_scores.items():
            if scores:
                avg_scores[model] = sum(scores) / len(scores)
        
        return avg_scores
    
    @staticmethod
    def get_interpretation(recall: Optional[float], ndcg: Optional[float]) -> str:
        """
        Gibt textuelle Interpretation der Retrieval-Performance zurück.
        
        Analysiert die Kombination von Recall und nDCG:
        - Hoher Recall + niedriger nDCG = Zu viel Rauschen (zu viele irrelevante Quellen)
        - Niedriger Recall + hoher nDCG = Relevantes Wissen fehlt (zu wenige Quellen)
        - Beide hoch = Sehr gute Performance
        - Beide niedrig = Schlechte Performance
        
        Args:
            recall: Recall@K Wert (0.0-1.0) oder None
            ndcg: nDCG@K Wert (0.0-1.0) oder None
            
        Returns:
            str: Textuelle Interpretation der Performance
        """
        if recall is None or ndcg is None:
            return "Keine Daten verfügbar"
        
        # Schwellenwerte für Klassifikation
        HIGH_THRESHOLD = 0.7
        MID_THRESHOLD = 0.4
        LOW_THRESHOLD = 0.2
        
        # Klassifiziere Recall
        if recall >= HIGH_THRESHOLD:
            recall_level = "high"
        elif recall >= MID_THRESHOLD:
            recall_level = "mid"
        else:
            recall_level = "low"
        
        # Klassifiziere nDCG
        if ndcg >= HIGH_THRESHOLD:
            ndcg_level = "high"
        elif ndcg >= MID_THRESHOLD:
            ndcg_level = "mid"
        else:
            ndcg_level = "low"
        
        # Interpretationen basierend auf Kombinationen
        if recall_level == "high" and ndcg_level == "high":
            return "Sehr gut - Relevante Dokumente gefunden und gut gerankt"
        
        elif recall_level == "high" and ndcg_level == "mid":
            return "Gut - Relevante Dokumente gefunden, Ranking könnte besser sein"
        
        elif recall_level == "high" and ndcg_level == "low":
            return "Zu viel Rauschen - Viele Dokumente gefunden, aber schlechtes Ranking (zu viele irrelevante Quellen)"
        
        elif recall_level == "mid" and ndcg_level == "high":
            return "Gut - Gefundene Dokumente gut gerankt, aber nicht alle relevanten gefunden"
        
        elif recall_level == "mid" and ndcg_level == "mid":
            return "Akzeptabel - Durchschnittliche Retrieval-Performance"
        
        elif recall_level == "mid" and ndcg_level == "low":
            return "Mäßig - Einige Dokumente gefunden, aber schlechtes Ranking"
        
        elif recall_level == "low" and ndcg_level == "high":
            return "Relevantes Wissen fehlt - Gutes Ranking, aber zu wenige relevante Dokumente gefunden"
        
        elif recall_level == "low" and ndcg_level == "mid":
            return "Schwach - Wenige Dokumente gefunden, Ranking durchschnittlich"
        
        else:  # both low
            return "Sehr schlecht - Kaum oder keine relevante Dokumente gefunden und schlecht gerankt"
