"""
Answer Quality Score Berechnung für LLM-as-a-Judge.

Berechnet einen normalisierten Quality Score basierend auf:
- F: Faithfulness (Treue zum Kontext)
- R: Answer Relevance (Relevanz der Antwort)
- C: Completeness (Vollständigkeit)
- L: Fluency (Sprachliche Qualität)
"""

from typing import Dict, Optional
from dataclasses import dataclass

from rag_csv.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityScores:
    """Container für die 4 Quality-Dimensionen."""
    faithfulness: float  # F: 1-5
    relevance: float     # R: 1-5
    completeness: float  # C: 1-5
    fluency: float       # L: 1-5


class AnswerQualityCalculator:
    """
    Berechnet Answer Quality Score nach der Formel:
    
    AnswerQualityScore = 0.35 · F + 0.25 · R + 0.25 · C + 0.15 · L
    
    Normalisiert auf [0, 1]:
    AnswerQualityScore[0,1] = (AnswerQualityScore - 1) / 4
    
    Bewertungsskala:
    1 = Sehr schlecht
    2 = Schwach
    3 = Akzeptabel
    4 = Gut
    5 = Sehr gut
    """
    
    # Gewichtungen für die 4 Dimensionen
    WEIGHT_FAITHFULNESS = 0.35
    WEIGHT_RELEVANCE = 0.25
    WEIGHT_COMPLETENESS = 0.25
    WEIGHT_FLUENCY = 0.15
    
    def __init__(self):
        """Initialisiert Calculator."""
        logger.debug("AnswerQualityCalculator initialisiert")
    
    @staticmethod
    def validate_score(score: float, name: str) -> float:
        """
        Validiert einen Score (muss zwischen 1 und 5 sein).
        
        Args:
            score: Score-Wert
            name: Name des Scores (für Logging)
            
        Returns:
            Validierter Score
            
        Raises:
            ValueError: Wenn Score außerhalb [1, 5]
        """
        if not (1 <= score <= 5):
            raise ValueError(f"{name} muss zwischen 1 und 5 liegen, ist aber {score}")
        return float(score)
    
    def calculate_raw_score(self, scores: QualityScores) -> float:
        """
        Berechnet den gewichteten Raw Score.
        
        Args:
            scores: QualityScores mit F, R, C, L
            
        Returns:
            Raw Score (zwischen 1 und 5)
        """
        # Validiere alle Scores
        f = self.validate_score(scores.faithfulness, "Faithfulness")
        r = self.validate_score(scores.relevance, "Relevance")
        c = self.validate_score(scores.completeness, "Completeness")
        l = self.validate_score(scores.fluency, "Fluency")
        
        # Berechne gewichteten Score
        raw_score = (
            self.WEIGHT_FAITHFULNESS * f +
            self.WEIGHT_RELEVANCE * r +
            self.WEIGHT_COMPLETENESS * c +
            self.WEIGHT_FLUENCY * l
        )
        
        logger.debug("Raw Score berechnet: %.4f (F=%.1f, R=%.1f, C=%.1f, L=%.1f)",
                    raw_score, f, r, c, l)
        
        return raw_score
    
    def normalize_score(self, raw_score: float) -> float:
        """
        Normalisiert Score auf [0, 1].
        
        Args:
            raw_score: Raw Score (1-5)
            
        Returns:
            Normalisierter Score (0-1)
        """
        normalized = (raw_score - 1.0) / 4.0
        
        # Sicherheitscheck (sollte immer erfüllt sein)
        normalized = max(0.0, min(1.0, normalized))
        
        logger.debug("Normalisierter Score: %.4f", normalized)
        
        return normalized
    
    def calculate(self, scores: QualityScores) -> Dict[str, float]:
        """
        Berechnet alle Scores (raw und normalized).
        
        Args:
            scores: QualityScores mit F, R, C, L
            
        Returns:
            Dict mit allen Score-Werten
        """
        raw_score = self.calculate_raw_score(scores)
        normalized_score = self.normalize_score(raw_score)
        
        return {
            "faithfulness": scores.faithfulness,
            "relevance": scores.relevance,
            "completeness": scores.completeness,
            "fluency": scores.fluency,
            "raw_score": raw_score,
            "normalized_score": normalized_score
        }
    
    @staticmethod
    def get_interpretation(score: float) -> str:
        """
        Gibt textuelle Interpretation eines Scores zurück.
        
        Args:
            score: Raw Score (1-5)
            
        Returns:
            Textuelle Interpretation
        """
        if score >= 4.5:
            return "Sehr gut"
        elif score >= 3.5:
            return "Gut"
        elif score >= 2.5:
            return "Akzeptabel"
        elif score >= 1.5:
            return "Schwach"
        else:
            return "Sehr schlecht"
