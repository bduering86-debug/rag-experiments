"""
Latency Tracking für RAG-Evaluation.

Misst die Latenz verschiedener Komponenten im RAG-Prozess:
- Retrieval (Vektorsuche)
- LLM-Inferenz
- Gesamtlatenz (End-to-End)
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class LatencyTracker:
    """
    Tracker für Latenz-Messungen im RAG-Prozess.
    
    Verwendung:
        tracker = LatencyTracker()
        tracker.start_total()
        
        tracker.start_retrieval()
        # ... retrieval code ...
        tracker.end_retrieval()
        
        tracker.start_llm()
        # ... llm code ...
        tracker.end_llm()
        
        tracker.end_total()
        metrics = tracker.get_metrics()
    """
    
    # Timestamps
    _total_start: Optional[float] = field(default=None, init=False)
    _total_end: Optional[float] = field(default=None, init=False)
    _retrieval_start: Optional[float] = field(default=None, init=False)
    _retrieval_end: Optional[float] = field(default=None, init=False)
    _llm_start: Optional[float] = field(default=None, init=False)
    _llm_end: Optional[float] = field(default=None, init=False)
    
    def start_total(self) -> None:
        """Startet die Gesamtzeitmessung."""
        self._total_start = time.time()
    
    def end_total(self) -> None:
        """Beendet die Gesamtzeitmessung."""
        self._total_end = time.time()
    
    def start_retrieval(self) -> None:
        """Startet die Retrieval-Zeitmessung."""
        self._retrieval_start = time.time()
    
    def end_retrieval(self) -> None:
        """Beendet die Retrieval-Zeitmessung."""
        self._retrieval_end = time.time()
    
    def start_llm(self) -> None:
        """Startet die LLM-Zeitmessung."""
        self._llm_start = time.time()
    
    def end_llm(self) -> None:
        """Beendet die LLM-Zeitmessung."""
        self._llm_end = time.time()
    
    @property
    def total_latency(self) -> Optional[float]:
        """Gesamtlatenz in Sekunden."""
        if self._total_start is None or self._total_end is None:
            return None
        return self._total_end - self._total_start
    
    @property
    def retrieval_duration(self) -> Optional[float]:
        """Retrieval-Dauer in Sekunden."""
        if self._retrieval_start is None or self._retrieval_end is None:
            return None
        return self._retrieval_end - self._retrieval_start
    
    @property
    def llm_duration(self) -> Optional[float]:
        """LLM-Inferenz-Dauer in Sekunden."""
        if self._llm_start is None or self._llm_end is None:
            return None
        return self._llm_end - self._llm_start
    
    def get_metrics(self) -> Dict[str, Optional[float]]:
        """
        Gibt alle Latenz-Metriken als Dictionary zurück.
        
        Returns:
            Dict mit Latenz-Werten in Sekunden (None wenn nicht gemessen)
        """
        return {
            "total_latency": self.total_latency,
            "retrieval_duration": self.retrieval_duration,
            "llm_duration": self.llm_duration
        }
    
    def reset(self) -> None:
        """Setzt alle Messungen zurück."""
        self._total_start = None
        self._total_end = None
        self._retrieval_start = None
        self._retrieval_end = None
        self._llm_start = None
        self._llm_end = None
