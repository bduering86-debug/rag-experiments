"""
Token Tracking für RAG-Evaluation mit Ollama.

Verarbeitet Token-Statistiken aus Ollama-Responses:
- Prompt-Tokens (Input)
- Generated-Tokens (Output)
- Tokens per Second (Throughput)
- Total Tokens
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class TokenTracker:
    """
    Tracker für Token-Metriken aus Ollama-Responses.
    
    Ollama liefert folgende Felder in der Response:
    - prompt_eval_count: Anzahl der Prompt-Tokens
    - eval_count: Anzahl der generierten Tokens
    - total_duration: Gesamtzeit in Nanosekunden
    - prompt_eval_duration: Zeit für Prompt-Verarbeitung in Nanosekunden
    - eval_duration: Zeit für Token-Generierung in Nanosekunden
    """
    
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    total_duration_ns: Optional[int] = None
    prompt_eval_duration_ns: Optional[int] = None
    eval_duration_ns: Optional[int] = None
    
    @classmethod
    def from_ollama_response(cls, response: Dict) -> 'TokenTracker':
        """
        Erstellt TokenTracker aus Ollama-Response.
        
        Args:
            response: Ollama API Response Dictionary
            
        Returns:
            TokenTracker mit extrahierten Metriken
        """
        prompt_tokens = response.get("prompt_eval_count")
        generated_tokens = response.get("eval_count")
        
        # Total tokens berechnen
        total_tokens = None
        if prompt_tokens is not None and generated_tokens is not None:
            total_tokens = prompt_tokens + generated_tokens
        
        return cls(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            total_tokens=total_tokens,
            total_duration_ns=response.get("total_duration"),
            prompt_eval_duration_ns=response.get("prompt_eval_duration"),
            eval_duration_ns=response.get("eval_duration")
        )
    
    @property
    def tokens_per_second(self) -> Optional[float]:
        """
        Berechnet Tokens pro Sekunde (nur generierte Tokens).
        
        Returns:
            Tokens/Sekunde oder None wenn nicht berechenbar
        """
        if self.generated_tokens is None or self.eval_duration_ns is None:
            return None
        
        if self.eval_duration_ns == 0:
            return None
        
        # Nanosekunden zu Sekunden konvertieren
        duration_seconds = self.eval_duration_ns / 1e9
        return self.generated_tokens / duration_seconds
    
    @property
    def total_duration_seconds(self) -> Optional[float]:
        """Gesamtdauer in Sekunden."""
        if self.total_duration_ns is None:
            return None
        return self.total_duration_ns / 1e9
    
    @property
    def prompt_eval_duration_seconds(self) -> Optional[float]:
        """Prompt-Eval-Dauer in Sekunden."""
        if self.prompt_eval_duration_ns is None:
            return None
        return self.prompt_eval_duration_ns / 1e9
    
    @property
    def eval_duration_seconds(self) -> Optional[float]:
        """Eval-Dauer in Sekunden."""
        if self.eval_duration_ns is None:
            return None
        return self.eval_duration_ns / 1e9
    
    def get_metrics(self) -> Dict[str, Optional[float]]:
        """
        Gibt alle Token-Metriken als Dictionary zurück.
        
        Returns:
            Dict mit Token-Statistiken
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_second": self.tokens_per_second,
            "total_duration_seconds": self.total_duration_seconds,
            "prompt_eval_duration_seconds": self.prompt_eval_duration_seconds,
            "eval_duration_seconds": self.eval_duration_seconds
        }
