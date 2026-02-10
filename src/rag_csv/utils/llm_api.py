"""
Generische LLM API Client Klasse.

Unterstützt verschiedene LLM APIs (OpenAI, Anthropic, etc.)
mit einem einheitlichen Interface.
"""

import os
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from rag_csv.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """Konfiguration für LLM API."""
    api_url: str
    api_key: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 1000


class LLMAPIClient:
    """
    Generischer Client für LLM APIs.
    
    Unterstützt OpenAI-kompatible APIs (OpenAI, Azure OpenAI, etc.)
    und kann leicht für andere APIs erweitert werden.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialisiert LLM API Client.
        
        Args:
            config: LLM Konfiguration
        """
        self.config = config
        
        if not self.config.api_key:
            raise ValueError("LLM_JUDGE_API_KEY muss in .env gesetzt sein")
        
        logger.info("LLM API Client initialisiert - Model: %s", self.config.model)
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Sendet Chat Completion Request an LLM API.
        
        Args:
            messages: Liste von Message-Dicts mit "role" und "content"
            temperature: Optional - überschreibt Config-Wert
            max_tokens: Optional - überschreibt Config-Wert
            
        Returns:
            Dict mit response, usage, etc.
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens
        }
        
        try:
            response = requests.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            # Extrahiere Antwort
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            
            return {
                "success": True,
                "content": content,
                "usage": usage,
                "model": data.get("model", self.config.model),
                "error": None
            }
            
        except requests.exceptions.RequestException as e:
            logger.error("LLM API Request fehlgeschlagen: %s", e)
            return {
                "success": False,
                "content": "",
                "usage": {},
                "model": self.config.model,
                "error": str(e)
            }
    
    def simple_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Vereinfachte Methode für einfache Prompts.
        
        Args:
            prompt: User-Prompt
            system_prompt: Optional System-Prompt
            temperature: Optional Temperature
            max_tokens: Optional Max Tokens
            
        Returns:
            Dict mit response, usage, etc.
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.chat_completion(messages, temperature, max_tokens)
