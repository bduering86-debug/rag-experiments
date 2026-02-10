"""
LLM-as-a-Judge für Antwortqualitätsbewertung.

Orchestriert:
1. Prompt-Erstellung für Judge-LLM
2. API-Aufruf an externes LLM
3. Parsing der LLM-Antwort
4. Berechnung des Quality Scores
"""

import re
import os
from typing import Dict, Optional, Any

from rag_csv.config.logging import get_logger
from rag_csv.utils.llm_api import LLMAPIClient, LLMConfig
from rag_csv.utils.answer_quality import AnswerQualityCalculator, QualityScores

logger = get_logger(__name__)


# System Prompt für Judge LLM
JUDGE_SYSTEM_PROMPT = """Du bist ein Experte für die Bewertung von IT-Support-Antworten. 
Deine Aufgabe ist es, Antworten auf IT-Support-Tickets anhand von 4 Kriterien zu bewerten.

Bewerte auf einer Skala von 1-5:
1 = Sehr schlecht: Kriterium nicht erfüllt, gravierende Mängel
2 = Schwach: Teilweise erfüllt, deutliche Fehler
3 = Akzeptabel: Grundsätzlich erfüllt, erkennbare Schwächen
4 = Gut: Weitgehend erfüllt, nur geringfügige Mängel
5 = Sehr gut: Vollständig erfüllt, keine relevanten Mängel

Antworte IMMER im folgenden Format (exakt diese Zeilen):
Faithfulness: <Zahl>
Relevance: <Zahl>
Completeness: <Zahl>
Fluency: <Zahl>
Begründung: <kurze Erklärung>
"""


# User Prompt Template
USER_PROMPT_TEMPLATE = """Bewerte die folgende IT-Support-Antwort:

**TICKET-BESCHREIBUNG:**
{ticket_description}

**BEREITGESTELLTER KONTEXT (aus Knowledge Base):**
{context}

**GENERIERTE ANTWORT:**
{generated_answer}

**KRITERIEN:**
1. **Faithfulness (F)**: Ist die Antwort treu zum bereitgestellten Kontext? Werden keine falschen oder erfundenen Informationen hinzugefügt?
2. **Relevance (R)**: Ist die Antwort relevant für das beschriebene Problem?
3. **Completeness (C)**: Ist die Antwort vollständig und beantwortet alle wichtigen Aspekte des Problems?
4. **Fluency (L)**: Ist die Antwort sprachlich korrekt und gut verständlich formuliert?

Bewerte jedes Kriterium von 1-5 und gib eine kurze Begründung.
"""


class LLMJudge:
    """
    LLM-as-a-Judge für Antwortqualitätsbewertung.
    
    Verwendet externes LLM zur Bewertung von RAG-Antworten
    und berechnet normalisierten Quality Score.
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialisiert LLM Judge.
        
        Args:
            api_url: Optional - überschreibt .env Wert
            api_key: Optional - überschreibt .env Wert
            model: Optional - überschreibt .env Wert
            temperature: Optional - überschreibt .env Wert
            max_tokens: Optional - überschreibt .env Wert
        """
        # Konfiguration aus .env laden oder Override verwenden
        config = LLMConfig(
            api_url=api_url or os.getenv("LLM_JUDGE_API_URL", ""),
            api_key=api_key or os.getenv("LLM_JUDGE_API_KEY", ""),
            model=model or os.getenv("LLM_JUDGE_MODEL", "gpt-4o-mini"),
            temperature=temperature if temperature is not None else float(os.getenv("LLM_JUDGE_TEMPERATURE", "0.1")),
            max_tokens=max_tokens if max_tokens is not None else int(os.getenv("LLM_JUDGE_MAX_TOKENS", "1000"))
        )
        
        self.llm_client = LLMAPIClient(config)
        self.quality_calculator = AnswerQualityCalculator()
        
        logger.info("LLM Judge initialisiert - Model: %s", config.model)
    
    def _build_prompt(
        self,
        ticket_description: str,
        context: str,
        generated_answer: str
    ) -> str:
        """
        Erstellt Prompt für Judge LLM.
        
        Args:
            ticket_description: Ticket-Beschreibung
            context: Bereitgestellter Kontext (aus KB)
            generated_answer: Generierte Antwort
            
        Returns:
            Formatierter Prompt
        """
        return USER_PROMPT_TEMPLATE.format(
            ticket_description=ticket_description,
            context=context,
            generated_answer=generated_answer
        )
    
    def _parse_judge_response(self, response: str) -> Optional[QualityScores]:
        """
        Parst LLM Judge Response und extrahiert Scores.
        
        Args:
            response: LLM Response Text
            
        Returns:
            QualityScores oder None bei Parse-Fehler
        """
        try:
            # Extrahiere Scores mit Regex
            f_match = re.search(r'Faithfulness:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            r_match = re.search(r'Relevance:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            c_match = re.search(r'Completeness:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            l_match = re.search(r'Fluency:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            
            if not all([f_match, r_match, c_match, l_match]):
                logger.warning("Konnte nicht alle Scores aus Judge Response extrahieren")
                logger.debug("Response: %s", response)
                return None
            
            scores = QualityScores(
                faithfulness=float(f_match.group(1)),
                relevance=float(r_match.group(1)),
                completeness=float(c_match.group(1)),
                fluency=float(l_match.group(1))
            )
            
            logger.debug("Scores geparst: F=%.1f, R=%.1f, C=%.1f, L=%.1f",
                        scores.faithfulness, scores.relevance,
                        scores.completeness, scores.fluency)
            
            return scores
            
        except Exception as e:
            logger.error("Fehler beim Parsen der Judge Response: %s", e)
            return None
    
    def evaluate(
        self,
        ticket_description: str,
        context: str,
        generated_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluiert eine Antwort mit LLM Judge.
        
        Args:
            ticket_description: Ticket-Beschreibung
            context: Bereitgestellter Kontext
            generated_answer: Generierte Antwort
            
        Returns:
            Dict mit Quality Scores und Metadaten
        """
        logger.info("Starte LLM Judge Evaluation")
        
        # 1. Prompt erstellen
        prompt = self._build_prompt(ticket_description, context, generated_answer)
        
        # 2. LLM anfragen
        result = self.llm_client.simple_prompt(
            prompt=prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT
        )
        
        if not result["success"]:
            logger.error("LLM Judge API Fehler: %s", result["error"])
            return {
                "success": False,
                "error": result["error"],
                "faithfulness": None,
                "relevance": None,
                "completeness": None,
                "fluency": None,
                "raw_score": None,
                "normalized_score": None,
                "judge_response": None
            }
        
        # 3. Response parsen
        scores = self._parse_judge_response(result["content"])
        
        if scores is None:
            logger.error("Konnte Judge Response nicht parsen")
            return {
                "success": False,
                "error": "Parse-Fehler",
                "faithfulness": None,
                "relevance": None,
                "completeness": None,
                "fluency": None,
                "raw_score": None,
                "normalized_score": None,
                "judge_response": result["content"]
            }
        
        # 4. Quality Score berechnen
        try:
            quality_metrics = self.quality_calculator.calculate(scores)
            
            logger.info("✓ Judge Evaluation erfolgreich: Score=%.4f",
                       quality_metrics["normalized_score"])
            
            return {
                "success": True,
                "error": None,
                "faithfulness": quality_metrics["faithfulness"],
                "relevance": quality_metrics["relevance"],
                "completeness": quality_metrics["completeness"],
                "fluency": quality_metrics["fluency"],
                "raw_score": quality_metrics["raw_score"],
                "normalized_score": quality_metrics["normalized_score"],
                "judge_response": result["content"],
                "judge_usage": result["usage"]
            }
            
        except Exception as e:
            logger.error("Fehler bei Score-Berechnung: %s", e)
            return {
                "success": False,
                "error": str(e),
                "faithfulness": scores.faithfulness,
                "relevance": scores.relevance,
                "completeness": scores.completeness,
                "fluency": scores.fluency,
                "raw_score": None,
                "normalized_score": None,
                "judge_response": result["content"]
            }
