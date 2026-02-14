#!/usr/bin/env python3
"""
RAG Answer Orchestrator für Testfall-Evaluation.

Orchestriert den kompletten RAG-Answer-Flow:
1. Lädt Testfälle (load_testcases)
2. Führt Retrieval durch
3. Sendet Ergebnisse an Ollama (low/mid/high Konfigurationen)
4. Berechnet Metriken (nDCG@K, Recall@K)
5. Loggt Ergebnisse
"""

import os
import csv
import time
import requests
import uuid
from typing import List, Dict, Any, Literal, Optional
from pathlib import Path
from datetime import datetime

from rag_csv.config.logging import get_logger
from rag_csv.config.settings import DataConfig, OllamaConfig, EvaluationConfig
from rag_csv.data.load_testcases import load_testcases, TESTCASES_FILE
from rag_csv.core.retrieval import search, SearchHit
from rag_csv.utils.nDCGTopK import nDCGTopK
from rag_csv.utils.RecallTopK import RecallTopK
from rag_csv.utils.latency import LatencyTracker
from rag_csv.utils.tokens import TokenTracker
from rag_csv.utils.llm_judge import LLMJudge
from rag_csv.utils.token_score import TokenScoreCalculator
from rag_csv.utils.latency_score import LatencyScoreCalculator
from rag_csv.utils.system_metrics_logger import SystemMetricsLogger
from rag_csv.utils.retrieval_score import RetrievalScoreCalculator

logger = get_logger(__name__)

# Evaluation Config laden
_eval_config = EvaluationConfig()


# Baseline-Modelle für alle Konfigurationen
BASELINE_MODELS = [
    "llama3.1:8b-instruct-q4_K_M",
    "granite3.1-dense:8b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q4_K_M"
]

# Zusätzliche Modelle für Mid-Profile
MID_PROFILE_MODELS = [
    "llama3.1:8b-instruct-q6_K",
    "granite3.1-dense:8b-instruct-q6_K",
    "qwen2.5:7b-instruct-q4_K_M",
]

# Zusätzliche Modelle für High-Profile
HIGH_PROFILE_MODELS = [
    "llama3.1:8b-instruct-q8_0",
    "granite3.1-dense:8b-instruct-q8_0",
    "qwen2.5:14b-instruct-q4_K_M",
]


class RAGAnswerOrchestrator:
    """
    Orchestrator für RAG-basierte Antwortgenerierung und Evaluation.
    """
    
    @staticmethod
    def _get_next_run_file(output_dir: Path, date_str: str) -> Path:
        """
        Findet den nächsten verfügbaren Run-Dateinamen.
        
        Args:
            output_dir: Ausgabeverzeichnis
            date_str: Datums-String im Format YYMMDD
            
        Returns:
            Path: Pfad zur nächsten Run-Datei (runs_YYMMDD_NNN.csv)
        """
        run_number = 1
        while True:
            filename = f"runs_{date_str}_{run_number:03d}.csv"
            filepath = output_dir / filename
            if not filepath.exists():
                return filepath
            run_number += 1
    
    def __init__(
        self,
        top_k: int = None,
        runs_per_testcase: int = None,
        output_dir: str = "output/experiment"
    ):
        """
        Initialisiert den RAG Answer Orchestrator.
        
        Args:
            top_k: Anzahl der zu retrievenden Dokumente (default: aus .env TOP_K)
            runs_per_testcase: Anzahl der Durchläufe pro Testfall (default: aus .env RUNS_PER_TESTCASE)
            output_dir: Verzeichnis für Evaluation-Ergebnisse
        """
        # Werte aus .env laden, falls nicht explizit angegeben
        self.top_k = top_k if top_k is not None else _eval_config.top_k
        self.runs_per_testcase = runs_per_testcase if runs_per_testcase is not None else _eval_config.runs_per_testcase
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment-ID generieren (Timestamp)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.ollama_config = OllamaConfig()
        
        # Metriken initialisieren
        self.ndcg_metric = nDCGTopK(k=self.top_k)
        self.recall_metric = RecallTopK(k=self.top_k)
        self.retrieval_score_calculator = RetrievalScoreCalculator()
        
        # LLM Judge initialisieren (optional)
        self.use_llm_judge = _eval_config.use_llm_judge
        self.llm_judge = None
        if self.use_llm_judge:
            try:
                self.llm_judge = LLMJudge(
                    api_url=_eval_config.llm_judge_api_url,
                    api_key=_eval_config.llm_judge_api_key,
                    model=_eval_config.llm_judge_model,
                    temperature=_eval_config.llm_judge_temperature,
                    max_tokens=_eval_config.llm_judge_max_tokens
                )
                logger.info("  - LLM Judge: aktiviert")
            except Exception as e:
                logger.warning("LLM Judge konnte nicht initialisiert werden: %s", e)
                self.use_llm_judge = False
        
        # Log directory für Testcase-Details
        self.log_dir = Path("output/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("RAG Answer Orchestrator initialisiert")
        logger.info("  - Top-K: %d", self.top_k)
        logger.info("  - Runs per Testcase: %d", self.runs_per_testcase)
        logger.info("  - Output Dir: %s", self.output_dir)
    
    def get_models_for_profile(self, profile: Literal["low", "mid", "high"]) -> List[str]:
        """
        Gibt die Modelle für ein bestimmtes Profil zurück.
        
        Args:
            profile: "low", "mid" oder "high"
            
        Returns:
            List[str]: Liste der Modellnamen
        """
        models = BASELINE_MODELS.copy()
        
        if profile == "mid":
            models.extend(MID_PROFILE_MODELS)
        elif profile == "high":
            models.extend(HIGH_PROFILE_MODELS)
        
        return models
    
    # manuell angepasste Parameterliste inkl. GPU-Inferenz
    def get_ollama_url_for_profile(self, profile: Literal["low", "mid", "high", "gpu"]) -> str:
        """
        Gibt die Ollama-URL für ein Profil zurück.
        
        Args:
            profile: "low", "mid" oder "high"
            
        Returns:
            str: Ollama URL
        """
        if profile == "low":
            return self.ollama_config.url_low_profile or self.ollama_config.url
        elif profile == "mid":
            return self.ollama_config.url_mid_profile or self.ollama_config.url
        elif profile == "high":
            return self.ollama_config.url_high_profile or self.ollama_config.url
        # maneuell hinzugefügt für gpu inferenz, das auf ultra_profile zurückgreift
        elif profile == "gpu":
            return self.ollama_config.url_ultra_profile or self.ollama_config.url
        
        return self.ollama_config.url
    # manuell angepasste Parameter für GPU-Inferenz
    def get_threads_for_profile(self, profile: Literal["low", "mid", "high", "gpu"]) -> int:
        """
        Gibt die Thread-Anzahl für ein Profil zurück.
        
        Args:
            profile: "low", "mid", "high" oder "gpu"
            
        Returns:
            int: Anzahl der Threads
        """
        if profile == "low":
            return self.ollama_config.threads_low
        elif profile == "mid":
            return self.ollama_config.threads_mid
        elif profile == "high":
            return self.ollama_config.threads_high
        elif profile == "gpu":
            return self.ollama_config.threads_high  # GPU-Inferenz nutzt gleiche Threads wie High-Profile
        
        return self.ollama_config.threads
    
    def retrieve_for_testcase(self, testcase: Dict[str, Any]) -> List[SearchHit]:
        """
        Führt Retrieval für einen Testfall durch.
        Nur KB-Artikel werden abgefragt, keine Incidents.
        
        Args:
            testcase: Dictionary mit Testfall-Daten
            
        Returns:
            List[SearchHit]: Retrieve-Ergebnisse (nur KB-Artikel)
        """
        query = testcase.get("ticket_description", "")
        
        # Retrieval durchführen - nur KB-Artikel, keine Incidents
        # preview_chars=0 bedeutet: vollständiger Text, nicht abgeschnitten
        hits = search(
            query=query,
            top_k=self.top_k,
            use_kb=True,
            use_incidents=False,  # Incidents nicht beim Retrieval verwenden
            preview_chars=0,  # Vollständige KB-Artikel, nicht abgeschnitten
            merge=True
        )
        
        return hits
    
    def build_prompt(self, testcase: Dict[str, Any], hits: List[SearchHit]) -> str:
        """
        Erstellt Prompt aus Testfall und Retrieval-Ergebnissen.
        
        Args:
            testcase: Testfall-Dictionary
            hits: Retrieval-Ergebnisse
            
        Returns:
            str: Formatierter Prompt
        """
        context_blocks = []
        for hit in hits:
            meta = hit.metadata
            header = f"[{hit.collection.upper()} - Score: {hit.score:.4f}]\n"
            context_blocks.append(header + hit.text)
        
        context = "\n\n-----\n\n".join(context_blocks)
        
        prompt = f"""Du bist ein IT-Support-Spezialist. Analysiere das gemeldete Problem und erstelle eine sachliche, strukturierte Lösung.

Kontext aus Wissensdatenbank und früheren Incidents:
{context}

Gemeldetes Problem: {testcase.get('ticket_description', '')}

Erstelle eine präzise Problemanalyse und Lösungsanleitung nach folgendem Format:

Problemanalyse:
- Beschreibe kurz das identifizierte Problem

Lösungsschritte:
1. [Erster konkreter Handlungsschritt]
2. [Zweiter konkreter Handlungsschritt]
3. [...]

Relevante KB-Artikel: 
[Gib die KB-IDs an, falls vorhanden]

Hinweis: Nutze ausschließlich die Informationen aus dem bereitgestellten Kontext.
"""
        return prompt
    
    def ask_ollama(
        self,
        prompt: str,
        model: str,
        url: str,
        threads: int,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Sendet Anfrage an Ollama und gibt Antwort zurück.
        
        Args:
            prompt: Prompt-Text
            model: Modellname
            url: Ollama URL
            threads: Anzahl Threads
            temperature: Temperature-Wert
            
        Returns:
            Dict mit response, model, duration etc.
        """
        start_time = time.time()
        
        try:
            resp = requests.post(
                url + "/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "options": {
                        "num_thread": threads,
                        "num_ctx": self.ollama_config.num_ctx,
                        "temperature": temperature
                    },
                    "stream": False,
                },
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            
            duration = time.time() - start_time
            
            return {
                "response": data.get("response", ""),
                "model": model,
                "duration": duration,
                "success": True,
                "error": None,
                "ollama_data": data  # Komplette Ollama-Response für Token-Tracking
            }
        except Exception as e:
            logger.error("Fehler bei Ollama-Anfrage: %s", e)
            duration = time.time() - start_time
            return {
                "response": "",
                "model": model,
                "duration": duration,
                "success": False,
                "error": str(e)
            }
    
    def compute_metrics(
        self,
        hits: List[SearchHit],
        gold_kb_id: str
    ) -> Dict[str, float]:
        """
        Berechnet Retrieval-Metriken.
        
        Args:
            hits: Retrieval-Ergebnisse
            gold_kb_id: Gold-Standard KB-ID
            
        Returns:
            Dict mit Metrik-Werten
        """
        # Extrahiere IDs aus Hits
        retrieved_ids = [
            hit.metadata.get("kb_id") or hit.metadata.get("ticket_id", "")
            for hit in hits
        ]
        
        # Gold KB ID als relevant Set
        relevant_ids = [gold_kb_id] if gold_kb_id else []
        
        # Debug-Logging
        logger.debug("Retrieved IDs: %s", retrieved_ids)
        logger.debug("Gold KB ID: %s", gold_kb_id)
        logger.debug("Relevant IDs: %s", relevant_ids)
        
        # Metriken berechnen
        metrics = {}
        
        if relevant_ids:
            try:
                ndcg = self.ndcg_metric.compute(retrieved_ids, relevant_ids)
                recall = self.recall_metric.compute(retrieved_ids, relevant_ids)
                
                metrics["ndcg@k"] = ndcg
                metrics["recall@k"] = recall
                
                # Prüfe ob Gold KB in Top-K ist (nur erste k Elemente!)
                top_k_ids = retrieved_ids[:self.top_k]
                metrics["gold_in_topk"] = gold_kb_id in top_k_ids
                
                # Zusätzliches Logging für Debugging
                if ndcg == 0.0 and recall == 0.0:
                    logger.warning("Keine Übereinstimmung gefunden!")
                    logger.warning("  Gold KB: %s", gold_kb_id)
                    logger.warning("  Retrieved: %s", retrieved_ids[:5])
                
            except Exception as e:
                logger.warning("Fehler bei Metrik-Berechnung: %s", e)
                metrics["ndcg@k"] = 0.0
                metrics["recall@k"] = 0.0
                metrics["gold_in_topk"] = False
        else:
            metrics["ndcg@k"] = 0.0
            metrics["recall@k"] = 0.0
            metrics["gold_in_topk"] = False
        
        # Platzhalter für LLM-as-a-Judge Metriken
        metrics["llm_judge_score"] = None  # TODO: Implementieren
        metrics["llm_judge_relevance"] = None  # TODO: Implementieren
        
        return metrics
    
    def evaluate_testcase(
        self,
        testcase: Dict[str, Any],
        profile: Literal["low", "mid", "high", "gpu"],
        model: str,
        run: int
    ) -> Dict[str, Any]:
        """
        Evaluiert einen einzelnen Testfall.
        
        Args:
            testcase: Testfall-Dictionary
            profile: Profil (low/mid/high/gpu)
            model: Modellname
            run: Run-Nummer (1-3)
            
        Returns:
            Dict mit Evaluation-Ergebnissen
        """
        # Generiere trace_id für diesen Testfall
        trace_id = str(uuid.uuid4())
        
        # Starte System Metrics Logging (speichert in output/metrics/[experiment_id])
        metrics_output_dir = Path("output/metrics") / self.experiment_id
        metrics_logger = SystemMetricsLogger(
            experiment_id=self.experiment_id,
            output_dir=str(metrics_output_dir),
            profile=profile
        )
        metrics_logger.start_logging(trace_id)
        
        try:
            # Gesamtzeit messen (Wall-Clock)
            wall_start = time.time()
            
            logger.info("Evaluiere Testcase %s - Profile: %s - Model: %s - Run: %d (trace_id: %s)",
                       testcase.get("test_case_id"), profile, model, run, trace_id)
        
            # Latenz-Tracking initialisieren
            latency_tracker = LatencyTracker()
            latency_tracker.start_total()
            
            # 1. Retrieval
            latency_tracker.start_retrieval()
            hits = self.retrieve_for_testcase(testcase)
            latency_tracker.end_retrieval()
            
            # 2. Prompt erstellen
            prompt = self.build_prompt(testcase, hits)
            
            # 3. Ollama anfragen
            url = self.get_ollama_url_for_profile(profile)
            threads = self.get_threads_for_profile(profile)
            
            latency_tracker.start_llm()
            ollama_result = self.ask_ollama(prompt, model, url, threads)
            latency_tracker.end_llm()
            
            # 4. Metriken berechnen
            gold_kb_id = testcase.get("gold_kb_id", "")
            metrics = self.compute_metrics(hits, gold_kb_id)
            
            # Token-Metriken extrahieren
            token_tracker = TokenTracker.from_ollama_response(
                ollama_result.get("ollama_data", {})
            )
            token_metrics = token_tracker.get_metrics()
            
            # Latenz-Messungen abschließen
            latency_tracker.end_total()
            latency_metrics = latency_tracker.get_metrics()
            
            # 5. Retrieval Score berechnen
            retrieval_score = self.retrieval_score_calculator.calculate(
                recall=metrics.get("recall@k"),
                ndcg=metrics.get("ndcg@k")
            )
            
            # 6. LLM Judge Evaluation (optional)
            judge_metrics = {}
            judge_result = None  # Initialize judge_result
            if self.use_llm_judge and self.llm_judge and ollama_result["success"]:
                try:
                    # Kontext aus Hits extrahieren
                    context = "\n\n".join([
                        f"[{hit.collection.upper()}] {hit.text}"
                        for hit in hits[:3]  # Nur erste 3 für Context
                    ])
                    
                    judge_result = self.llm_judge.evaluate(
                        ticket_description=testcase.get("ticket_description", ""),
                        context=context,
                        generated_answer=ollama_result["response"]
                    )
                    
                    judge_metrics = {
                        "judge_faithfulness": judge_result.get("faithfulness"),
                        "judge_relevance": judge_result.get("relevance"),
                        "judge_completeness": judge_result.get("completeness"),
                        "judge_fluency": judge_result.get("fluency"),
                        "judge_raw_score": judge_result.get("raw_score"),
                        "judge_normalized_score": judge_result.get("normalized_score"),
                        "judge_success": judge_result.get("success"),
                        "judge_error": judge_result.get("error")
                    }
                    
                    if judge_result.get("success"):
                        logger.info("  ✓ Judge Score: %.4f (F=%.1f, R=%.1f, C=%.1f, L=%.1f)",
                                   judge_metrics["judge_normalized_score"],
                                   judge_metrics["judge_faithfulness"],
                                   judge_metrics["judge_relevance"],
                                   judge_metrics["judge_completeness"],
                                   judge_metrics["judge_fluency"])
                    else:
                        logger.warning("  ✗ Judge Evaluation fehlgeschlagen: %s", judge_result.get("error"))
                        
                except Exception as e:
                    logger.error("Fehler bei LLM Judge Evaluation: %s", e)
                    judge_metrics = {
                        "judge_faithfulness": None,
                        "judge_relevance": None,
                        "judge_completeness": None,
                        "judge_fluency": None,
                        "judge_raw_score": None,
                        "judge_normalized_score": None,
                        "judge_success": False,
                        "judge_error": str(e)
                    }
            
            # Gesamtzeit berechnen (Wall-Clock)
            total_wall_time = time.time() - wall_start
            
            # Judge Feedback extrahieren (falls vorhanden)
            judge_feedback = ""
            if judge_result and judge_result.get("success"):
                judge_feedback = judge_result.get("feedback", "")
            
            # 7. Ergebnis zusammenstellen
            result = {
                "experiment_id": self.experiment_id,
                "trace_id": trace_id,
                "test_case_id": testcase.get("test_case_id"),
                "profile": profile,
                "model": model,
                "repetition": run,
            "category": testcase.get("category"),
            "service": testcase.get("service"),
            "difficulty_level": testcase.get("difficulty_level"),
            "gold_kb_id": gold_kb_id,
            "gold_kb_fulltext": testcase.get("gold_kb_fulltext", ""),  # Gold KB Volltext hinzufügen
            "retrieved_count": len(hits),
            "response": ollama_result["response"],
            "ollama_success": ollama_result["success"],
            "ollama_error": ollama_result.get("error"),
            "total_wall_time": total_wall_time,  # Gesamtzeit inkl. allem
            "prompt": prompt,  # Vollständiger Prompt für CSV
            "llm_response": ollama_result["response"],  # Vollständige LLM-Antwort für CSV
            "judge_feedback": judge_feedback,  # Vollständiges Judge-Feedback für CSV
            "retrieval_score": retrieval_score,  # Kombinierter Retrieval Score
            **metrics,
            **latency_metrics,
            **token_metrics,
            **judge_metrics
            }
            
            logger.info("  ✓ nDCG@%d: %.4f | Recall@%d: %.4f | Gold in Top-K: %s",
                       self.top_k, metrics["ndcg@k"],
                       self.top_k, metrics["recall@k"],
                       metrics["gold_in_topk"])
            logger.info("  ✓ Latency: %.2fs (Retrieval: %.2fs, LLM: %.2fs)",
                       latency_metrics["total_latency"] or 0,
                       latency_metrics["retrieval_duration"] or 0,
                       latency_metrics["llm_duration"] or 0)
            logger.info("  ✓ Total Wall Time: %.2fs (inkl. Judge & Overhead)", total_wall_time)
            logger.info("  ✓ Tokens: %d prompt + %d generated = %d total | %.1f tok/s",
                       token_metrics["prompt_tokens"] or 0,
                       token_metrics["generated_tokens"] or 0,
                       token_metrics["total_tokens"] or 0,
                       token_metrics["tokens_per_second"] or 0)
            
            # Verkürzte LLM-Antwort loggen
            response_preview = ollama_result["response"][:150] + "..." if len(ollama_result["response"]) > 150 else ollama_result["response"]
            logger.info("  ✓ LLM Response: %s", response_preview.replace("\n", " "))
            
            # Judge-Ergebnis loggen (falls vorhanden)
            if judge_metrics.get("judge_success"):
                logger.info("  ✓ Judge: F=%.1f R=%.1f C=%.1f L=%.1f → Score=%.3f",
                           judge_metrics["judge_faithfulness"] or 0,
                           judge_metrics["judge_relevance"] or 0,
                           judge_metrics["judge_completeness"] or 0,
                           judge_metrics["judge_fluency"] or 0,
                           judge_metrics["judge_normalized_score"] or 0)
            
            return result
        
        finally:
            # Stoppe System Metrics Logging
            metrics_logger.stop_logging()
    
    def log_testcase_details(self, testcases: List[Dict[str, Any]]) -> str:
        """
        Loggt Testcase-Retrieval-Metriken in eine separate Log-Datei.
        Format: TC-ID: gold=KB-ID | found=True/False | rank=N | score=X.XX | retrieve_k=K | in_retrieve_k=True/False | retrieve_rank=N
        
        Args:
            testcases: Liste der Testcase-Dictionaries
            
        Returns:
            str: Pfad zur Log-Datei
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"testcases_retrieval_{timestamp}.log"
        
        logger.info("📊 Führe Retrieval-Tests für alle Testcases durch...")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            # Header schreiben
            f.write(f"✓ {len(testcases)} Testcases Retrieval-Test\n")
            f.write(f"Spalten: test_case_id, gold_kb_id, found, rank, score, retrieve_k, in_retrieve_k, retrieve_rank\n")
            
            success_count = 0
            
            # Für jeden Testcase Retrieval durchführen
            for tc in testcases:
                test_case_id = tc.get('test_case_id', 'N/A')
                gold_kb_id = tc.get('gold_kb_id', '')
                
                # Testcase-Query zusammenstellen
                query = f"{tc.get('ticket_title', '')} {tc.get('ticket_description', '')}"
                
                try:
                    # Retrieval durchführen (größeres K für retrieve_rank)
                    hits = search(query, top_k=40, preview_chars=0)
                    
                    # Gold KB suchen
                    found = False
                    rank = None
                    score = None
                    retrieve_rank = None
                    
                    # In Top-K suchen
                    for i, hit in enumerate(hits[:self.top_k], 1):
                        hit_kb_id = hit.metadata.get('kb_id') if hit.metadata else None
                        if hit_kb_id == gold_kb_id:
                            found = True
                            rank = i
                            score = hit.score
                            break
                    
                    # In vollständigem Retrieval-Set suchen (für retrieve_rank)
                    for i, hit in enumerate(hits, 1):
                        hit_kb_id = hit.metadata.get('kb_id') if hit.metadata else None
                        if hit_kb_id == gold_kb_id:
                            retrieve_rank = i
                            break
                    
                    in_retrieve_k = retrieve_rank is not None and retrieve_rank <= 40
                    
                    if found:
                        success_count += 1
                    
                    # Log-Zeile schreiben
                    f.write(f"{test_case_id}: gold={gold_kb_id or 'nan'} | "
                           f"found={found} | "
                           f"rank={rank} | "
                           f"score={score if score else 'None'} | "
                           f"retrieve_k={self.top_k} | "
                           f"in_retrieve_k={in_retrieve_k} | "
                           f"retrieve_rank={retrieve_rank}\n")
                    
                except Exception as e:
                    logger.error(f"Fehler bei Retrieval für {test_case_id}: {e}")
                    f.write(f"{test_case_id}: ERROR | {str(e)}\n")
            
            # Summary schreiben
            f.write(f"\nSummary: found {success_count}/{len(testcases)} testcases with gold in top-{self.top_k}\n")
        
        logger.info("📝 Testcase-Retrieval-Metriken geloggt: %s", log_file)
        logger.info(f"   ✓ {success_count}/{len(testcases)} Gold-KB in Top-{self.top_k} gefunden")
        return str(log_file)
    
    def run_evaluation(
        self,
        testcases_file: str = None,
        profiles: List[str] = None
    ) -> str:
        """
        Führt komplette Evaluation durch.
        
        Args:
            testcases_file: Pfad zur Testcases-CSV (optional)
            profiles: Liste der Profile (default: ["low", "mid", "high", "gpu"])
            
        Returns:
            str: Pfad zur Ergebnis-CSV
        """
        logger.info("=== RAG Evaluation gestartet ===")
        
        if profiles is None:
            profiles = ["low", "mid", "high", "gpu"]
        
        # Testfälle laden
        logger.info("📁 Lade Testfälle...")
        df = load_testcases(testcases_file or TESTCASES_FILE)
        testcases = df.to_dict(orient="records")
        logger.info("✓ %d Testfälle geladen", len(testcases))
        
        # Testcase-Details loggen
        self.log_testcase_details(testcases)
        
        # Output-Datei vorbereiten
        date_str = datetime.now().strftime("%y%m%d")
        output_file = self._get_next_run_file(self.output_dir, date_str)
        
        # CSV-Header schreiben (trace_id für Korrelation mit System-Metriken)
        fieldnames = [
            "experiment_id",
            "trace_id",
            "profile",
            "model",
            "test_case_id",
            "repetition",
            "recall@k",
            "ndcg@k",
            "retrieval_score",
            "retrieval_interpretation",
            "latency_ms",
            "total_wall_time_ms",
            "tokens_per_s",
            "llm_judge_f",
            "llm_judge_r",
            "llm_judge_c",
            "llm_judge_l",
            "llm_judge_score",
            "llm_judge_interpretation",
            "prompt_tokens",
            "completion_tokens",
            "error_flag",
            "gold_kb_id",
            "gold_kb_fulltext",
            "prompt",
            "llm_response",
            "judge_feedback"
        ]
        
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        
            logger.info("💾 Speichere Ergebnisse kontinuierlich nach: %s", output_file)
        logger.info("📏 System-Metriken werden gespeichert in: output/metrics/%s/", self.experiment_id)
        
        # Ergebnis-Liste für Aggregation
        results = []
        
        # Für jedes Profil
        for profile in profiles:
            logger.info("\n=== Profil: %s ===", profile.upper())
            models = self.get_models_for_profile(profile)
            logger.info("Modelle: %s", models)
            
            # Für jedes Modell
            for model in models:
                logger.info("\n--- Modell: %s ---", model)
                
                # Für jeden Testfall
                for testcase in testcases:
                    # Mehrfach-Runs pro Testfall
                    for run in range(1, self.runs_per_testcase + 1):
                        result = self.evaluate_testcase(testcase, profile, model, run)
                        results.append(result)
                        
                        # Sofort nach jedem Run speichern
                        # Berechne Judge Interpretation falls Raw Score vorhanden
                        judge_interpretation = None
                        if result.get("judge_raw_score") is not None:
                            from rag_csv.utils.answer_quality import AnswerQualityCalculator
                            judge_interpretation = AnswerQualityCalculator.get_interpretation(result.get("judge_raw_score"))
                        
                        # Berechne Retrieval Interpretation
                        retrieval_interpretation = self.retrieval_score_calculator.get_interpretation(
                            recall=result.get("recall@k"),
                            ndcg=result.get("ndcg@k")
                        )
                        
                        compact = {
                            "experiment_id": result.get("experiment_id"),
                            "trace_id": result.get("trace_id"),
                            "profile": result.get("profile"),
                            "model": result.get("model"),
                            "test_case_id": result.get("test_case_id"),
                            "repetition": result.get("repetition"),
                            "recall@k": result.get("recall@k"),
                            "ndcg@k": result.get("ndcg@k"),
                            "retrieval_score": result.get("retrieval_score"),
                            "retrieval_interpretation": retrieval_interpretation,
                            "latency_ms": int(result.get("total_latency", 0) * 1000) if result.get("total_latency") else None,
                            "total_wall_time_ms": int(result.get("total_wall_time", 0) * 1000) if result.get("total_wall_time") else None,
                            "tokens_per_s": result.get("tokens_per_second"),
                            "llm_judge_f": result.get("judge_faithfulness"),
                            "llm_judge_r": result.get("judge_relevance"),
                            "llm_judge_c": result.get("judge_completeness"),
                            "llm_judge_l": result.get("judge_fluency"),
                            "llm_judge_score": result.get("judge_normalized_score"),
                            "llm_judge_interpretation": judge_interpretation,
                            "prompt_tokens": result.get("prompt_tokens"),
                            "completion_tokens": result.get("generated_tokens"),
                            "error_flag": 0 if result.get("ollama_success") else 1,
                            "gold_kb_id": result.get("gold_kb_id", ""),
                            "gold_kb_fulltext": result.get("gold_kb_fulltext", ""),
                            "prompt": result.get("prompt", ""),
                            "llm_response": result.get("llm_response", ""),
                            "judge_feedback": result.get("judge_feedback", "")
                        }
                        
                        with open(output_file, "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writerow(compact)
        
        logger.info("\n💾 Alle Ergebnisse gespeichert: %s", output_file)
        logger.info("=== Evaluation abgeschlossen ===")
        logger.info("Ergebnisse: %s", output_file)
        logger.info("Gesamt: %d Evaluationen", len(results))
        
        # Berechne Token- und Latency-Scores
        logger.info("\n📊 Berechne Performance-Scores...")
        
        token_calculator = TokenScoreCalculator()
        token_scores = token_calculator.calculate_scores(results)
        
        latency_calculator = LatencyScoreCalculator()
        latency_scores = latency_calculator.calculate_scores(results)
        
        # Speichere Scores in separater Datei
        scores_file = self.output_dir / f"scores_{self.experiment_id}.csv"
        all_models = set(list(token_scores.keys()) + list(latency_scores.keys()))
        
        with open(scores_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "token_score", "latency_score"])
            writer.writeheader()
            
            for model in sorted(all_models):
                writer.writerow({
                    "model": model,
                    "token_score": token_scores.get(model),
                    "latency_score": latency_scores.get(model)
                })
        
        logger.info("Performance-Scores gespeichert: %s", scores_file)
        
        return str(output_file)
    



def main():
    """CLI Entry Point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Answer Orchestrator - Testfall Evaluation")
    parser.add_argument(
        "--testcases",
        type=str,
        help="Pfad zur Testcases CSV"
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=["low", "mid", "high", "gpu"],
        default=["low", "mid", "high"],
        help="Profile für Evaluation (default: all)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Anzahl der zu retrievenden Dokumente (default: {_eval_config.top_k} aus .env)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help=f"Anzahl der Durchläufe pro Testfall (default: {_eval_config.runs_per_testcase} aus .env)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/experiment",
        help="Ausgabeverzeichnis (default: output/experiment)"
    )
    
    args = parser.parse_args()
    
    orchestrator = RAGAnswerOrchestrator(
        top_k=args.top_k,
        runs_per_testcase=args.runs,
        output_dir=args.output_dir
    )
    
    try:
        output_file = orchestrator.run_evaluation(
            testcases_file=args.testcases,
            profiles=args.profiles
        )
        print(f"\n✅ Evaluation abgeschlossen")
        print(f"Ergebnisse: {output_file}")
    except Exception as e:
        logger.error("❌ Fehler: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
