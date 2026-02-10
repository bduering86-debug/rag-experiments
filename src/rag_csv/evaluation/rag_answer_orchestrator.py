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
from typing import List, Dict, Any, Literal
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics

from rag_csv.config.logging import get_logger
from rag_csv.config.settings import DataConfig, OllamaConfig, EvaluationConfig
from rag_csv.data.load_testcases import load_testcases, TESTCASES_FILE
from rag_csv.core.retrieval import search, SearchHit
from rag_csv.utils.nDCGTopK import nDCGTopK
from rag_csv.utils.RecallTopK import RecallTopK
from rag_csv.utils.latency import LatencyTracker
from rag_csv.utils.tokens import TokenTracker
from rag_csv.utils.llm_judge import LLMJudge

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
        # Gesamtzeit messen (Wall-Clock)
        wall_start = time.time()
        
        logger.info("Evaluiere Testcase %s - Profile: %s - Model: %s - Run: %d",
                   testcase.get("test_case_id"), profile, model, run)
        
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
        
        # 5. LLM Judge Evaluation (optional)
        judge_metrics = {}
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
        
        # 6. Ergebnis zusammenstellen
        result = {
            "experiment_id": self.experiment_id,
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
        
        # Output-Datei vorbereiten
        date_str = datetime.now().strftime("%y%m%d")
        output_file = self._get_next_run_file(self.output_dir, date_str)
        
        # CSV-Header schreiben
        fieldnames = [
            "experiment_id",
            "profile",
            "model",
            "test_case_id",
            "repetition",
            "recall@k",
            "ndcg@k",
            "latency_ms",
            "total_wall_time_ms",
            "tokens_per_s",
            "llm_judge_score",
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
                        compact = {
                            "experiment_id": result.get("experiment_id"),
                            "profile": result.get("profile"),
                            "model": result.get("model"),
                            "test_case_id": result.get("test_case_id"),
                            "repetition": result.get("repetition"),
                            "recall@k": result.get("recall@k"),
                            "ndcg@k": result.get("ndcg@k"),
                            "latency_ms": int(result.get("total_latency", 0) * 1000) if result.get("total_latency") else None,
                            "total_wall_time_ms": int(result.get("total_wall_time", 0) * 1000) if result.get("total_wall_time") else None,
                            "tokens_per_s": result.get("tokens_per_second"),
                            "llm_judge_score": result.get("judge_normalized_score"),
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
        
        # Aggregiere Ergebnisse nach Testfall × Modell × Profile
        logger.info("\n📊 Aggregiere Ergebnisse...")
        agg_file = self._aggregate_results(results)
        logger.info("Aggregierte Ergebnisse: %s", agg_file)
        
        return str(output_file)
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Aggregiert Ergebnisse nach Testfall × Modell × Profile.
        
        Berechnet Mittelwerte und Mediane über alle Runs.
        
        Args:
            results: Liste der Evaluation-Ergebnisse
            
        Returns:
            str: Pfad zur aggregierten CSV
        """
        # Gruppiere nach (experiment_id, profile, model, test_case_id)
        groups = defaultdict(list)
        
        for r in results:
            key = (
                r.get("experiment_id"),
                r.get("profile"),
                r.get("model"),
                r.get("test_case_id")
            )
            groups[key].append(r)
        
        # Aggregiere jede Gruppe
        aggregated = []
        
        for key, group_results in groups.items():
            experiment_id, profile, model, test_case_id = key
            
            # Extrahiere Werte
            recalls = [r.get("recall@k") for r in group_results if r.get("recall@k") is not None]
            ndcgs = [r.get("ndcg@k") for r in group_results if r.get("ndcg@k") is not None]
            latencies = [r.get("total_latency") for r in group_results if r.get("total_latency") is not None]
            wall_times = [r.get("total_wall_time") for r in group_results if r.get("total_wall_time") is not None]
            tokens_per_s = [r.get("tokens_per_second") for r in group_results if r.get("tokens_per_second") is not None]
            judge_scores = [r.get("judge_normalized_score") for r in group_results if r.get("judge_normalized_score") is not None]
            
            # Berechne Statistiken
            agg = {
                "experiment_id": experiment_id,
                "profile": profile,
                "model": model,
                "test_case_id": test_case_id,
                "n_runs": len(group_results),
                "recall_mean": statistics.mean(recalls) if recalls else None,
                "ndcg_mean": statistics.mean(ndcgs) if ndcgs else None,
                "latency_mean": statistics.mean(latencies) if latencies else None,
                "latency_median": statistics.median(latencies) if latencies else None,
                "wall_time_mean": statistics.mean(wall_times) if wall_times else None,
                "wall_time_median": statistics.median(wall_times) if wall_times else None,
                "tokens_s_mean": statistics.mean(tokens_per_s) if tokens_per_s else None,
                "tokens_s_median": statistics.median(tokens_per_s) if tokens_per_s else None,
                "judge_mean": statistics.mean(judge_scores) if judge_scores else None,
            }
            
            aggregated.append(agg)
        
        # Speichere aggregierte Ergebnisse
        output_file = self.output_dir / f"case_agg_{self.experiment_id}.csv"
        
        fieldnames = [
            "experiment_id",
            "profile",
            "model",
            "test_case_id",
            "n_runs",
            "recall_mean",
            "ndcg_mean",
            "latency_mean",
            "latency_median",
            "wall_time_mean",
            "wall_time_median",
            "tokens_s_mean",
            "tokens_s_median",
            "judge_mean"
        ]
        
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregated)
        
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
