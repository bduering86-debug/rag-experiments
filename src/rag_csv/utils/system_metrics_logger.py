#!/usr/bin/env python3
"""
System Metrics Logger für Monitoring der Systemauslastung während der Evaluation.

Sammelt Metriken von einem externen Metrics Service in konfigurierbaren Intervallen
und speichert diese mit trace_id für spätere Korrelation.
"""

import csv
import time
import requests
import threading
from typing import Literal, Optional
from pathlib import Path
from datetime import datetime

from rag_csv.config.logging import get_logger
from rag_csv.config.settings import OllamaConfig


class SystemMetricsLogger:
    """
    Loggt System-Metriken von einem Metrics-Service während der Testfall-Ausführung.
    
    Sammelt periodisch Metriken und speichert diese in einer CSV-Datei mit trace_id
    für die Korrelation mit den Evaluation-Ergebnissen.
    """
    
    def __init__(
        self,
        experiment_id: str,
        output_dir: str = "output/experiment",
        profile: Literal["low", "mid", "high", "gpu", "local"] = "low",
        file_prefix: str = "system_metrics"
    ):
        """
        Initialisiert den System Metrics Logger.
        
        Args:
            experiment_id: Eindeutige ID des Experiments
            output_dir: Verzeichnis für die Metrics-CSV
            profile: Profil für die Metrics-Endpoint-Auswahl
            file_prefix: Präfix für CSV-Dateiname (z.B. "system_metrics" oder "embedding_metrics")
        """
        self.logger = get_logger(f"{__name__}.SystemMetricsLogger")
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.file_prefix = file_prefix
        
        # Lade Config
        self.config = OllamaConfig()
        
        # Wähle Endpoint basierend auf Profil
        self.metrics_endpoint = self._get_metrics_endpoint(profile)
        # Verwende lokales Intervall für "local" Profile, sonst Standard-Intervall
        self.interval = self.config.metrics_local_interval if profile == "local" else self.config.metrics_interval
        
        # CSV-Datei vorbereiten
        self.csv_file = self.output_dir / f"{self.file_prefix}_{self.experiment_id}.csv"
        self._init_csv()
        
        # Thread-Kontrolle
        self._stop_event = threading.Event()
        self._logging_thread: Optional[threading.Thread] = None
        
        self.logger.info(f"System Metrics Logger initialisiert - Endpoint: {self.metrics_endpoint}, Intervall: {self.interval}s")
    
    def _get_metrics_endpoint(self, profile: Literal["low", "mid", "high", "gpu", "local"]) -> str:
        """
        Gibt den Metrics-Endpoint für das angegebene Profil zurück.
        
        Args:
            profile: Profil-Name
            
        Returns:
            str: Metrics-Endpoint URL
        """
        if profile == "local":
            return self.config.metrics_local_endpoint
        elif profile == "low":
            return self.config.metrics_low_endpoint
        elif profile == "mid":
            return self.config.metrics_mid_endpoint
        elif profile == "high":
            return self.config.metrics_high_endpoint
        elif profile == "gpu":
            return self.config.metrics_ultra_endpoint
        
        self.logger.warning(f"Unbekanntes Profil: {profile}, verwende local-endpoint")
        return self.config.metrics_local_endpoint
    
    def _init_csv(self):
        """Initialisiert die CSV-Datei mit Header."""
        if not self.csv_file.exists():
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "timestamp",
                    "trace_id",
                    "profile",
                    "snapshot_type",
                    "cpu_usage",
                    "memory_usage",
                    "ram_used_mb",
                    "ram_available_mb",
                    "ram_total_mb",
                    "ollama_proc_cpu_percent",
                    "ollama_proc_rss_mb",
                    "gpu_usage",
                    "gpu_memory",
                    "host",
                    "ts_epoch",
                    "response_raw"
                ])
                writer.writeheader()
            self.logger.info(f"Metrics CSV initialisiert: {self.csv_file}")
    
    def _fetch_metrics(self, trace_id: str, snapshot_type: str = "continuous") -> dict:
        """
        Ruft Metriken vom Metrics-Service ab.
        
        Args:
            trace_id: Trace-ID für Korrelation
            snapshot_type: Art des Snapshots ("pre", "continuous", "post")
            
        Returns:
            dict: Metrics-Daten
        """
        if not self.metrics_endpoint:
            return {
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id,
                "profile": self.profile,
                "snapshot_type": snapshot_type,
                "cpu_usage": None,
                "memory_usage": None,
                "ram_used_mb": None,
                "ram_available_mb": None,
                "ram_total_mb": None,
                "ollama_proc_cpu_percent": None,
                "ollama_proc_rss_mb": None,
                "gpu_usage": None,
                "gpu_memory": None,
                "host": None,
                "ts_epoch": None,
                "response_raw": "NO_ENDPOINT_CONFIGURED"
            }
        
        try:
            headers = {"X-Trace-ID": trace_id}
            response = requests.get(
                self.metrics_endpoint,
                headers=headers,
                timeout=5
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extrahiere relevante Metriken basierend auf dem tatsächlichen Response-Format
            # Unterstützt sowohl alte als auch neue Attribut-Namen
            return {
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id or data.get("trace_id", ""),
                "profile": self.profile,
                "snapshot_type": snapshot_type,
                "cpu_usage": data.get("cpu_system_percent"),
                "memory_usage": data.get("ram_system_percent"),
                "ram_used_mb": data.get("ram_used_mb"),
                "ram_available_mb": data.get("ram_available_mb"),
                "ram_total_mb": data.get("ram_total_mb"),
                "ollama_proc_cpu_percent": data.get("ollama_proc_cpu_percent"),
                "ollama_proc_rss_mb": data.get("ollama_proc_rss_mb"),
                "gpu_usage": data.get("gpu_usage"),
                "gpu_memory": data.get("gpu_memory"),
                "host": data.get("host"),
                "ts_epoch": data.get("ts_epoch"),
                "response_raw": str(data)
            }
            
        except requests.RequestException as e:
            self.logger.warning(f"Fehler beim Abrufen der Metriken: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id,
                "profile": self.profile,
                "snapshot_type": snapshot_type,
                "cpu_usage": None,
                "memory_usage": None,
                "ram_used_mb": None,
                "ram_available_mb": None,
                "ram_total_mb": None,
                "ollama_proc_cpu_percent": None,
                "ollama_proc_rss_mb": None,
                "gpu_usage": None,
                "gpu_memory": None,
                "host": None,
                "ts_epoch": None,
                "response_raw": f"ERROR: {str(e)}"
            }
    
    def _write_metrics(self, metrics: dict):
        """
        Schreibt Metriken in die CSV-Datei.
        
        Args:
            metrics: Dictionary mit Metrik-Daten
        """
        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp",
                "trace_id",
                "profile",
                "snapshot_type",
                "cpu_usage",
                "memory_usage",
                "ram_used_mb",
                "ram_available_mb",
                "ram_total_mb",
                "ollama_proc_cpu_percent",
                "ollama_proc_rss_mb",
                "gpu_usage",
                "gpu_memory",
                "host",
                "ts_epoch",
                "response_raw"
            ])
            writer.writerow(metrics)
    
    def _log_metrics_loop(self, trace_id: str):
        """
        Logging-Loop der periodisch Metriken sammelt.
        
        Args:
            trace_id: Trace-ID für diesen Testfall
        """
        while not self._stop_event.is_set():
            metrics = self._fetch_metrics(trace_id, snapshot_type="continuous")
            self._write_metrics(metrics)
            
            # Warte für das nächste Intervall oder bis Stop-Signal
            self._stop_event.wait(self.interval)
    
    def capture_snapshot(self, trace_id: str, snapshot_type: str = "pre"):
        """
        Erfasst einen einzelnen Snapshot der System-Metriken.
        
        Args:
            trace_id: Trace-ID für Korrelation
            snapshot_type: Art des Snapshots ("pre" = vor Testfall, "post" = nach Testfall)
        """
        metrics = self._fetch_metrics(trace_id, snapshot_type=snapshot_type)
        self._write_metrics(metrics)
        self.logger.debug(f"{snapshot_type.upper()}-Snapshot erfasst für trace_id: {trace_id}")
    
    def start_logging(self, trace_id: str):
        """
        Startet das periodische Logging der Metriken.
        
        Args:
            trace_id: Eindeutige Trace-ID für diesen Testfall
        """
        if self._logging_thread and self._logging_thread.is_alive():
            self.logger.warning("Logging läuft bereits, stoppe zuerst den vorherigen Thread")
            self.stop_logging()
        
        self._stop_event.clear()
        self._logging_thread = threading.Thread(
            target=self._log_metrics_loop,
            args=(trace_id,),
            daemon=True
        )
        self._logging_thread.start()
        self.logger.debug(f"Metrics Logging gestartet für trace_id: {trace_id}")
    
    def stop_logging(self):
        """Stoppt das Metrics-Logging."""
        if self._logging_thread and self._logging_thread.is_alive():
            self._stop_event.set()
            self._logging_thread.join(timeout=self.interval + 2)
            self.logger.debug("Metrics Logging gestoppt")
