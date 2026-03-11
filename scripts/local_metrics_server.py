#!/usr/bin/env python3
"""
Lokaler System Metrics Server für Embedding-Phase Monitoring.
Stellt System-Metriken (CPU, RAM, Prozess-Info) im gleichen Format wie der Remote Ollama Metrics Server bereit.
"""

import time
import logging
import psutil
from flask import Flask, request, jsonify
from datetime import datetime

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Port-Konfiguration - WICHTIG: Muss ein anderer Port als der Embedding-Server sein!
METRICS_PORT = 8081  # Embedding-Server läuft auf 8080
HOST = "0.0.0.0"


def get_system_metrics():
    """
    Erfasst System-Metriken ähnlich wie der Remote Ollama Metrics Server.
    
    Returns:
        dict: System-Metriken (CPU, RAM, optional GPU)
    """
    # System-weite Metriken
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    ram_used_mb = memory.used / (1024 * 1024)
    ram_available_mb = memory.available / (1024 * 1024)
    ram_total_mb = memory.total / (1024 * 1024)
    ram_percent = memory.percent
    
    # Versuche Python-Embedding-Prozess zu finden (Text-Embeddings-Inference oder ähnlich)
    embedding_proc_cpu = 0.0
    embedding_proc_rss_mb = 0.0
    
    try:
        # Suche nach text-embeddings Prozess
        for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_info']):
            try:
                proc_name = proc.info['name'].lower()
                if 'text-embed' in proc_name or 'embedding' in proc_name:
                    embedding_proc_cpu += proc.info.get('cpu_percent', 0.0) or 0.0
                    mem_info = proc.info.get('memory_info')
                    if mem_info:
                        embedding_proc_rss_mb += mem_info.rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.warning(f"Fehler beim Abrufen der Embedding-Prozess-Metriken: {e}")
    
    # GPU-Metriken (optional - falls nvidia-smi verfügbar)
    gpu_usage = None
    gpu_memory = None
    
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) == 2:
                gpu_usage = float(parts[0].strip())
                gpu_memory = float(parts[1].strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass  # GPU nicht verfügbar oder nvidia-smi nicht installiert
    
    return {
        "cpu_system_percent": round(cpu_percent, 1),
        "ram_system_percent": round(ram_percent, 1),
        "ram_used_mb": round(ram_used_mb, 3),
        "ram_available_mb": round(ram_available_mb, 3),
        "ram_total_mb": round(ram_total_mb, 3),
        "ollama_proc_cpu_percent": round(embedding_proc_cpu, 1),  # Embedding-Prozess statt Ollama
        "ollama_proc_rss_mb": round(embedding_proc_rss_mb, 3),
        "gpu_usage": gpu_usage,
        "gpu_memory": gpu_memory
    }


@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Metrics Endpoint - kompatibel mit SystemMetricsLogger.
    
    Erwartet optional X-Trace-ID Header für Korrelation.
    """
    # Trace-ID aus Header extrahieren
    trace_id = request.headers.get('X-Trace-ID', 'no-trace-id')
    
    # System-Metriken erfassen
    metrics_data = get_system_metrics()
    
    # Response im gleichen Format wie Remote Ollama Server
    response = {
        "ts_epoch": time.time(),
        "trace_id": trace_id,
        "host": "rag-orchestrator",  # Hostname zur Identifikation
        **metrics_data
    }
    
    logger.debug(f"Metrics abgerufen für trace_id={trace_id}")
    
    return jsonify(response)


@app.route('/health', methods=['GET'])
def health():
    """Health Check Endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "local-metrics-server"
    })


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Lokaler System Metrics Server")
    logger.info("=" * 60)
    logger.info(f"Port: {METRICS_PORT}")
    logger.info(f"Metrics Endpoint: http://localhost:{METRICS_PORT}/metrics")
    logger.info(f"Health Endpoint: http://localhost:{METRICS_PORT}/health")
    logger.info("=" * 60)
    logger.info("Verwendung im .env:")
    logger.info(f"  METRICS_LOCAL_ENDPOINT=http://localhost:{METRICS_PORT}/metrics")
    logger.info("=" * 60)
    
    app.run(
        host=HOST,
        port=METRICS_PORT,
        debug=False,
        use_reloader=False
    )
