#!/bin/bash
# Start-Skript für den lokalen System Metrics Server

# Aktiviere Virtual Environment
source ~/lcenv/bin/activate

# Wechsle ins Projektverzeichnis
cd /home/bduering/rag_csv

# Starte den Metrics Server
python scripts/local_metrics_server.py
