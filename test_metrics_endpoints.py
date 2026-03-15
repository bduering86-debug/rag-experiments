#!/usr/bin/env python3
"""
Test-Skript für System Metrics Logger - alle Endpoints und CSV-Funktionalität
"""

import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

from src.rag_csv.utils.system_metrics_logger import SystemMetricsLogger
from src.rag_csv.config.settings import OllamaConfig

def test_endpoint(profile: str, endpoint: str):
    """Testet einen einzelnen Metrics-Endpoint."""
    print(f"\n{'='*70}")
    print(f"Test für Profil: {profile}")
    print(f"Endpoint: {endpoint}")
    print(f"{'='*70}")
    
    if not endpoint:
        print(f"❌ Kein Endpoint konfiguriert für Profil '{profile}'")
        return False
    
    try:
        # Test 1: Endpoint erreichbar
        print(f"\n1. Teste Erreichbarkeit...")
        response = requests.get(endpoint, timeout=5)
        response.raise_for_status()
        
        print(f"   ✓ Status Code: {response.status_code}")
        
        # Test 2: JSON-Format
        print(f"\n2. Teste JSON-Format...")
        data = response.json()
        print(f"   ✓ JSON erfolgreich geparst")
        
        # Test 3: Erwartete Felder prüfen
        print(f"\n3. Prüfe erwartete Felder...")
        expected_fields = [
            "cpu_system_percent",
            "ram_system_percent",
            "ram_used_mb",
            "ram_available_mb",
            "ram_total_mb",
            "ollama_proc_cpu_percent",
            "ollama_proc_rss_mb"
        ]
        
        # GPU-spezifische Felder (optional)
        gpu_fields = ["gpu_usage", "gpu_memory"]
        
        missing_fields = []
        for field in expected_fields:
            if field in data:
                value = data[field]
                print(f"   ✓ {field}: {value}")
            else:
                missing_fields.append(field)
                print(f"   ⚠ {field}: FEHLT")
        
        # GPU-Felder prüfen (für GPU-Profil)
        if profile == "gpu":
            print(f"\n   GPU-Felder:")
            for field in gpu_fields:
                if field in data:
                    value = data[field]
                    print(f"   ✓ {field}: {value}")
                else:
                    print(f"   ⚠ {field}: FEHLT (optional)")
        
        # Test 4: Zusätzliche Felder anzeigen
        print(f"\n4. Zusätzliche Felder:")
        all_checked = expected_fields + gpu_fields
        for key, value in data.items():
            if key not in all_checked:
                print(f"   • {key}: {value}")
        
        if missing_fields:
            print(f"\n⚠ Warnung: {len(missing_fields)} Felder fehlen: {', '.join(missing_fields)}")
        else:
            print(f"\n✅ Alle erwarteten Felder vorhanden")
        
        return True
        
    except requests.RequestException as e:
        print(f"❌ Fehler beim Verbinden zum Endpoint: {e}")
        return False
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        return False


def test_csv_writing(profile: str):
    """Testet das Schreiben der CSV für ein Profil."""
    print(f"\n{'='*70}")
    print(f"Test CSV-Schreiben für Profil: {profile}")
    print(f"{'='*70}")
    
    try:
        # Test-Verzeichnis erstellen
        test_dir = Path("output/test_metrics")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # SystemMetricsLogger initialisieren
        experiment_id = f"test_{profile}_{int(time.time())}"
        logger = SystemMetricsLogger(
            experiment_id=experiment_id,
            output_dir=str(test_dir),
            profile=profile
        )
        
        print(f"\n1. SystemMetricsLogger initialisiert")
        print(f"   Endpoint: {logger.metrics_endpoint}")
        print(f"   CSV-Datei: {logger.csv_file}")
        
        # Teste Metrics-Abruf
        print(f"\n2. Teste Metrics-Abruf...")
        test_trace_id = f"test_trace_{profile}_{int(time.time())}"
        metrics = logger._fetch_metrics(test_trace_id)
        
        print(f"   ✓ Metrics abgerufen:")
        for key, value in metrics.items():
            if key != "response_raw":
                print(f"     • {key}: {value}")
        
        # Teste CSV-Schreiben
        print(f"\n3. Teste CSV-Schreiben...")
        logger.start_logging(test_trace_id)
        time.sleep(2)  # 2 Sekunden sammeln
        logger.stop_logging()
        
        # Prüfe ob CSV existiert und Daten enthält
        if logger.csv_file.exists():
            with open(logger.csv_file, "r") as f:
                lines = f.readlines()
                print(f"   ✓ CSV-Datei existiert")
                print(f"   ✓ Zeilen geschrieben: {len(lines)} (inkl. Header)")
                
                if len(lines) > 1:
                    print(f"\n   Erste Datenzeile (gekürzt):")
                    data_line = lines[1][:200]
                    print(f"   {data_line}...")
                    print(f"\n✅ CSV erfolgreich geschrieben")
                    return True
                else:
                    print(f"   ⚠ Keine Daten geschrieben")
                    return False
        else:
            print(f"   ❌ CSV-Datei wurde nicht erstellt")
            return False
            
    except Exception as e:
        print(f"❌ Fehler beim CSV-Test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("System Metrics Endpoints Test")
    print("="*70)
    
    # Lade Config
    config = OllamaConfig()
    
    # Definiere Profiles und Endpoints
    profiles = {
        "low": config.metrics_low_endpoint,
        "mid": config.metrics_mid_endpoint,
        "high": config.metrics_high_endpoint,
        "gpu": config.metrics_ultra_endpoint
    }
    
    print(f"\nKonfigurierte Endpoints:")
    for profile, endpoint in profiles.items():
        status = "✓" if endpoint else "✗"
        print(f"  {status} {profile:10} -> {endpoint or 'NICHT KONFIGURIERT'}")
    
    # Test 1: Alle Endpoints testen
    print(f"\n\n{'#'*70}")
    print("# PHASE 1: Endpoint-Erreichbarkeit")
    print(f"{'#'*70}")
    
    endpoint_results = {}
    for profile, endpoint in profiles.items():
        result = test_endpoint(profile, endpoint)
        endpoint_results[profile] = result
    
    # Test 2: CSV-Funktionalität für verfügbare Endpoints
    print(f"\n\n{'#'*70}")
    print("# PHASE 2: CSV-Schreib-Funktionalität")
    print(f"{'#'*70}")
    
    csv_results = {}
    for profile, endpoint_ok in endpoint_results.items():
        if endpoint_ok:
            result = test_csv_writing(profile)
            csv_results[profile] = result
        else:
            print(f"\n⊘ Überspringe CSV-Test für '{profile}' (Endpoint nicht erreichbar)")
            csv_results[profile] = False
    
    # Zusammenfassung
    print(f"\n\n{'='*70}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*70}")
    
    print(f"\nEndpoint-Tests:")
    for profile, result in endpoint_results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {profile:10}")
    
    print(f"\nCSV-Tests:")
    for profile, result in csv_results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {profile:10}")
    
    # Gesamtergebnis
    total_endpoint = sum(1 for r in endpoint_results.values() if r)
    total_csv = sum(1 for r in csv_results.values() if r)
    
    print(f"\n{'='*70}")
    print(f"Endpoint-Tests erfolgreich: {total_endpoint}/{len(endpoint_results)}")
    print(f"CSV-Tests erfolgreich:      {total_csv}/{len(csv_results)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
