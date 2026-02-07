#!/usr/bin/env python3
"""
Script zum Laden und Verarbeiten der Testcases-Datei.
"""

import pandas as pd
import os
from pathlib import Path

# Pfad zur Testcases-Datei
BASE_DIR = Path(__file__).parent.parent.parent.parent
TESTCASES_FILE = BASE_DIR / "data" / "testcaes.csv"


def load_testcases(file_path: str | Path = TESTCASES_FILE) -> pd.DataFrame:
    """
    Lädt die Testcases-CSV-Datei und gibt einen bereinigten DataFrame zurück.
    
    Args:
        file_path: Pfad zur CSV-Datei (Standard: data/testcaes.csv)
    
    Returns:
        pd.DataFrame: DataFrame mit den Testcases
    """
    # CSV mit korrektem Separator laden (Semikolon)
    # Encoding: ISO-8859-1 / Latin-1 für deutsche Umlaute
    df = pd.read_csv(file_path, sep=';', encoding='latin-1')
    
    # Erste Spalte ist fehlerhaft zusammengesetzt - umbenennen
    if 'prompt:test_casetest_case_id' in df.columns:
        df = df.rename(columns={'prompt:test_casetest_case_id': 'test_case_id'})
    
    # Relevante Spalten identifizieren und bereinigen
    # Die wichtigsten Spalten basierend auf der Struktur:
    expected_columns = [
        'test_case_id',
        'ticket_title', 
        'ticket_description',
        'category',
        'service',
        'issue_type',
        'difficulty_level',
        'gold_kb_id',
        'gold_kb_fulltext',
        'has_golden_solution'
    ]
    
    # Nur vorhandene Spalten auswählen
    available_columns = [col for col in expected_columns if col in df.columns]
    df_clean = df[available_columns].copy()
    
    # Leere Spalten entfernen
    df_clean = df_clean.dropna(axis=1, how='all')
    
    # NaN-Werte in den wichtigsten String-Spalten durch leere Strings ersetzen
    string_columns = ['ticket_title', 'ticket_description', 'gold_kb_fulltext']
    for col in string_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('')
    
    print(f"✓ {len(df_clean)} Testcases erfolgreich geladen")
    print(f"Spalten: {list(df_clean.columns)}")
    
    return df_clean


def get_testcase_by_id(df: pd.DataFrame, test_case_id: str) -> pd.Series | None:
    """
    Holt einen spezifischen Testcase anhand der ID.
    
    Args:
        df: DataFrame mit den Testcases
        test_case_id: ID des gewünschten Testcases (z.B. 'TC-P-01')
    
    Returns:
        pd.Series: Der Testcase als Series oder None falls nicht gefunden
    """
    result = df[df['test_case_id'] == test_case_id]
    if len(result) == 0:
        print(f"⚠ Testcase {test_case_id} nicht gefunden")
        return None
    return result.iloc[0]


def filter_by_difficulty(df: pd.DataFrame, difficulty: str) -> pd.DataFrame:
    """
    Filtert Testcases nach Schwierigkeitsgrad.
    
    Args:
        df: DataFrame mit den Testcases
        difficulty: Schwierigkeitsgrad ('low', 'mid', 'high')
    
    Returns:
        pd.DataFrame: Gefilterter DataFrame
    """
    if 'difficulty_level' not in df.columns:
        print("⚠ Spalte 'difficulty_level' nicht vorhanden")
        return df
    
    filtered = df[df['difficulty_level'].str.lower() == difficulty.lower()]
    print(f"✓ {len(filtered)} Testcases mit Schwierigkeitsgrad '{difficulty}' gefunden")
    return filtered


def filter_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Filtert Testcases nach Kategorie.
    
    Args:
        df: DataFrame mit den Testcases
        category: Kategorie (z.B. 'Hardware', 'Access', 'Cloud')
    
    Returns:
        pd.DataFrame: Gefilterter DataFrame
    """
    if 'category' not in df.columns:
        print("⚠ Spalte 'category' nicht vorhanden")
        return df
    
    filtered = df[df['category'].str.lower() == category.lower()]
    print(f"✓ {len(filtered)} Testcases in Kategorie '{category}' gefunden")
    return filtered


def main():
    """Beispiel-Nutzung des Scripts."""
    print("=== Testcases Loader ===\n")
    
    # Testcases laden
    df = load_testcases()
    print(f"\nAnzahl Testcases: {len(df)}")
    print(f"\nSpalten: {list(df.columns)}\n")
    
    # Statistiken anzeigen
    if 'difficulty_level' in df.columns:
        print("Verteilung nach Schwierigkeitsgrad:")
        print(df['difficulty_level'].value_counts())
        print()
    
    if 'category' in df.columns:
        print("Verteilung nach Kategorie:")
        print(df['category'].value_counts())
        print()
    
    # Beispiel: Einen spezifischen Testcase abrufen
    print("=== Beispiel: Testcase TC-P-01 ===")
    testcase = get_testcase_by_id(df, 'TC-P-01')
    if testcase is not None:
        print(f"ID: {testcase['test_case_id']}")
        print(f"Titel: {testcase['ticket_title']}")
        print(f"Kategorie: {testcase['category']}")
        print(f"Service: {testcase['service']}")
        print(f"Schwierigkeit: {testcase['difficulty_level']}")
        print(f"Gold KB ID: {testcase['gold_kb_id']}")
    
    print("\n=== Beispiel: Filter nach Schwierigkeit 'low' ===")
    low_cases = filter_by_difficulty(df, 'low')
    print(f"Gefundene IDs: {low_cases['test_case_id'].tolist()[:5]}...")
    
    print("\n=== Beispiel: Filter nach Kategorie 'Hardware' ===")
    hardware_cases = filter_by_category(df, 'Hardware')
    print(f"Gefundene IDs: {hardware_cases['test_case_id'].tolist()}")


if __name__ == "__main__":
    main()
