#!/usr/bin/env python3
"""
Setup-Datei für rag-csv Projekt.

Ermöglicht Installation mit:
  pip install -e .
  pip install -e ".[dev,bench]"
"""

from setuptools import setup, find_packages

setup(
    name="rag-csv",
    version="1.0.0",
    description="Retrieval Augmented Generation System für CSV-basierte Knowledge Bases",
    author="Team RAG",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
)
