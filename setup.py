#!/usr/bin/env python3
"""
Setup-Datei für rag-csv Projekt.

Ermöglicht Installation mit:
  pip install -e .
  pip install -e ".[dev,bench]"
"""

import os
from textwrap import dedent

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


ENV_TEMPLATE = dedent(
  """
  # Logging
  LOG_LEVEL=INFO
  LOG_TO_CONSOLE=true
  LOG_TO_FILE=true
  LOG_PATH=output/logs
  LOG_FILE={name}.log

  # Qdrant
  QDRANT_URL=http://localhost:6333
  QDRANT_API_KEY=
  QDRANT_INC_COLLECTION=incidents
  QDRANT_KB_COLLECTION=knowledgebase

  # Embeddings (Ollama native /api/embed endpoint)
  EMBEDDING_URL=http://localhost:11434/api/embed
  EMBEDDING_MODEL=bge-m3
  EMBEDDING_DIM=1024

  # Ollama (LLM Server)
  OLLAMA_THREADS=8
  OLLAMA_URL=http://localhost:11434
  OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M

  # Data
  DATA_DIR=data
  INCIDENT_CSV=synthetic_incidents_with_kb.csv
  KB_CSV=kb_articles_llm.csv

  # Generator
  OUTPUT_DIR=output/generator
  OUTPUT_CSV_PATH=output/generator
  OUTPUT_CSV_FILENAME=generated_tickets.csv
  TOTAL_TICKETS=1200
  TICKETS_PER_CALL=5
  OLLAMA_MODEL_INCIDENTS=phi4-mini:latest
  GENERATOR_MODEL_INCIDENTS=phi4-mini:latest
  GENERATOR_TEMPERATURE=0.5
  GENERATOR_MAX_TOKENS=512
  GENERATOR_TOP_P=0.9
  GENERATOR_REPEAT_PENALTY=1.1
  GENERATOR_CTX_TOKENS=4096
  GENERATOR_SEED=-1
  GENERATOR_NUM_PREDICT=1500

  # Knowledge-Base Generator
  GENERATOR_MODEL_KNOWLEDGEBASE=llama3.2:3b
  GENERATOR_MODEL_KNOWLEDGEBASE_TEST=phi3:3.8b
  GENERATOR_TICKETS_FOR_KB_CONTEXT=10
  GENERATOR_KB_TEMPERATURE=0.5
  GENERATOR_KB_TOP_P=0.9
  GENERATOR_KB_REPEAT_PENALTY=1.1
  GENERATOR_KB_CTX_TOKENS=8192
  GENERATOR_KB_NUM_PREDICT=1500
  """
).lstrip()


def ensure_env_file() -> None:
  project_root = os.path.dirname(os.path.abspath(__file__))
  env_path = os.path.join(project_root, ".env")
  if os.path.exists(env_path):
    return
  with open(env_path, "w", encoding="utf-8") as f:
    f.write(ENV_TEMPLATE)


class InstallWithEnv(install):
  def run(self):
    super().run()
    ensure_env_file()


class DevelopWithEnv(develop):
  def run(self):
    super().run()
    ensure_env_file()

setup(
    name="rag-csv",
    version="1.0.0",
    description="Retrieval Augmented Generation System für CSV-basierte Knowledge Bases",
    author="Team RAG",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
  cmdclass={
    "install": InstallWithEnv,
    "develop": DevelopWithEnv,
  },
)
