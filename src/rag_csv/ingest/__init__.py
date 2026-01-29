"""
Ingestion von Daten in Qdrant-Vektordatenbank.
"""

# Stub-Funktionen für API-Kompatibilität
def ingest_incidents():
    """Ingest incident data."""
    from . import incidents
    incidents.main()

def ingest_kb():
    """Ingest KB data."""
    from . import kb
    kb.main()

def setup_collections():
    """Setup Qdrant collections."""
    from . import setup
    setup.main()

__all__ = [
    "ingest_incidents",
    "ingest_kb",
    "setup_collections",
]

__all__ = [
    "ingest_incidents",
    "ingest_kb",
    "setup_collections",
]
