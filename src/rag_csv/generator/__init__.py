"""
Datengenerierung: KBs und Ticketgenerierung, Evaluierung von Modellen.
"""

from .kb import KBGenerator
from .tickets import TicketGenerator

__all__ = [
    "KB_Generator",
    "TicketGenerator",
]
