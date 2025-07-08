"""Coup package."""

__version__ = "0.1.0"

from .game import CoupGame
from .player import CoupPlayer
from .types import ActionType, Character, GamePhase
from .card import Card, Deck
from .cli import CoupCLI

__all__ = [
    "CoupGame",
    "CoupPlayer", 
    "ActionType",
    "Character",
    "GamePhase",
    "Card",
    "Deck",
    "CoupCLI"
]