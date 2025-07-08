"""Type definitions for Coup game."""

from enum import Enum
from typing import Protocol, List, Optional


class Character(Enum):
    """Character types in Coup."""
    DUKE = "Duke"
    ASSASSIN = "Assassin"
    CAPTAIN = "Captain"
    CONTESSA = "Contessa"
    AMBASSADOR = "Ambassador"


class ActionType(Enum):
    """Types of actions available in the game."""
    INCOME = "Income"
    FOREIGN_AID = "Foreign Aid"
    COUP = "Coup"
    TAX = "Tax"
    ASSASSINATE = "Assassinate"
    EXCHANGE = "Exchange"
    STEAL = "Steal"


class GamePhase(Enum):
    """Game phases."""
    WAITING_FOR_PLAYERS = "Waiting for Players"
    ACTIVE = "Active"
    FINISHED = "Finished"


class Player(Protocol):
    """Player interface."""
    name: str
    coins: int
    cards: List['Card']
    is_eliminated: bool
    
    def has_character(self, character: Character) -> bool:
        """Check if player has a specific character."""
        ...
    
    def lose_card(self, character: Character) -> bool:
        """Remove a card from player's hand."""
        ...


class GameObserver(Protocol):
    """Observer interface for game events."""
    
    def on_action_taken(self, player: str, action: ActionType, target: Optional[str] = None) -> None:
        """Called when an action is taken."""
        ...
    
    def on_challenge_made(self, challenger: str, challenged: str) -> None:
        """Called when a challenge is made."""
        ...
    
    def on_player_eliminated(self, player: str) -> None:
        """Called when a player is eliminated."""
        ...