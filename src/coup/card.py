"""Card implementation for Coup game."""

from dataclasses import dataclass
from typing import List, Set
from .types import Character, ActionType


@dataclass(frozen=True)
class Card:
    """Represents a character card in Coup."""
    character: Character
    
    def __str__(self) -> str:
        return self.character.value


class CardAbilities:
    """Defines what actions each character can perform or block."""
    
    _ABILITIES = {
        Character.DUKE: {
            'actions': {ActionType.TAX},
            'blocks': {ActionType.FOREIGN_AID}
        },
        Character.ASSASSIN: {
            'actions': {ActionType.ASSASSINATE},
            'blocks': set()
        },
        Character.CAPTAIN: {
            'actions': {ActionType.STEAL},
            'blocks': {ActionType.STEAL}
        },
        Character.CONTESSA: {
            'actions': set(),
            'blocks': {ActionType.ASSASSINATE}
        },
        Character.AMBASSADOR: {
            'actions': {ActionType.EXCHANGE},
            'blocks': {ActionType.STEAL}
        }
    }
    
    @classmethod
    def can_perform_action(cls, character: Character, action: ActionType) -> bool:
        """Check if a character can perform a specific action."""
        return action in cls._ABILITIES[character]['actions']
    
    @classmethod
    def can_block_action(cls, character: Character, action: ActionType) -> bool:
        """Check if a character can block a specific action."""
        return action in cls._ABILITIES[character]['blocks']
    
    @classmethod
    def get_actions(cls, character: Character) -> Set[ActionType]:
        """Get all actions a character can perform."""
        return cls._ABILITIES[character]['actions'].copy()
    
    @classmethod
    def get_blocks(cls, character: Character) -> Set[ActionType]:
        """Get all actions a character can block."""
        return cls._ABILITIES[character]['blocks'].copy()


class Deck:
    """Manages the deck of cards."""
    
    def __init__(self) -> None:
        self._cards: List[Card] = []
        self._create_deck()
    
    def _create_deck(self) -> None:
        """Create a standard Coup deck with 3 of each character."""
        for character in Character:
            for _ in range(3):
                self._cards.append(Card(character))
    
    def shuffle(self) -> None:
        """Shuffle the deck."""
        import random
        random.shuffle(self._cards)
    
    def draw(self) -> Card:
        """Draw a card from the deck."""
        if not self._cards:
            raise ValueError("Cannot draw from empty deck")
        return self._cards.pop()
    
    def add_card(self, card: Card) -> None:
        """Add a card back to the deck."""
        self._cards.append(card)
    
    def is_empty(self) -> bool:
        """Check if deck is empty."""
        return len(self._cards) == 0
    
    def size(self) -> int:
        """Get number of cards in deck."""
        return len(self._cards)