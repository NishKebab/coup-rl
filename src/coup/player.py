"""Player implementation for Coup game."""

from typing import List, Optional
from .card import Card
from .types import Character


class CoupPlayer:
    """Represents a player in the Coup game."""
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.coins = 2
        self.cards: List[Card] = []
        self.is_eliminated = False
    
    def add_card(self, card: Card) -> None:
        """Add a card to the player's hand."""
        if len(self.cards) >= 2:
            raise ValueError("Player cannot have more than 2 cards")
        self.cards.append(card)
    
    def has_character(self, character: Character) -> bool:
        """Check if player has a specific character."""
        return any(card.character == character for card in self.cards)
    
    def lose_card(self, character: Character) -> bool:
        """Remove a card from player's hand. Returns True if card was found and removed."""
        for i, card in enumerate(self.cards):
            if card.character == character:
                self.cards.pop(i)
                if len(self.cards) == 0:
                    self.is_eliminated = True
                return True
        return False
    
    def lose_card_by_choice(self, card_index: int) -> Optional[Card]:
        """Remove a card by index, allowing player to choose which card to lose."""
        if 0 <= card_index < len(self.cards):
            card = self.cards.pop(card_index)
            if len(self.cards) == 0:
                self.is_eliminated = True
            return card
        return None
    
    def add_coins(self, amount: int) -> None:
        """Add coins to player's total."""
        self.coins += amount
    
    def spend_coins(self, amount: int) -> bool:
        """Spend coins. Returns True if player had enough coins."""
        if self.coins >= amount:
            self.coins -= amount
            return True
        return False
    
    def can_afford_coup(self) -> bool:
        """Check if player can afford to coup (7 coins)."""
        return self.coins >= 7
    
    def must_coup(self) -> bool:
        """Check if player must coup (10+ coins)."""
        return self.coins >= 10
    
    def get_card_count(self) -> int:
        """Get number of cards player has."""
        return len(self.cards)
    
    def get_characters(self) -> List[Character]:
        """Get list of characters player has."""
        return [card.character for card in self.cards]
    
    def __str__(self) -> str:
        return f"{self.name} ({self.coins} coins, {len(self.cards)} cards)"
    
    def __repr__(self) -> str:
        return f"CoupPlayer(name='{self.name}', coins={self.coins}, cards={len(self.cards)})"


class PlayerActions:
    """Handles player action validation and effects."""
    
    def __init__(self, player: CoupPlayer) -> None:
        self.player = player
    
    def can_take_income(self) -> bool:
        """Check if player can take income (always available)."""
        return not self.player.is_eliminated
    
    def can_take_foreign_aid(self) -> bool:
        """Check if player can take foreign aid (always available unless blocked)."""
        return not self.player.is_eliminated
    
    def can_coup(self) -> bool:
        """Check if player can coup (needs 7 coins)."""
        return not self.player.is_eliminated and self.player.can_afford_coup()
    
    def can_tax(self) -> bool:
        """Check if player can tax (needs Duke)."""
        return not self.player.is_eliminated and self.player.has_character(Character.DUKE)
    
    def can_assassinate(self) -> bool:
        """Check if player can assassinate (needs Assassin and 3 coins)."""
        return (not self.player.is_eliminated and 
                self.player.has_character(Character.ASSASSIN) and 
                self.player.coins >= 3)
    
    def can_steal(self) -> bool:
        """Check if player can steal (needs Captain)."""
        return not self.player.is_eliminated and self.player.has_character(Character.CAPTAIN)
    
    def can_exchange(self) -> bool:
        """Check if player can exchange (needs Ambassador)."""
        return not self.player.is_eliminated and self.player.has_character(Character.AMBASSADOR)
    
    def execute_income(self) -> None:
        """Execute income action (gain 1 coin)."""
        self.player.add_coins(1)
    
    def execute_foreign_aid(self) -> None:
        """Execute foreign aid action (gain 2 coins)."""
        self.player.add_coins(2)
    
    def execute_coup(self) -> bool:
        """Execute coup action (spend 7 coins). Returns True if successful."""
        return self.player.spend_coins(7)
    
    def execute_tax(self) -> None:
        """Execute tax action (gain 3 coins)."""
        self.player.add_coins(3)
    
    def execute_assassinate(self) -> bool:
        """Execute assassinate action (spend 3 coins). Returns True if successful."""
        return self.player.spend_coins(3)
    
    def execute_steal(self, target: CoupPlayer) -> int:
        """Execute steal action. Returns amount stolen."""
        amount = min(2, target.coins)
        if target.spend_coins(amount):
            self.player.add_coins(amount)
            return amount
        return 0