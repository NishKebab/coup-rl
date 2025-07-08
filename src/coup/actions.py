"""Action system for Coup game."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Set
from .types import ActionType, Character
from .player import CoupPlayer
from .card import Card, CardAbilities


@dataclass
class ActionResult:
    """Result of an action attempt."""
    success: bool
    message: str
    coins_gained: int = 0
    coins_lost: int = 0
    cards_lost: List[Card] = None
    
    def __post_init__(self) -> None:
        if self.cards_lost is None:
            self.cards_lost = []


class Action(ABC):
    """Base class for all game actions."""
    
    def __init__(self, action_type: ActionType) -> None:
        self.action_type = action_type
    
    @abstractmethod
    def can_be_performed_by(self, player: CoupPlayer) -> bool:
        """Check if the action can be performed by the given player."""
        pass
    
    @abstractmethod
    def requires_character(self) -> Optional[Character]:
        """Return the character required to perform this action, if any."""
        pass
    
    @abstractmethod
    def can_be_blocked(self) -> bool:
        """Check if this action can be blocked."""
        pass
    
    @abstractmethod
    def get_blocking_characters(self) -> Set[Character]:
        """Get characters that can block this action."""
        pass
    
    @abstractmethod
    def execute(self, player: CoupPlayer, target: Optional[CoupPlayer] = None) -> ActionResult:
        """Execute the action."""
        pass


class IncomeAction(Action):
    """Income action - gain 1 coin."""
    
    def __init__(self) -> None:
        super().__init__(ActionType.INCOME)
    
    def can_be_performed_by(self, player: CoupPlayer) -> bool:
        return not player.is_eliminated
    
    def requires_character(self) -> Optional[Character]:
        return None
    
    def can_be_blocked(self) -> bool:
        return False
    
    def get_blocking_characters(self) -> Set[Character]:
        return set()
    
    def execute(self, player: CoupPlayer, target: Optional[CoupPlayer] = None) -> ActionResult:
        player.add_coins(1)
        return ActionResult(True, f"{player.name} gained 1 coin from income", coins_gained=1)


class ForeignAidAction(Action):
    """Foreign Aid action - gain 2 coins."""
    
    def __init__(self) -> None:
        super().__init__(ActionType.FOREIGN_AID)
    
    def can_be_performed_by(self, player: CoupPlayer) -> bool:
        return not player.is_eliminated
    
    def requires_character(self) -> Optional[Character]:
        return None
    
    def can_be_blocked(self) -> bool:
        return True
    
    def get_blocking_characters(self) -> Set[Character]:
        return {Character.DUKE}
    
    def execute(self, player: CoupPlayer, target: Optional[CoupPlayer] = None) -> ActionResult:
        player.add_coins(2)
        return ActionResult(True, f"{player.name} gained 2 coins from foreign aid", coins_gained=2)


class CoupAction(Action):
    """Coup action - spend 7 coins to eliminate target's card."""
    
    def __init__(self) -> None:
        super().__init__(ActionType.COUP)
    
    def can_be_performed_by(self, player: CoupPlayer) -> bool:
        return not player.is_eliminated and player.can_afford_coup()
    
    def requires_character(self) -> Optional[Character]:
        return None
    
    def can_be_blocked(self) -> bool:
        return False
    
    def get_blocking_characters(self) -> Set[Character]:
        return set()
    
    def execute(self, player: CoupPlayer, target: Optional[CoupPlayer] = None) -> ActionResult:
        if target is None:
            return ActionResult(False, "Coup requires a target")
        
        if not player.spend_coins(7):
            return ActionResult(False, "Not enough coins to coup")
        
        return ActionResult(True, f"{player.name} couped {target.name}", coins_lost=7)


class TaxAction(Action):
    """Tax action - gain 3 coins (requires Duke)."""
    
    def __init__(self) -> None:
        super().__init__(ActionType.TAX)
    
    def can_be_performed_by(self, player: CoupPlayer) -> bool:
        return not player.is_eliminated and player.has_character(Character.DUKE)
    
    def requires_character(self) -> Optional[Character]:
        return Character.DUKE
    
    def can_be_blocked(self) -> bool:
        return False
    
    def get_blocking_characters(self) -> Set[Character]:
        return set()
    
    def execute(self, player: CoupPlayer, target: Optional[CoupPlayer] = None) -> ActionResult:
        player.add_coins(3)
        return ActionResult(True, f"{player.name} gained 3 coins from tax", coins_gained=3)


class AssassinateAction(Action):
    """Assassinate action - spend 3 coins to eliminate target's card (requires Assassin)."""
    
    def __init__(self) -> None:
        super().__init__(ActionType.ASSASSINATE)
    
    def can_be_performed_by(self, player: CoupPlayer) -> bool:
        return (not player.is_eliminated and 
                player.has_character(Character.ASSASSIN) and 
                player.coins >= 3)
    
    def requires_character(self) -> Optional[Character]:
        return Character.ASSASSIN
    
    def can_be_blocked(self) -> bool:
        return True
    
    def get_blocking_characters(self) -> Set[Character]:
        return {Character.CONTESSA}
    
    def execute(self, player: CoupPlayer, target: Optional[CoupPlayer] = None) -> ActionResult:
        if target is None:
            return ActionResult(False, "Assassinate requires a target")
        
        if not player.spend_coins(3):
            return ActionResult(False, "Not enough coins to assassinate")
        
        return ActionResult(True, f"{player.name} assassinated {target.name}", coins_lost=3)


class StealAction(Action):
    """Steal action - take up to 2 coins from target (requires Captain)."""
    
    def __init__(self) -> None:
        super().__init__(ActionType.STEAL)
    
    def can_be_performed_by(self, player: CoupPlayer) -> bool:
        return not player.is_eliminated and player.has_character(Character.CAPTAIN)
    
    def requires_character(self) -> Optional[Character]:
        return Character.CAPTAIN
    
    def can_be_blocked(self) -> bool:
        return True
    
    def get_blocking_characters(self) -> Set[Character]:
        return {Character.CAPTAIN, Character.AMBASSADOR}
    
    def execute(self, player: CoupPlayer, target: Optional[CoupPlayer] = None) -> ActionResult:
        if target is None:
            return ActionResult(False, "Steal requires a target")
        
        amount = min(2, target.coins)
        if target.spend_coins(amount):
            player.add_coins(amount)
            return ActionResult(True, f"{player.name} stole {amount} coins from {target.name}", coins_gained=amount)
        
        return ActionResult(False, "Target has no coins to steal")


class ExchangeAction(Action):
    """Exchange action - look at deck and exchange cards (requires Ambassador)."""
    
    def __init__(self) -> None:
        super().__init__(ActionType.EXCHANGE)
    
    def can_be_performed_by(self, player: CoupPlayer) -> bool:
        return not player.is_eliminated and player.has_character(Character.AMBASSADOR)
    
    def requires_character(self) -> Optional[Character]:
        return Character.AMBASSADOR
    
    def can_be_blocked(self) -> bool:
        return False
    
    def get_blocking_characters(self) -> Set[Character]:
        return set()
    
    def execute(self, player: CoupPlayer, target: Optional[CoupPlayer] = None) -> ActionResult:
        return ActionResult(True, f"{player.name} exchanged cards with the deck")


class ActionFactory:
    """Factory for creating action instances."""
    
    _ACTION_CLASSES = {
        ActionType.INCOME: IncomeAction,
        ActionType.FOREIGN_AID: ForeignAidAction,
        ActionType.COUP: CoupAction,
        ActionType.TAX: TaxAction,
        ActionType.ASSASSINATE: AssassinateAction,
        ActionType.STEAL: StealAction,
        ActionType.EXCHANGE: ExchangeAction,
    }
    
    @classmethod
    def create_action(cls, action_type: ActionType) -> Action:
        """Create an action instance of the specified type."""
        if action_type not in cls._ACTION_CLASSES:
            raise ValueError(f"Unknown action type: {action_type}")
        
        return cls._ACTION_CLASSES[action_type]()
    
    @classmethod
    def get_available_actions(cls, player: CoupPlayer) -> List[ActionType]:
        """Get all actions available to a player."""
        available = []
        
        for action_type in ActionType:
            action = cls.create_action(action_type)
            if action.can_be_performed_by(player):
                available.append(action_type)
        
        return available