"""Main game logic for Coup."""

from typing import List, Optional, Dict, Set
from .types import GamePhase, ActionType, Character
from .player import CoupPlayer
from .card import Deck, Card
from .actions import Action, ActionFactory, ActionResult


class GameState:
    """Manages the current state of the game."""
    
    def __init__(self) -> None:
        self.phase = GamePhase.WAITING_FOR_PLAYERS
        self.current_player_index = 0
        self.players: List[CoupPlayer] = []
        self.deck = Deck()
        self.action_log: List[str] = []
        self.pending_action: Optional[Action] = None
        self.pending_target: Optional[CoupPlayer] = None
        self.challenge_window_open = False
        self.block_window_open = False
        
    def add_player(self, player: CoupPlayer) -> None:
        """Add a player to the game."""
        if self.phase != GamePhase.WAITING_FOR_PLAYERS:
            raise ValueError("Cannot add players after game has started")
        
        if len(self.players) >= 6:
            raise ValueError("Maximum 6 players allowed")
        
        self.players.append(player)
    
    def get_current_player(self) -> CoupPlayer:
        """Get the current player."""
        return self.players[self.current_player_index]
    
    def next_player(self) -> None:
        """Move to the next player."""
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        while self.players[self.current_player_index].is_eliminated:
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
    
    def get_active_players(self) -> List[CoupPlayer]:
        """Get all non-eliminated players."""
        return [p for p in self.players if not p.is_eliminated]
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return len(self.get_active_players()) <= 1
    
    def get_winner(self) -> Optional[CoupPlayer]:
        """Get the winner if game is over."""
        active = self.get_active_players()
        return active[0] if len(active) == 1 else None
    
    def log_action(self, message: str) -> None:
        """Log an action to the game log."""
        self.action_log.append(message)


class ChallengeSystem:
    """Handles challenge mechanics."""
    
    def __init__(self, game_state: GameState) -> None:
        self.game_state = game_state
    
    def can_challenge(self, action: Action, challenger: CoupPlayer) -> bool:
        """Check if a player can challenge an action."""
        required_character = action.requires_character()
        return (required_character is not None and 
                not challenger.is_eliminated and 
                challenger != self.game_state.get_current_player())
    
    def resolve_challenge(self, action: Action, actor: CoupPlayer, challenger: CoupPlayer) -> bool:
        """Resolve a challenge. Returns True if challenge succeeds."""
        required_character = action.requires_character()
        
        if required_character is None:
            return False
        
        if actor.has_character(required_character):
            challenger.lose_card_by_choice(0)
            self._shuffle_and_redraw(actor, required_character)
            self.game_state.log_action(f"{challenger.name} lost challenge against {actor.name}")
            return False
        else:
            actor.lose_card_by_choice(0)
            self.game_state.log_action(f"{challenger.name} successfully challenged {actor.name}")
            return True
    
    def _shuffle_and_redraw(self, player: CoupPlayer, character: Character) -> None:
        """Shuffle character back into deck and redraw."""
        for i, card in enumerate(player.cards):
            if card.character == character:
                old_card = player.cards.pop(i)
                self.game_state.deck.add_card(old_card)
                self.game_state.deck.shuffle()
                new_card = self.game_state.deck.draw()
                player.add_card(new_card)
                break


class BlockSystem:
    """Handles blocking mechanics."""
    
    def __init__(self, game_state: GameState) -> None:
        self.game_state = game_state
    
    def can_block(self, action: Action, blocker: CoupPlayer, target: Optional[CoupPlayer] = None) -> bool:
        """Check if a player can block an action."""
        if not action.can_be_blocked() or blocker.is_eliminated:
            return False
        
        blocking_characters = action.get_blocking_characters()
        
        if action.action_type == ActionType.FOREIGN_AID:
            return any(blocker.has_character(char) for char in blocking_characters)
        
        if target and blocker == target:
            return any(blocker.has_character(char) for char in blocking_characters)
        
        return False
    
    def get_potential_blockers(self, action: Action, target: Optional[CoupPlayer] = None) -> List[CoupPlayer]:
        """Get all players who could potentially block an action."""
        blockers = []
        
        for player in self.game_state.get_active_players():
            if self.can_block(action, player, target):
                blockers.append(player)
        
        return blockers


class CoupGame:
    """Main game controller for Coup."""
    
    def __init__(self) -> None:
        self.state = GameState()
        self.challenge_system = ChallengeSystem(self.state)
        self.block_system = BlockSystem(self.state)
        self.action_factory = ActionFactory()
    
    def add_player(self, name: str) -> CoupPlayer:
        """Add a player to the game."""
        player = CoupPlayer(name)
        self.state.add_player(player)
        return player
    
    def start_game(self) -> None:
        """Start the game."""
        if len(self.state.players) < 2:
            raise ValueError("Need at least 2 players to start")
        
        self.state.deck.shuffle()
        
        for player in self.state.players:
            for _ in range(2):
                card = self.state.deck.draw()
                player.add_card(card)
        
        self.state.phase = GamePhase.ACTIVE
        self.state.log_action("Game started")
    
    def get_available_actions(self, player: CoupPlayer) -> List[ActionType]:
        """Get available actions for a player."""
        if player != self.state.get_current_player():
            return []
        
        return self.action_factory.get_available_actions(player)
    
    def attempt_action(self, action_type: ActionType, target: Optional[CoupPlayer] = None) -> ActionResult:
        """Attempt to perform an action."""
        current_player = self.state.get_current_player()
        action = self.action_factory.create_action(action_type)
        
        if not action.can_be_performed_by(current_player):
            return ActionResult(False, "Action not available to current player")
        
        if action.requires_character() and not current_player.has_character(action.requires_character()):
            return ActionResult(False, "Player does not have required character")
        
        if target and target.is_eliminated:
            return ActionResult(False, "Cannot target eliminated player")
        
        if current_player.must_coup() and action_type != ActionType.COUP:
            return ActionResult(False, "Must coup when having 10+ coins")
        
        self.state.pending_action = action
        self.state.pending_target = target
        
        if action.requires_character():
            self.state.challenge_window_open = True
            return ActionResult(True, f"Action declared, waiting for challenges")
        
        if action.can_be_blocked():
            self.state.block_window_open = True
            return ActionResult(True, f"Action declared, waiting for blocks")
        
        return self._execute_action(action, current_player, target)
    
    def challenge_action(self, challenger: CoupPlayer) -> ActionResult:
        """Challenge the pending action."""
        if not self.state.challenge_window_open or not self.state.pending_action:
            return ActionResult(False, "No action to challenge")
        
        if not self.challenge_system.can_challenge(self.state.pending_action, challenger):
            return ActionResult(False, "Cannot challenge this action")
        
        current_player = self.state.get_current_player()
        challenge_succeeded = self.challenge_system.resolve_challenge(
            self.state.pending_action, current_player, challenger
        )
        
        self.state.challenge_window_open = False
        
        if challenge_succeeded:
            self.state.pending_action = None
            self.state.pending_target = None
            self.state.next_player()
            return ActionResult(True, f"Challenge succeeded, {current_player.name} loses turn")
        else:
            if self.state.pending_action.can_be_blocked():
                self.state.block_window_open = True
                return ActionResult(True, f"Challenge failed, waiting for blocks")
            else:
                return self._execute_pending_action()
    
    def block_action(self, blocker: CoupPlayer) -> ActionResult:
        """Block the pending action."""
        if not self.state.block_window_open or not self.state.pending_action:
            return ActionResult(False, "No action to block")
        
        if not self.block_system.can_block(self.state.pending_action, blocker, self.state.pending_target):
            return ActionResult(False, "Cannot block this action")
        
        self.state.block_window_open = False
        self.state.pending_action = None
        self.state.pending_target = None
        self.state.next_player()
        
        return ActionResult(True, f"{blocker.name} blocked the action")
    
    def resolve_pending_action(self) -> ActionResult:
        """Resolve the pending action if no challenges or blocks."""
        if self.state.challenge_window_open or self.state.block_window_open:
            self.state.challenge_window_open = False
            self.state.block_window_open = False
        
        return self._execute_pending_action()
    
    def _execute_pending_action(self) -> ActionResult:
        """Execute the pending action."""
        if not self.state.pending_action:
            return ActionResult(False, "No pending action")
        
        current_player = self.state.get_current_player()
        result = self._execute_action(self.state.pending_action, current_player, self.state.pending_target)
        
        self.state.pending_action = None
        self.state.pending_target = None
        
        return result
    
    def _execute_action(self, action: Action, player: CoupPlayer, target: Optional[CoupPlayer] = None) -> ActionResult:
        """Execute an action immediately."""
        result = action.execute(player, target)
        
        if result.success:
            self.state.log_action(result.message)
            
            if action.action_type in [ActionType.COUP, ActionType.ASSASSINATE] and target:
                self._handle_card_loss(target)
            
            if not self.state.is_game_over():
                self.state.next_player()
            else:
                self.state.phase = GamePhase.FINISHED
        
        return result
    
    def _handle_card_loss(self, player: CoupPlayer) -> None:
        """Handle a player losing a card."""
        if player.get_card_count() > 0:
            player.lose_card_by_choice(0)
            self.state.log_action(f"{player.name} lost a card")
    
    def exchange_cards(self, player: CoupPlayer, cards_to_keep: List[int]) -> ActionResult:
        """Handle exchange action card selection."""
        if len(cards_to_keep) != player.get_card_count():
            return ActionResult(False, "Must keep same number of cards")
        
        cards_drawn = []
        for _ in range(2):
            if not self.state.deck.is_empty():
                cards_drawn.append(self.state.deck.draw())
        
        all_cards = player.cards + cards_drawn
        
        kept_cards = [all_cards[i] for i in cards_to_keep]
        returned_cards = [card for i, card in enumerate(all_cards) if i not in cards_to_keep]
        
        player.cards = kept_cards
        
        for card in returned_cards:
            self.state.deck.add_card(card)
        
        self.state.deck.shuffle()
        
        return ActionResult(True, f"{player.name} exchanged cards")
    
    def get_game_state(self) -> Dict:
        """Get current game state as dictionary."""
        return {
            'phase': self.state.phase.value,
            'current_player': self.state.get_current_player().name if self.state.phase == GamePhase.ACTIVE else None,
            'players': [
                {
                    'name': p.name,
                    'coins': p.coins,
                    'cards': len(p.cards),
                    'eliminated': p.is_eliminated
                }
                for p in self.state.players
            ],
            'winner': self.state.get_winner().name if self.state.get_winner() else None,
            'action_log': self.state.action_log[-10:]  # Last 10 actions
        }