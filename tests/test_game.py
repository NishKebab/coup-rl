"""Tests for the Coup game."""

import pytest
from src.coup.game import CoupGame
from src.coup.player import CoupPlayer
from src.coup.types import ActionType, Character
from src.coup.card import Card


class TestCoupGame:
    """Test cases for the main game logic."""
    
    def test_game_initialization(self):
        """Test game starts in correct state."""
        game = CoupGame()
        assert game.state.phase.value == "Waiting for Players"
        assert len(game.state.players) == 0
    
    def test_add_players(self):
        """Test adding players to game."""
        game = CoupGame()
        player1 = game.add_player("Alice")
        player2 = game.add_player("Bob")
        
        assert len(game.state.players) == 2
        assert player1.name == "Alice"
        assert player2.name == "Bob"
    
    def test_start_game(self):
        """Test starting a game."""
        game = CoupGame()
        game.add_player("Alice")
        game.add_player("Bob")
        
        game.start_game()
        
        assert game.state.phase.value == "Active"
        assert all(len(p.cards) == 2 for p in game.state.players)
        assert all(p.coins == 2 for p in game.state.players)
    
    def test_start_game_requires_minimum_players(self):
        """Test game requires at least 2 players."""
        game = CoupGame()
        game.add_player("Alice")
        
        with pytest.raises(ValueError):
            game.start_game()
    
    def test_income_action(self):
        """Test income action gives 1 coin."""
        game = CoupGame()
        game.add_player("Alice")
        game.add_player("Bob")
        game.start_game()
        
        player = game.state.get_current_player()
        initial_coins = player.coins
        
        result = game.attempt_action(ActionType.INCOME)
        
        assert result.success
        assert player.coins == initial_coins + 1
    
    def test_foreign_aid_action(self):
        """Test foreign aid action gives 2 coins."""
        game = CoupGame()
        game.add_player("Alice")
        game.add_player("Bob")
        game.start_game()
        
        player = game.state.get_current_player()
        initial_coins = player.coins
        
        result = game.attempt_action(ActionType.FOREIGN_AID)
        assert result.success
        
        # Foreign aid opens block window, so resolve it
        game.resolve_pending_action()
        
        assert player.coins == initial_coins + 2
    
    def test_coup_action(self):
        """Test coup action."""
        game = CoupGame()
        game.add_player("Alice")
        game.add_player("Bob")
        game.start_game()
        
        player = game.state.get_current_player()
        target = game.state.players[1] if game.state.players[0] == player else game.state.players[0]
        
        player.coins = 7
        
        result = game.attempt_action(ActionType.COUP, target)
        
        assert result.success
        assert player.coins == 0
    
    def test_player_turn_rotation(self):
        """Test that turns rotate between players."""
        game = CoupGame()
        game.add_player("Alice")
        game.add_player("Bob")
        game.start_game()
        
        first_player = game.state.get_current_player()
        game.attempt_action(ActionType.INCOME)
        second_player = game.state.get_current_player()
        
        assert first_player != second_player
    
    def test_game_state_export(self):
        """Test game state can be exported."""
        game = CoupGame()
        game.add_player("Alice")
        game.add_player("Bob")
        game.start_game()
        
        state = game.get_game_state()
        
        assert state['phase'] == 'Active'
        assert len(state['players']) == 2
        assert state['current_player'] in ['Alice', 'Bob']


class TestPlayer:
    """Test cases for player functionality."""
    
    def test_player_initialization(self):
        """Test player starts with correct values."""
        player = CoupPlayer("Alice")
        assert player.name == "Alice"
        assert player.coins == 2
        assert len(player.cards) == 0
        assert not player.is_eliminated
    
    def test_add_card(self):
        """Test adding cards to player."""
        player = CoupPlayer("Alice")
        card = Card(Character.DUKE)
        
        player.add_card(card)
        
        assert len(player.cards) == 1
        assert player.has_character(Character.DUKE)
    
    def test_lose_card(self):
        """Test losing a card."""
        player = CoupPlayer("Alice")
        card = Card(Character.DUKE)
        player.add_card(card)
        
        result = player.lose_card(Character.DUKE)
        
        assert result is True
        assert len(player.cards) == 0
        assert not player.has_character(Character.DUKE)
    
    def test_player_elimination(self):
        """Test player elimination when losing last card."""
        player = CoupPlayer("Alice")
        card = Card(Character.DUKE)
        player.add_card(card)
        
        player.lose_card(Character.DUKE)
        
        assert player.is_eliminated
    
    def test_coin_management(self):
        """Test coin adding and spending."""
        player = CoupPlayer("Alice")
        
        player.add_coins(5)
        assert player.coins == 7
        
        result = player.spend_coins(3)
        assert result is True
        assert player.coins == 4
        
        result = player.spend_coins(10)
        assert result is False
        assert player.coins == 4


class TestCardAbilities:
    """Test card abilities and deck functionality."""
    
    def test_card_creation(self):
        """Test card creation."""
        card = Card(Character.DUKE)
        assert card.character == Character.DUKE
        assert str(card) == "Duke"
    
    def test_deck_initialization(self):
        """Test deck has correct cards."""
        from src.coup.card import Deck
        deck = Deck()
        
        assert deck.size() == 15  # 3 of each character, 5 characters
        assert not deck.is_empty()
    
    def test_deck_operations(self):
        """Test deck draw and shuffle."""
        from src.coup.card import Deck
        deck = Deck()
        
        card = deck.draw()
        assert isinstance(card, Card)
        assert deck.size() == 14
        
        deck.add_card(card)
        assert deck.size() == 15