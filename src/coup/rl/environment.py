"""PettingZoo environment wrapper for Coup game."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers

from ..game import CoupGame
from ..types import ActionType, Character, GamePhase
from ..player import CoupPlayer
from ..actions import ActionFactory


class CoupEnvironment(AECEnv):
    """Multi-agent environment for Coup using PettingZoo."""
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "coup_v0",
        "is_parallelizable": False,
        "render_fps": 1,
    }
    
    def __init__(self, num_players: int = 4, max_cycles: int = 500):
        """Initialize the Coup environment.
        
        Args:
            num_players: Number of players (2-6)
            max_cycles: Maximum number of cycles before game ends
        """
        super().__init__()
        
        if not 2 <= num_players <= 6:
            raise ValueError("Number of players must be between 2 and 6")
        
        self.num_players = num_players
        self.max_cycles = max_cycles
        
        # Create agent names
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents = self.possible_agents[:]
        
        # Initialize game components
        self.game = CoupGame()
        self.action_factory = ActionFactory()
        
        # Environment state
        self.agent_selector = agent_selector(self.agents)
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}
        self.cycle_count = 0
        
        # Action and observation spaces
        self._init_spaces()
        
        # Game state tracking
        self.challenge_phase = False
        self.block_phase = False
        self.pending_action = None
        self.pending_target = None
        self.action_history = []
        
        # Initialize info dicts
        self.infos = {agent: {} for agent in self.agents}
        
    def _init_spaces(self) -> None:
        """Initialize action and observation spaces."""
        # Action space: 
        # 0-6: Basic actions (Income, Foreign Aid, Coup, Tax, Assassinate, Exchange, Steal)
        # 7-12: Target selection for targeted actions (player 0-5)
        # 13: Challenge
        # 14: Block
        # 15: Pass (no action)
        self.action_spaces = {agent: spaces.Discrete(16) for agent in self.agents}
        
        # Observation space components:
        # - Own coins (0-50+)
        # - Own cards (5 binary flags for each character)
        # - Other players' coins (0-50+ each)
        # - Other players' card counts (0-2 each)
        # - Game phase (3 binary flags)
        # - Current player indicator (binary for each player)
        # - Recent actions history (last 10 actions)
        # - Challenge/block opportunities (binary flags)
        # - Bluff detection features (success rates, patterns)
        obs_size = (
            1 +  # own coins
            5 +  # own cards (binary for each character)
            (self.num_players - 1) * 2 +  # other players (coins + card count)
            3 +  # game phase
            self.num_players +  # current player indicator
            10 * 3 +  # action history (action_type, actor, target)
            2 +  # challenge/block opportunities
            self.num_players * 3  # bluff detection (challenge_success, block_success, bluff_rate)
        )
        
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=100, shape=(obs_size,), dtype=np.float32)
            for agent in self.agents
        }
        
        # State space for global state
        self.state_space = spaces.Box(
            low=0, high=100, 
            shape=(self.num_players * 10,), 
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Dict[str, np.ndarray]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset game
        self.game = CoupGame()
        for i in range(self.num_players):
            self.game.add_player(f"player_{i}")
        self.game.start_game()
        
        # Reset environment state
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Reset agent selector
        self.agent_selector.reinit(self.agents)
        self.agent_selection = self.agent_selector.next()
        
        # Reset tracking variables
        self.cycle_count = 0
        self.challenge_phase = False
        self.block_phase = False
        self.pending_action = None
        self.pending_target = None
        self.action_history = []
        
        # Return initial observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        return observations
    
    def step(self, action: int) -> None:
        """Execute one step of the environment."""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return
        
        # Process action
        self._process_action(self.agent_selection, action)
        
        # Update rewards
        self._update_rewards()
        
        # Check for game end
        self._check_termination()
        
        # Update cycle count
        self.cycle_count += 1
        if self.cycle_count >= self.max_cycles:
            self.truncations = {agent: True for agent in self.agents}
        
        # Select next agent
        if not all(self.terminations.values()) and not all(self.truncations.values()):
            self._select_next_agent()
    
    def _process_action(self, agent: str, action: int) -> None:
        """Process a single agent action."""
        player_idx = self.agent_name_mapping[agent]
        player = self.game.state.players[player_idx]
        
        # Skip if player is eliminated
        if player.is_eliminated:
            return
        
        if self.challenge_phase:
            self._handle_challenge_action(agent, action)
        elif self.block_phase:
            self._handle_block_action(agent, action)
        else:
            self._handle_main_action(agent, action)
    
    def _handle_main_action(self, agent: str, action: int) -> None:
        """Handle main game actions."""
        player_idx = self.agent_name_mapping[agent]
        
        # Only current player can take main actions
        if player_idx != self.game.state.current_player_index:
            return
        
        if action < 7:  # Basic actions
            action_type = list(ActionType)[action]
            target = None
            
            # Handle targeted actions
            if action_type in [ActionType.COUP, ActionType.ASSASSINATE, ActionType.STEAL]:
                # Need to wait for target selection
                self.pending_action = action_type
                return
            
            result = self.game.attempt_action(action_type, target)
            self._record_action(agent, action_type, target, result)
            
            # Check if challenge/block window opened
            if self.game.state.challenge_window_open:
                self.challenge_phase = True
            elif self.game.state.block_window_open:
                self.block_phase = True
            else:
                self.game.resolve_pending_action()
        
        elif 7 <= action <= 12:  # Target selection
            if self.pending_action:
                target_idx = action - 7
                if target_idx < len(self.game.state.players):
                    target = self.game.state.players[target_idx]
                    
                    result = self.game.attempt_action(self.pending_action, target)
                    self._record_action(agent, self.pending_action, target, result)
                    
                    if self.game.state.challenge_window_open:
                        self.challenge_phase = True
                    elif self.game.state.block_window_open:
                        self.block_phase = True
                    else:
                        self.game.resolve_pending_action()
                
                self.pending_action = None
    
    def _handle_challenge_action(self, agent: str, action: int) -> None:
        """Handle challenge phase actions."""
        player_idx = self.agent_name_mapping[agent]
        player = self.game.state.players[player_idx]
        
        if action == 13:  # Challenge
            result = self.game.challenge_action(player)
            self._record_action(agent, "challenge", None, result)
            self.challenge_phase = False
            
            if self.game.state.block_window_open:
                self.block_phase = True
            else:
                self.game.resolve_pending_action()
        
        elif action == 15:  # Pass
            # Check if all non-current players have passed
            self.challenge_phase = False
            if self.game.state.block_window_open:
                self.block_phase = True
            else:
                self.game.resolve_pending_action()
    
    def _handle_block_action(self, agent: str, action: int) -> None:
        """Handle block phase actions."""
        player_idx = self.agent_name_mapping[agent]
        player = self.game.state.players[player_idx]
        
        if action == 14:  # Block
            result = self.game.block_action(player)
            self._record_action(agent, "block", None, result)
            self.block_phase = False
        
        elif action == 15:  # Pass
            # Check if all potential blockers have passed
            self.block_phase = False
            self.game.resolve_pending_action()
    
    def _record_action(self, agent: str, action_type: Union[ActionType, str], target: Optional[CoupPlayer], result) -> None:
        """Record action in history."""
        target_name = target.name if target else None
        self.action_history.append({
            'agent': agent,
            'action': action_type,
            'target': target_name,
            'success': result.success if hasattr(result, 'success') else True,
            'cycle': self.cycle_count
        })
        
        # Keep only last 10 actions
        if len(self.action_history) > 10:
            self.action_history.pop(0)
    
    def _get_observation(self, agent: str) -> np.ndarray:
        """Get observation for a specific agent."""
        player_idx = self.agent_name_mapping[agent]
        player = self.game.state.players[player_idx]
        
        obs = []
        
        # Own coins
        obs.append(player.coins / 50.0)  # Normalize
        
        # Own cards (binary encoding)
        for character in Character:
            obs.append(1.0 if player.has_character(character) else 0.0)
        
        # Other players' information
        for i, other_player in enumerate(self.game.state.players):
            if i != player_idx:
                obs.append(other_player.coins / 50.0)
                obs.append(len(other_player.cards) / 2.0)
        
        # Game phase
        obs.extend([
            1.0 if self.game.state.phase == GamePhase.WAITING_FOR_PLAYERS else 0.0,
            1.0 if self.game.state.phase == GamePhase.ACTIVE else 0.0,
            1.0 if self.game.state.phase == GamePhase.FINISHED else 0.0
        ])
        
        # Current player indicator
        for i in range(self.num_players):
            obs.append(1.0 if i == self.game.state.current_player_index else 0.0)
        
        # Action history (last 10 actions)
        for i in range(10):
            if i < len(self.action_history):
                action_data = self.action_history[-(i+1)]
                obs.extend([
                    hash(str(action_data['action'])) % 100 / 100.0,  # Action type
                    self.agent_name_mapping.get(action_data['agent'], 0) / self.num_players,  # Actor
                    self.agent_name_mapping.get(action_data['target'], 0) / self.num_players if action_data['target'] else 0.0  # Target
                ])
            else:
                obs.extend([0.0, 0.0, 0.0])
        
        # Challenge/block opportunities
        obs.extend([
            1.0 if self.challenge_phase else 0.0,
            1.0 if self.block_phase else 0.0
        ])
        
        # Bluff detection features (placeholder - would need more sophisticated tracking)
        for i in range(self.num_players):
            obs.extend([0.5, 0.5, 0.5])  # challenge_success_rate, block_success_rate, estimated_bluff_rate
        
        return np.array(obs, dtype=np.float32)
    
    def _select_next_agent(self) -> None:
        """Select the next agent to act."""
        if self.challenge_phase or self.block_phase:
            # In challenge/block phases, cycle through all non-current players
            current_idx = self.game.state.current_player_index
            next_idx = (self.agent_name_mapping[self.agent_selection] + 1) % self.num_players
            
            # Skip current player and eliminated players
            while (next_idx == current_idx or 
                   self.game.state.players[next_idx].is_eliminated):
                next_idx = (next_idx + 1) % self.num_players
                if next_idx == current_idx:  # Full cycle
                    break
            
            self.agent_selection = self.agents[next_idx]
        else:
            # Normal play - select current player
            current_idx = self.game.state.current_player_index
            self.agent_selection = self.agents[current_idx]
    
    def _update_rewards(self) -> None:
        """Update rewards based on current game state."""
        for agent in self.agents:
            player_idx = self.agent_name_mapping[agent]
            player = self.game.state.players[player_idx]
            
            # Reset rewards
            self.rewards[agent] = 0
            
            # Reward for staying alive
            if not player.is_eliminated:
                self.rewards[agent] += 0.1
            
            # Reward for having cards
            self.rewards[agent] += len(player.cards) * 0.5
            
            # Reward for having coins
            self.rewards[agent] += player.coins * 0.01
            
            # Big reward for winning
            if self.game.state.is_game_over() and not player.is_eliminated:
                self.rewards[agent] += 100
    
    def _check_termination(self) -> None:
        """Check if the game should terminate."""
        if self.game.state.is_game_over():
            for agent in self.agents:
                self.terminations[agent] = True
        
        # Terminate eliminated players
        for agent in self.agents:
            player_idx = self.agent_name_mapping[agent]
            player = self.game.state.players[player_idx]
            if player.is_eliminated:
                self.terminations[agent] = True
    
    def observe(self, agent: str) -> np.ndarray:
        """Get observation for agent."""
        return self._get_observation(agent)
    
    def state(self) -> np.ndarray:
        """Get global state."""
        state = []
        for player in self.game.state.players:
            state.extend([
                player.coins,
                len(player.cards),
                1 if player.is_eliminated else 0,
                1 if player.has_character(Character.DUKE) else 0,
                1 if player.has_character(Character.ASSASSIN) else 0,
                1 if player.has_character(Character.CAPTAIN) else 0,
                1 if player.has_character(Character.CONTESSA) else 0,
                1 if player.has_character(Character.AMBASSADOR) else 0,
                1 if player == self.game.state.get_current_player() else 0,
                len(self.action_history)
            ])
        
        return np.array(state[:self.state_space.shape[0]], dtype=np.float32)
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "human":
            print(f"\n=== Coup Game State (Cycle {self.cycle_count}) ===")
            print(f"Current agent: {self.agent_selection}")
            print(f"Challenge phase: {self.challenge_phase}")
            print(f"Block phase: {self.block_phase}")
            
            for i, player in enumerate(self.game.state.players):
                status = "ELIMINATED" if player.is_eliminated else "ACTIVE"
                current = " (CURRENT)" if i == self.game.state.current_player_index else ""
                print(f"Player {i}: {player.coins} coins, {len(player.cards)} cards - {status}{current}")
            
            if self.action_history:
                print(f"Last action: {self.action_history[-1]}")
        
        return None
    
    def close(self) -> None:
        """Close the environment."""
        pass