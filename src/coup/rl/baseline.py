"""Baseline strategies for evaluating Coup RL agents."""

import numpy as np
import random
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from ..types import ActionType, Character
from ..game import CoupGame


class BaselineAgent(ABC):
    """Abstract base class for baseline agents."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.action_counts = {i: 0 for i in range(16)}
    
    @abstractmethod
    def act(self, observation: np.ndarray, game_state: Dict[str, Any], training: bool = False) -> int:
        """Select action based on observation."""
        pass
    
    def reset_hidden_state(self) -> None:
        """Reset any internal state (compatibility with RL agents)."""
        pass
    
    def store_experience(self, reward: float, next_observation: np.ndarray, done: bool) -> None:
        """Store experience (compatibility with RL agents)."""
        pass


class RandomAgent(BaselineAgent):
    """Random action selection agent."""
    
    def act(self, observation: np.ndarray, game_state: Dict[str, Any], training: bool = False) -> int:
        """Select random valid action."""
        valid_actions = self._get_valid_actions(observation, game_state)
        action = random.choice(valid_actions)
        self.action_counts[action] += 1
        return action
    
    def _get_valid_actions(self, observation: np.ndarray, game_state: Dict[str, Any]) -> List[int]:
        """Get valid actions based on game state."""
        valid_actions = [15]  # Always allow pass
        
        if game_state.get('challenge_phase', False):
            valid_actions.append(13)  # Challenge
        elif game_state.get('block_phase', False):
            valid_actions.append(14)  # Block
        else:
            # Main actions
            valid_actions.extend([0, 1])  # Income, Foreign Aid
            
            # Coup if enough coins
            coins = observation[0] * 50  # Denormalize
            if coins >= 7:
                valid_actions.append(2)  # Coup
                valid_actions.extend(range(7, 13))  # Target selection
            
            # Character actions
            valid_actions.extend([3, 4, 5, 6])  # Tax, Assassinate, Exchange, Steal
        
        return valid_actions


class HonestAgent(BaselineAgent):
    """Agent that never bluffs - only uses actions for characters it actually has."""
    
    def act(self, observation: np.ndarray, game_state: Dict[str, Any], training: bool = False) -> int:
        """Select action honestly based on actual cards."""
        valid_actions = self._get_honest_actions(observation, game_state)
        
        # Prefer character actions if available
        character_actions = [3, 4, 5, 6]  # Tax, Assassinate, Exchange, Steal
        honest_char_actions = [a for a in character_actions if a in valid_actions]
        
        if honest_char_actions and random.random() < 0.7:
            action = random.choice(honest_char_actions)
        else:
            action = random.choice(valid_actions)
        
        self.action_counts[action] += 1
        return action
    
    def _get_honest_actions(self, observation: np.ndarray, game_state: Dict[str, Any]) -> List[int]:
        """Get actions that don't require bluffing."""
        valid_actions = [15]  # Pass
        
        if game_state.get('challenge_phase', False):
            # Never challenge (honest agents assume others are honest)
            return valid_actions
        elif game_state.get('block_phase', False):
            # Only block if we actually have the blocking character
            own_cards = observation[1:6]  # Character flags
            
            # Check if we can honestly block based on cards
            if own_cards[0] == 1:  # Duke - can block Foreign Aid
                valid_actions.append(14)
            if own_cards[2] == 1 or own_cards[4] == 1:  # Captain or Ambassador - can block Steal
                valid_actions.append(14)
            if own_cards[3] == 1:  # Contessa - can block Assassinate
                valid_actions.append(14)
            
            return valid_actions
        else:
            # Main actions
            valid_actions.extend([0, 1])  # Income, Foreign Aid (always available)
            
            coins = observation[0] * 50  # Denormalize
            if coins >= 7:
                valid_actions.append(2)  # Coup
                valid_actions.extend(range(7, 13))  # Target selection
            
            # Only use character actions if we actually have the character
            own_cards = observation[1:6]  # Character flags
            
            if own_cards[0] == 1:  # Duke
                valid_actions.append(3)  # Tax
            if own_cards[1] == 1 and coins >= 3:  # Assassin
                valid_actions.append(4)  # Assassinate
                valid_actions.extend(range(7, 13))  # Target selection
            if own_cards[4] == 1:  # Ambassador
                valid_actions.append(5)  # Exchange
            if own_cards[2] == 1:  # Captain
                valid_actions.append(6)  # Steal
                valid_actions.extend(range(7, 13))  # Target selection
            
            return valid_actions


class AggressiveAgent(BaselineAgent):
    """Agent that plays aggressively - frequently uses character actions and challenges."""
    
    def act(self, observation: np.ndarray, game_state: Dict[str, Any], training: bool = False) -> int:
        """Select aggressive actions."""
        valid_actions = self._get_valid_actions(observation, game_state)
        
        if game_state.get('challenge_phase', False):
            # Challenge frequently
            if random.random() < 0.6:
                action = 13  # Challenge
            else:
                action = 15  # Pass
        elif game_state.get('block_phase', False):
            # Block frequently (often bluffing)
            if random.random() < 0.5:
                action = 14  # Block
            else:
                action = 15  # Pass
        else:
            coins = observation[0] * 50
            
            # Prioritize aggressive actions
            if coins >= 7:
                # Prefer Coup or Assassinate
                if 2 in valid_actions and random.random() < 0.4:
                    action = 2  # Coup
                elif 4 in valid_actions and random.random() < 0.3:
                    action = 4  # Assassinate
                else:
                    action = random.choice(valid_actions)
            else:
                # Prefer character actions (including bluffs)
                character_actions = [a for a in [3, 4, 5, 6] if a in valid_actions]
                if character_actions and random.random() < 0.7:
                    action = random.choice(character_actions)
                else:
                    action = random.choice(valid_actions)
        
        # Handle target selection for targeted actions
        if action in [2, 4, 6] and action not in [7, 8, 9, 10, 11, 12]:
            # Need to select target
            target_actions = [a for a in range(7, 13) if a in valid_actions]
            if target_actions:
                action = random.choice(target_actions)
        
        self.action_counts[action] += 1
        return action
    
    def _get_valid_actions(self, observation: np.ndarray, game_state: Dict[str, Any]) -> List[int]:
        """Get all valid actions (including potential bluffs)."""
        valid_actions = [15]  # Pass
        
        if game_state.get('challenge_phase', False):
            valid_actions.append(13)  # Challenge
        elif game_state.get('block_phase', False):
            valid_actions.append(14)  # Block
        else:
            # All main actions
            valid_actions.extend([0, 1, 3, 4, 5, 6])  # Income, Foreign Aid, Tax, Assassinate, Exchange, Steal
            
            coins = observation[0] * 50
            if coins >= 7:
                valid_actions.append(2)  # Coup
            if coins >= 3:
                valid_actions.append(4)  # Assassinate (if not already added)
            
            # Target selection
            valid_actions.extend(range(7, 13))
        
        return valid_actions


class DefensiveAgent(BaselineAgent):
    """Agent that plays defensively - avoids risky actions and challenges conservatively."""
    
    def act(self, observation: np.ndarray, game_state: Dict[str, Any], training: bool = False) -> int:
        """Select defensive actions."""
        valid_actions = self._get_valid_actions(observation, game_state)
        
        if game_state.get('challenge_phase', False):
            # Challenge rarely (only when very confident)
            if random.random() < 0.2:
                action = 13  # Challenge
            else:
                action = 15  # Pass
        elif game_state.get('block_phase', False):
            # Block when it makes sense defensively
            if random.random() < 0.7:  # Often block to protect self
                action = 14  # Block
            else:
                action = 15  # Pass
        else:
            coins = observation[0] * 50
            cards = sum(observation[1:6])  # Number of cards (approximation)
            
            # Prefer safe actions
            if coins >= 10:
                # Must coup
                action = 2 if 2 in valid_actions else random.choice(valid_actions)
            elif coins >= 7 and cards == 1:
                # Vulnerable position - prefer coup
                action = 2 if 2 in valid_actions else 0  # Coup or Income
            else:
                # Prefer Income and Foreign Aid
                safe_actions = [a for a in [0, 1] if a in valid_actions]  # Income, Foreign Aid
                if safe_actions and random.random() < 0.6:
                    action = random.choice(safe_actions)
                else:
                    # Occasionally use character actions (conservatively)
                    char_actions = [a for a in [3, 5] if a in valid_actions]  # Tax, Exchange (safer)
                    if char_actions and random.random() < 0.3:
                        action = random.choice(char_actions)
                    else:
                        action = random.choice(valid_actions)
        
        # Handle target selection for targeted actions
        if action in [2, 4, 6] and action not in range(7, 13):
            target_actions = [a for a in range(7, 13) if a in valid_actions]
            if target_actions:
                action = random.choice(target_actions)
        
        self.action_counts[action] += 1
        return action
    
    def _get_valid_actions(self, observation: np.ndarray, game_state: Dict[str, Any]) -> List[int]:
        """Get valid actions with defensive bias."""
        valid_actions = [15]  # Pass
        
        if game_state.get('challenge_phase', False):
            valid_actions.append(13)  # Challenge
        elif game_state.get('block_phase', False):
            valid_actions.append(14)  # Block
        else:
            # Conservative main actions
            valid_actions.extend([0, 1])  # Income, Foreign Aid
            
            coins = observation[0] * 50
            if coins >= 7:
                valid_actions.append(2)  # Coup
                valid_actions.extend(range(7, 13))  # Target selection
            
            # Limited character actions
            valid_actions.extend([3, 5])  # Tax, Exchange (safer actions)
            
            if coins >= 3:
                valid_actions.append(4)  # Assassinate
            valid_actions.append(6)  # Steal
        
        return valid_actions


class AlwaysChallengeAgent(BaselineAgent):
    """Agent that always challenges character actions."""
    
    def act(self, observation: np.ndarray, game_state: Dict[str, Any], training: bool = False) -> int:
        """Always challenge when possible."""
        if game_state.get('challenge_phase', False):
            action = 13  # Challenge
        elif game_state.get('block_phase', False):
            action = 15  # Pass (don't block much)
        else:
            # Play normally otherwise
            valid_actions = self._get_basic_actions(observation, game_state)
            action = random.choice(valid_actions)
        
        self.action_counts[action] += 1
        return action
    
    def _get_basic_actions(self, observation: np.ndarray, game_state: Dict[str, Any]) -> List[int]:
        """Get basic non-character actions."""
        valid_actions = [0, 1]  # Income, Foreign Aid
        
        coins = observation[0] * 50
        if coins >= 7:
            valid_actions.append(2)  # Coup
            valid_actions.extend(range(7, 13))  # Target selection
        
        return valid_actions


class BaselineStrategies:
    """Factory for creating baseline strategy agents."""
    
    def __init__(self):
        self.strategies = {
            'random': RandomAgent,
            'honest': HonestAgent,
            'aggressive': AggressiveAgent,
            'defensive': DefensiveAgent,
            'always_challenge': AlwaysChallengeAgent
        }
    
    def create_baseline_agent(self, strategy: str, agent_id: str) -> BaselineAgent:
        """Create a baseline agent with the specified strategy."""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategies.keys())}")
        
        return self.strategies[strategy](agent_id)
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available baseline strategies."""
        return list(self.strategies.keys())
    
    def evaluate_strategy_against_baseline(self, agent, baseline_strategy: str, 
                                         num_games: int = 100) -> Dict[str, float]:
        """Evaluate an agent against a specific baseline strategy."""
        baseline_agent = self.create_baseline_agent(baseline_strategy, "baseline")
        
        # This would require running games - simplified for now
        return {
            'win_rate': 0.5,  # Placeholder
            'avg_game_length': 20.0,
            'avg_reward': 0.0
        }


class AdaptiveBaselineAgent(BaselineAgent):
    """Baseline agent that adapts its strategy based on opponent behavior."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.opponent_profiles = {}
        self.game_history = []
        self.adaptation_rate = 0.1
        
        # Strategy weights
        self.strategy_weights = {
            'aggressive': 0.25,
            'defensive': 0.25,
            'honest': 0.25,
            'random': 0.25
        }
        
        # Sub-agents for different strategies
        self.sub_agents = {
            'aggressive': AggressiveAgent(f"{agent_id}_aggressive"),
            'defensive': DefensiveAgent(f"{agent_id}_defensive"),
            'honest': HonestAgent(f"{agent_id}_honest"),
            'random': RandomAgent(f"{agent_id}_random")
        }
    
    def act(self, observation: np.ndarray, game_state: Dict[str, Any], training: bool = False) -> int:
        """Select action based on adaptive strategy."""
        # Select strategy based on current weights
        strategy = np.random.choice(
            list(self.strategy_weights.keys()),
            p=list(self.strategy_weights.values())
        )
        
        # Use selected sub-agent
        action = self.sub_agents[strategy].act(observation, game_state, training)
        
        # Record action and strategy used
        self.game_history.append({
            'observation': observation.copy(),
            'action': action,
            'strategy': strategy,
            'game_state': game_state.copy()
        })
        
        self.action_counts[action] += 1
        return action
    
    def update_strategy_weights(self, game_result: Dict[str, Any]) -> None:
        """Update strategy weights based on game outcome."""
        won = game_result.get('won', False)
        
        # Count strategy usage in this game
        strategy_usage = {strategy: 0 for strategy in self.strategy_weights}
        for action_record in self.game_history:
            strategy_usage[action_record['strategy']] += 1
        
        # Update weights based on performance
        total_actions = sum(strategy_usage.values())
        if total_actions > 0:
            for strategy, count in strategy_usage.items():
                contribution = count / total_actions
                
                if won:
                    # Increase weight for strategies used in winning game
                    self.strategy_weights[strategy] += self.adaptation_rate * contribution
                else:
                    # Decrease weight for strategies used in losing game
                    self.strategy_weights[strategy] -= self.adaptation_rate * contribution * 0.5
                
                # Keep weights non-negative
                self.strategy_weights[strategy] = max(0.01, self.strategy_weights[strategy])
        
        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        for strategy in self.strategy_weights:
            self.strategy_weights[strategy] /= total_weight
        
        # Clear game history
        self.game_history.clear()
    
    def get_strategy_distribution(self) -> Dict[str, float]:
        """Get current strategy weight distribution."""
        return self.strategy_weights.copy()