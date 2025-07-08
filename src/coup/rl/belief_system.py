"""Belief system for opponent modeling in Coup."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from ..types import Character, ActionType


class PlayerBelief:
    """Tracks beliefs about a single opponent."""
    
    def __init__(self, player_id: str):
        self.player_id = player_id
        
        # Character probabilities (belief that player has each character)
        self.character_probs = {char: 0.2 for char in Character}  # Start uniform
        
        # Behavioral patterns
        self.action_history = deque(maxlen=100)
        self.bluff_history = deque(maxlen=50)
        self.challenge_history = deque(maxlen=50)
        self.block_history = deque(maxlen=50)
        
        # Bluffing tendencies
        self.bluff_frequency = 0.0
        self.bluff_success_rate = 0.0
        self.challenge_accuracy = 0.0
        self.block_accuracy = 0.0
        
        # Situational patterns
        self.action_patterns = defaultdict(lambda: defaultdict(int))  # context -> action -> count
        self.risk_tolerance = 0.5  # 0 = very cautious, 1 = very aggressive
        
        # Coin management patterns
        self.coin_thresholds = {
            'coup_threshold': 7,
            'save_threshold': 5,
            'aggressive_threshold': 3
        }
    
    def update_action(self, action: ActionType, context: Dict[str, Any], success: bool) -> None:
        """Update beliefs based on observed action."""
        self.action_history.append({
            'action': action,
            'context': context,
            'success': success,
            'timestamp': len(self.action_history)
        })
        
        # Update character probabilities based on action
        self._update_character_beliefs(action, success)
        
        # Update behavioral patterns
        self._update_behavioral_patterns(action, context, success)
    
    def _update_character_beliefs(self, action: ActionType, success: bool) -> None:
        """Update character probability beliefs."""
        # Character-specific actions
        char_actions = {
            ActionType.TAX: Character.DUKE,
            ActionType.ASSASSINATE: Character.ASSASSIN,
            ActionType.STEAL: Character.CAPTAIN,
            ActionType.EXCHANGE: Character.AMBASSADOR
        }
        
        if action in char_actions:
            required_char = char_actions[action]
            
            if success:
                # If action succeeded without challenge, increase probability
                self.character_probs[required_char] = min(0.95, self.character_probs[required_char] + 0.3)
            else:
                # If action failed due to challenge, decrease probability
                self.character_probs[required_char] = max(0.05, self.character_probs[required_char] - 0.4)
        
        # Normalize probabilities
        total_prob = sum(self.character_probs.values())
        if total_prob > 0:
            for char in self.character_probs:
                self.character_probs[char] /= total_prob
    
    def _update_behavioral_patterns(self, action: ActionType, context: Dict[str, Any], success: bool) -> None:
        """Update behavioral pattern beliefs."""
        # Context-based action patterns
        situation = self._get_situation_key(context)
        self.action_patterns[situation][action] += 1
        
        # Risk tolerance update
        if action in [ActionType.COUP, ActionType.ASSASSINATE]:
            self.risk_tolerance = min(1.0, self.risk_tolerance + 0.1)
        elif action in [ActionType.INCOME, ActionType.FOREIGN_AID]:
            self.risk_tolerance = max(0.0, self.risk_tolerance - 0.05)
    
    def _get_situation_key(self, context: Dict[str, Any]) -> str:
        """Generate situation key for pattern matching."""
        coins = context.get('coins', 0)
        cards = context.get('cards', 0)
        opponents = context.get('opponents', 0)
        
        coin_level = 'low' if coins < 3 else 'medium' if coins < 7 else 'high'
        card_level = 'single' if cards == 1 else 'double'
        opponent_level = 'few' if opponents <= 2 else 'many'
        
        return f"{coin_level}_{card_level}_{opponent_level}"
    
    def update_bluff(self, was_bluffing: bool, was_challenged: bool, challenge_success: bool) -> None:
        """Update bluffing-related beliefs."""
        self.bluff_history.append({
            'was_bluffing': was_bluffing,
            'was_challenged': was_challenged,
            'challenge_success': challenge_success
        })
        
        # Recalculate bluff statistics
        if self.bluff_history:
            bluff_count = sum(1 for h in self.bluff_history if h['was_bluffing'])
            self.bluff_frequency = bluff_count / len(self.bluff_history)
            
            if bluff_count > 0:
                successful_bluffs = sum(1 for h in self.bluff_history 
                                      if h['was_bluffing'] and not h['challenge_success'])
                self.bluff_success_rate = successful_bluffs / bluff_count
    
    def update_challenge(self, challenged_action: ActionType, was_correct: bool) -> None:
        """Update challenge-related beliefs."""
        self.challenge_history.append({
            'action': challenged_action,
            'correct': was_correct
        })
        
        # Recalculate challenge accuracy
        if self.challenge_history:
            correct_challenges = sum(1 for h in self.challenge_history if h['correct'])
            self.challenge_accuracy = correct_challenges / len(self.challenge_history)
    
    def get_character_probability(self, character: Character) -> float:
        """Get probability that player has a specific character."""
        return self.character_probs[character]
    
    def get_bluff_probability(self, action: ActionType) -> float:
        """Get probability that player is bluffing for a specific action."""
        if action not in [ActionType.TAX, ActionType.ASSASSINATE, ActionType.STEAL, ActionType.EXCHANGE]:
            return 0.0
        
        char_actions = {
            ActionType.TAX: Character.DUKE,
            ActionType.ASSASSINATE: Character.ASSASSIN,
            ActionType.STEAL: Character.CAPTAIN,
            ActionType.EXCHANGE: Character.AMBASSADOR
        }
        
        if action in char_actions:
            has_char_prob = self.character_probs[char_actions[action]]
            return max(0.0, 1.0 - has_char_prob) * self.bluff_frequency
        
        return 0.0
    
    def predict_action(self, context: Dict[str, Any]) -> Dict[ActionType, float]:
        """Predict action probabilities based on context."""
        situation = self._get_situation_key(context)
        action_counts = self.action_patterns[situation]
        
        if not action_counts:
            # No data for this situation, return uniform
            return {action: 1.0 / len(ActionType) for action in ActionType}
        
        total = sum(action_counts.values())
        return {action: count / total for action, count in action_counts.items()}


class BeliefSystem:
    """Complete belief system for opponent modeling."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.player_beliefs = {}
        self.game_state_history = deque(maxlen=1000)
        
        # Meta-learning parameters
        self.meta_patterns = defaultdict(lambda: defaultdict(int))
        self.game_phase_patterns = defaultdict(lambda: defaultdict(int))
        
        # Equilibrium detection
        self.strategy_convergence = defaultdict(list)
        self.exploitability_estimates = defaultdict(float)
    
    def add_opponent(self, player_id: str) -> None:
        """Add a new opponent to track."""
        if player_id not in self.player_beliefs:
            self.player_beliefs[player_id] = PlayerBelief(player_id)
    
    def update_observation(self, observation: np.ndarray, game_state: Dict[str, Any]) -> None:
        """Update beliefs based on new observation."""
        # Store game state
        self.game_state_history.append({
            'observation': observation.copy(),
            'game_state': game_state.copy(),
            'timestamp': len(self.game_state_history)
        })
        
        # Update meta-patterns
        self._update_meta_patterns(game_state)
    
    def update_opponent_action(self, opponent_id: str, action: ActionType, 
                             context: Dict[str, Any], success: bool) -> None:
        """Update beliefs about opponent's action."""
        if opponent_id not in self.player_beliefs:
            self.add_opponent(opponent_id)
        
        self.player_beliefs[opponent_id].update_action(action, context, success)
    
    def update_bluff_result(self, opponent_id: str, was_bluffing: bool, 
                          was_challenged: bool, challenge_success: bool) -> None:
        """Update beliefs about opponent's bluffing."""
        if opponent_id in self.player_beliefs:
            self.player_beliefs[opponent_id].update_bluff(was_bluffing, was_challenged, challenge_success)
    
    def update_challenge_result(self, opponent_id: str, challenged_action: ActionType, 
                              was_correct: bool) -> None:
        """Update beliefs about opponent's challenge accuracy."""
        if opponent_id in self.player_beliefs:
            self.player_beliefs[opponent_id].update_challenge(challenged_action, was_correct)
    
    def _update_meta_patterns(self, game_state: Dict[str, Any]) -> None:
        """Update meta-level patterns about game flow."""
        phase = game_state.get('phase', 'unknown')
        current_player = game_state.get('current_player', 'unknown')
        
        # Track phase transitions
        if len(self.game_state_history) > 1:
            prev_state = self.game_state_history[-2]['game_state']
            prev_phase = prev_state.get('phase', 'unknown')
            
            if phase != prev_phase:
                self.game_phase_patterns[prev_phase][phase] += 1
    
    def should_challenge(self, opponent_id: str, action: ActionType, 
                        context: Dict[str, Any]) -> float:
        """Get confidence level for challenging opponent's action."""
        if opponent_id not in self.player_beliefs:
            return 0.3  # Default moderate confidence
        
        belief = self.player_beliefs[opponent_id]
        
        # Get bluff probability
        bluff_prob = belief.get_bluff_probability(action)
        
        # Adjust based on opponent's challenge accuracy (they might be baiting)
        challenge_accuracy = belief.challenge_accuracy
        bait_adjustment = (1 - challenge_accuracy) * 0.2  # Reduce confidence if they're often wrong
        
        # Adjust based on game context
        context_adjustment = self._get_context_adjustment(context)
        
        confidence = bluff_prob - bait_adjustment + context_adjustment
        return max(0.0, min(1.0, confidence))
    
    def should_block(self, opponent_id: str, action: ActionType, 
                    context: Dict[str, Any]) -> float:
        """Get confidence level for blocking opponent's action."""
        if opponent_id not in self.player_beliefs:
            return 0.3  # Default moderate confidence
        
        belief = self.player_beliefs[opponent_id]
        
        # For blocking, we want to know if we should claim to have the blocking character
        # This is essentially a bluffing decision on our part
        
        # Consider opponent's likelihood to challenge our block
        challenge_likelihood = belief.challenge_accuracy * 0.8  # Slight discount
        
        # Consider the value of blocking (depends on action being blocked)
        action_value = self._get_action_value(action, context)
        
        # Block confidence based on risk-reward
        block_confidence = action_value * (1 - challenge_likelihood)
        
        return max(0.0, min(1.0, block_confidence))
    
    def _get_context_adjustment(self, context: Dict[str, Any]) -> float:
        """Get context-based adjustment for challenge/block decisions."""
        coins = context.get('coins', 0)
        cards = context.get('cards', 0)
        opponents_left = context.get('opponents', 0)
        
        adjustment = 0.0
        
        # Late game: more aggressive
        if opponents_left <= 2:
            adjustment += 0.2
        
        # Low cards: more defensive
        if cards == 1:
            adjustment -= 0.1
        
        # High coins: more confident
        if coins >= 7:
            adjustment += 0.1
        
        return adjustment
    
    def _get_action_value(self, action: ActionType, context: Dict[str, Any]) -> float:
        """Get the value of preventing an action."""
        # High value actions to block
        if action == ActionType.ASSASSINATE:
            return 0.8
        elif action == ActionType.STEAL:
            return 0.6
        elif action == ActionType.TAX:
            return 0.4
        elif action == ActionType.FOREIGN_AID:
            return 0.3
        
        return 0.2
    
    def get_challenge_confidence(self) -> float:
        """Get general confidence in making challenges."""
        if not self.player_beliefs:
            return 0.5
        
        # Average challenge accuracy across all opponents
        accuracies = [belief.challenge_accuracy for belief in self.player_beliefs.values()]
        avg_accuracy = np.mean(accuracies) if accuracies else 0.5
        
        return avg_accuracy
    
    def get_opponent_risk_profile(self, opponent_id: str) -> Dict[str, float]:
        """Get risk profile for opponent."""
        if opponent_id not in self.player_beliefs:
            return {'risk_tolerance': 0.5, 'bluff_frequency': 0.2, 'challenge_accuracy': 0.5}
        
        belief = self.player_beliefs[opponent_id]
        return {
            'risk_tolerance': belief.risk_tolerance,
            'bluff_frequency': belief.bluff_frequency,
            'challenge_accuracy': belief.challenge_accuracy
        }
    
    def predict_opponent_action(self, opponent_id: str, context: Dict[str, Any]) -> Dict[ActionType, float]:
        """Predict opponent's likely actions."""
        if opponent_id not in self.player_beliefs:
            return {action: 1.0 / len(ActionType) for action in ActionType}
        
        return self.player_beliefs[opponent_id].predict_action(context)
    
    def get_exploitability_estimate(self, opponent_id: str) -> float:
        """Get estimate of how exploitable an opponent is."""
        if opponent_id not in self.player_beliefs:
            return 0.5
        
        belief = self.player_beliefs[opponent_id]
        
        # Higher exploitability if:
        # - Very predictable patterns
        # - Poor challenge accuracy
        # - Extreme risk tolerance
        
        pattern_predictability = self._calculate_pattern_predictability(belief)
        challenge_weakness = 1 - belief.challenge_accuracy
        risk_extremity = abs(belief.risk_tolerance - 0.5) * 2
        
        exploitability = (pattern_predictability + challenge_weakness + risk_extremity) / 3
        
        self.exploitability_estimates[opponent_id] = exploitability
        return exploitability
    
    def _calculate_pattern_predictability(self, belief: PlayerBelief) -> float:
        """Calculate how predictable an opponent's patterns are."""
        if not belief.action_patterns:
            return 0.5
        
        # Calculate entropy of action patterns
        predictability_scores = []
        
        for situation, actions in belief.action_patterns.items():
            if not actions:
                continue
            
            total = sum(actions.values())
            probs = [count / total for count in actions.values()]
            
            # Calculate entropy
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            max_entropy = np.log2(len(actions))
            
            # Predictability = 1 - normalized_entropy
            predictability = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
            predictability_scores.append(predictability)
        
        return np.mean(predictability_scores) if predictability_scores else 0.5