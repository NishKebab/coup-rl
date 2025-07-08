"""PPO Agent with LSTM for Coup game."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any
import random
from collections import deque, namedtuple

from .belief_system import BeliefSystem


Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done', 
    'log_prob', 'value', 'advantage', 'hidden_state'
])


class CoupPolicyNetwork(nn.Module):
    """Policy network with LSTM for memory and bluffing."""
    
    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 256, lstm_size: int = 128):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lstm_size = lstm_size
        
        # Input processing
        self.input_layer = nn.Linear(obs_size, hidden_size)
        
        # LSTM for memory and sequential decision making
        self.lstm = nn.LSTM(hidden_size, lstm_size, batch_first=True)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(lstm_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Bluffing head (probability of bluffing)
        self.bluff_head = nn.Sequential(
            nn.Linear(lstm_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network."""
        batch_size = x.size(0)
        
        # Process input
        x = F.relu(self.input_layer(x))
        
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # LSTM forward pass
        if hidden is None:
            h0 = torch.zeros(1, batch_size, self.lstm_size, device=x.device)
            c0 = torch.zeros(1, batch_size, self.lstm_size, device=x.device)
            hidden = (h0, c0)
        
        lstm_out, new_hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.squeeze(1)  # Remove sequence dimension
        
        # Policy, value, and bluff outputs
        policy_logits = self.policy_head(lstm_out)
        value = self.value_head(lstm_out)
        bluff_prob = self.bluff_head(lstm_out)
        
        return policy_logits, value, bluff_prob, new_hidden
    
    def get_action(self, state: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                   valid_actions: Optional[List[int]] = None) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get action from policy."""
        policy_logits, value, bluff_prob, new_hidden = self.forward(state, hidden)
        
        # Mask invalid actions
        if valid_actions is not None:
            mask = torch.full_like(policy_logits, -float('inf'))
            mask[0, valid_actions] = 0
            policy_logits = policy_logits + mask
        
        # Sample action
        probs = F.softmax(policy_logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value, bluff_prob, new_hidden


class CoupPPOAgent:
    """PPO agent for Coup with LSTM memory and bluffing capabilities."""
    
    def __init__(self, obs_size: int, action_size: int, agent_id: str, 
                 hidden_size: int = 256, lstm_size: int = 128,
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.5, entropy_coef: float = 0.01,
                 bluff_coef: float = 0.1):
        
        self.agent_id = agent_id
        self.obs_size = obs_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.bluff_coef = bluff_coef
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network
        self.policy_net = CoupPolicyNetwork(obs_size, action_size, hidden_size, lstm_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.buffer = deque(maxlen=10000)
        
        # LSTM hidden state
        self.hidden_state = None
        
        # Belief system for opponent modeling
        self.belief_system = BeliefSystem(agent_id)
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.bluff_attempts = []
        self.bluff_successes = []
        
        # Action statistics
        self.action_counts = {i: 0 for i in range(action_size)}
        self.challenge_success_rate = 0.0
        self.block_success_rate = 0.0
        
    def reset_hidden_state(self) -> None:
        """Reset LSTM hidden state."""
        self.hidden_state = None
    
    def get_valid_actions(self, observation: np.ndarray, game_state: Dict) -> List[int]:
        """Get valid actions based on game state."""
        valid_actions = []
        
        # Always allow pass action
        valid_actions.append(15)
        
        # Check game phase
        if game_state.get('challenge_phase', False):
            valid_actions.append(13)  # Challenge
        elif game_state.get('block_phase', False):
            valid_actions.append(14)  # Block
        else:
            # Main actions
            valid_actions.extend([0, 1])  # Income, Foreign Aid
            
            # Coup (if enough coins)
            if observation[0] * 50 >= 7:  # Denormalize coins
                valid_actions.append(2)
                valid_actions.extend(range(7, 13))  # Target selection
            
            # Character-specific actions (would need to check if bluffing)
            valid_actions.extend([3, 4, 5, 6])  # Tax, Assassinate, Exchange, Steal
        
        return valid_actions
    
    def act(self, observation: np.ndarray, game_state: Dict, training: bool = True) -> int:
        """Select action based on observation."""
        state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Get valid actions
        valid_actions = self.get_valid_actions(observation, game_state)
        
        with torch.no_grad():
            action, log_prob, value, bluff_prob, self.hidden_state = self.policy_net.get_action(
                state_tensor, self.hidden_state, valid_actions
            )
        
        # Update belief system
        self.belief_system.update_observation(observation, game_state)
        
        # Decide whether to bluff
        if training and bluff_prob.item() > 0.5:
            self.bluff_attempts.append(1)
            # Modify action based on bluffing strategy
            action = self._apply_bluff_strategy(action, valid_actions, observation)
        else:
            self.bluff_attempts.append(0)
        
        # Track action statistics
        self.action_counts[action] += 1
        
        # Store experience if training
        if training:
            self.buffer.append({
                'state': observation,
                'action': action,
                'log_prob': log_prob.item(),
                'value': value.item(),
                'bluff_prob': bluff_prob.item(),
                'hidden_state': self.hidden_state
            })
        
        return action
    
    def _apply_bluff_strategy(self, action: int, valid_actions: List[int], observation: np.ndarray) -> int:
        """Apply bluffing strategy to action selection."""
        # If attempting a character action without the character, it's a bluff
        character_actions = [3, 4, 5, 6]  # Tax, Assassinate, Exchange, Steal
        
        if action in character_actions:
            # Check if we actually have the required character
            own_cards = observation[1:6]  # Character flags
            required_chars = {3: 0, 4: 1, 5: 4, 6: 2}  # Action to character mapping
            
            if action in required_chars:
                char_idx = required_chars[action]
                if own_cards[char_idx] == 0:  # Don't have the character
                    # This is a bluff - proceed with the action
                    return action
        
        # For challenge/block decisions, use opponent modeling
        if action in [13, 14]:  # Challenge or Block
            belief_confidence = self.belief_system.get_challenge_confidence()
            if belief_confidence > 0.7:  # High confidence in challenge
                return action
        
        return action
    
    def store_experience(self, reward: float, next_observation: np.ndarray, done: bool) -> None:
        """Store experience in buffer."""
        if self.buffer:
            experience = self.buffer[-1]
            experience['reward'] = reward
            experience['next_state'] = next_observation
            experience['done'] = done
    
    def update_policy(self, batch_size: int = 64, epochs: int = 10) -> Dict[str, float]:
        """Update policy using PPO."""
        if len(self.buffer) < batch_size:
            return {}
        
        # Convert buffer to tensors
        experiences = list(self.buffer)
        states = torch.FloatTensor(np.array([exp['state'] for exp in experiences])).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in experiences]).to(self.device)
        old_log_probs = torch.FloatTensor([exp['log_prob'] for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in experiences]).to(self.device)
        values = torch.FloatTensor([exp['value'] for exp in experiences]).to(self.device)
        dones = torch.FloatTensor([float(exp['done']) for exp in experiences]).to(self.device)
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        bluff_loss = 0
        
        for _ in range(epochs):
            # Sample batch
            batch_indices = torch.randperm(len(experiences))[:batch_size]
            
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]
            
            # Forward pass
            policy_logits, new_values, bluff_probs, _ = self.policy_net(batch_states)
            
            # Policy loss
            probs = F.softmax(policy_logits, dim=-1)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(batch_actions)
            
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
            policy_loss_batch = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss_batch = F.mse_loss(new_values.squeeze(), batch_returns)
            
            # Entropy loss
            entropy_loss_batch = -dist.entropy().mean()
            
            # Bluff loss (encourage strategic bluffing)
            bluff_target = torch.FloatTensor([exp['bluff_prob'] for exp in experiences])[batch_indices].to(self.device)
            bluff_loss_batch = F.mse_loss(bluff_probs.squeeze(), bluff_target)
            
            # Total loss
            loss = (policy_loss_batch + 
                   self.value_loss_coef * value_loss_batch + 
                   self.entropy_coef * entropy_loss_batch +
                   self.bluff_coef * bluff_loss_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            policy_loss += policy_loss_batch.item()
            value_loss += value_loss_batch.item()
            entropy_loss += entropy_loss_batch.item()
            bluff_loss += bluff_loss_batch.item()
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'total_loss': total_loss / epochs,
            'policy_loss': policy_loss / epochs,
            'value_loss': value_loss / epochs,
            'entropy_loss': entropy_loss / epochs,
            'bluff_loss': bluff_loss / epochs
        }
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        return advantages
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'episode_rewards': self.episode_rewards.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'bluff_attempts': sum(self.bluff_attempts),
            'bluff_success_rate': np.mean(self.bluff_successes) if self.bluff_successes else 0,
            'action_distribution': self.action_counts.copy(),
            'challenge_success_rate': self.challenge_success_rate,
            'block_success_rate': self.block_success_rate
        }
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'statistics': self.get_statistics()
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'statistics' in checkpoint:
            stats = checkpoint['statistics']
            self.episode_rewards = stats.get('episode_rewards', [])
            self.episode_lengths = stats.get('episode_lengths', [])
            self.action_counts = stats.get('action_distribution', {})
            self.challenge_success_rate = stats.get('challenge_success_rate', 0.0)
            self.block_success_rate = stats.get('block_success_rate', 0.0)