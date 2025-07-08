"""Multi-agent training loop with self-play for Coup."""

import numpy as np
import torch
import random
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict, deque
import logging
from datetime import datetime

from .environment import CoupEnvironment
from .agent import CoupPPOAgent
from .baseline import BaselineStrategies
from .metrics import MetricsTracker


class SelfPlayTrainer:
    """Self-play trainer for Coup agents."""
    
    def __init__(self, 
                 num_players: int = 4,
                 obs_size: int = None,
                 action_size: int = 16,
                 save_dir: str = "models",
                 log_dir: str = "logs"):
        
        self.num_players = num_players
        self.action_size = action_size
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Environment
        self.env = CoupEnvironment(num_players=num_players)
        
        # Calculate observation size if not provided
        if obs_size is None:
            dummy_obs = self.env.observation_spaces[self.env.agents[0]]
            obs_size = dummy_obs.shape[0]
        self.obs_size = obs_size
        
        # Agents
        self.agents = {}
        self.agent_pool = []  # For opponent rotation
        
        # Training state
        self.episode = 0
        self.total_timesteps = 0
        
        # Metrics and logging
        self.metrics = MetricsTracker()
        self.setup_logging()
        
        # Training parameters
        self.training_config = {
            'episodes_per_agent_update': 10,
            'agent_pool_size': 8,
            'opponent_rotation_frequency': 50,
            'evaluation_frequency': 100,
            'save_frequency': 500,
            'max_episodes': 10000
        }
        
        # Baseline strategies for evaluation
        self.baselines = BaselineStrategies()
        
        # Nash equilibrium detection
        self.strategy_history = defaultdict(lambda: deque(maxlen=100))
        self.convergence_threshold = 0.05
        self.last_convergence_check = 0
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = os.path.join(self.log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def create_agent(self, agent_id: str) -> CoupPPOAgent:
        """Create a new agent."""
        return CoupPPOAgent(
            obs_size=self.obs_size,
            action_size=self.action_size,
            agent_id=agent_id
        )
    
    def initialize_agents(self) -> None:
        """Initialize training agents."""
        # Create main training agents
        for i in range(self.num_players):
            agent_id = f"agent_{i}"
            self.agents[agent_id] = self.create_agent(agent_id)
        
        # Create agent pool for opponent rotation
        for i in range(self.training_config['agent_pool_size']):
            agent_id = f"pool_agent_{i}"
            self.agent_pool.append(self.create_agent(agent_id))
        
        self.logger.info(f"Initialized {len(self.agents)} main agents and {len(self.agent_pool)} pool agents")
    
    def select_opponents(self, main_agent_id: str) -> List[CoupPPOAgent]:
        """Select opponents for training episode."""
        # Mix of main agents and pool agents
        opponents = []
        
        # Always include some main agents
        other_main_agents = [agent for agent_id, agent in self.agents.items() 
                           if agent_id != main_agent_id]
        opponents.extend(random.sample(other_main_agents, min(2, len(other_main_agents))))
        
        # Fill remaining slots with pool agents
        remaining_slots = self.num_players - 1 - len(opponents)
        if remaining_slots > 0:
            pool_sample = random.sample(self.agent_pool, min(remaining_slots, len(self.agent_pool)))
            opponents.extend(pool_sample)
        
        return opponents[:self.num_players - 1]
    
    def run_episode(self, agents: List[CoupPPOAgent], training: bool = True) -> Dict[str, Any]:
        """Run a single episode."""
        observations = self.env.reset()
        episode_rewards = {agent.agent_id: 0 for agent in agents}
        episode_length = 0
        episode_actions = defaultdict(list)
        
        # Reset agent hidden states
        for agent in agents:
            agent.reset_hidden_state()
        
        # Agent mapping
        agent_mapping = {self.env.agents[i]: agents[i] for i in range(len(agents))}
        
        # Episode loop
        while not all(self.env.terminations.values()) and not all(self.env.truncations.values()):
            current_agent_name = self.env.agent_selection
            current_agent = agent_mapping[current_agent_name]
            
            # Skip if agent is terminated
            if self.env.terminations[current_agent_name] or self.env.truncations[current_agent_name]:
                self.env.step(None)
                continue
            
            # Get observation and game state
            observation = observations[current_agent_name]
            game_state = {
                'challenge_phase': self.env.challenge_phase,
                'block_phase': self.env.block_phase,
                'current_player': current_agent_name,
                'phase': self.env.game.state.phase.value
            }
            
            # Select action
            action = current_agent.act(observation, game_state, training=training)
            episode_actions[current_agent.agent_id].append(action)
            
            # Store previous observation for experience
            prev_observation = observation.copy()
            
            # Execute action
            self.env.step(action)
            
            # Get new observation and reward
            new_observations = {}
            for agent_name in self.env.agents:
                if not self.env.terminations[agent_name] and not self.env.truncations[agent_name]:
                    new_observations[agent_name] = self.env.observe(agent_name)
            
            reward = self.env.rewards[current_agent_name]
            done = self.env.terminations[current_agent_name] or self.env.truncations[current_agent_name]
            
            # Store experience
            if training:
                next_obs = new_observations.get(current_agent_name, observation)
                current_agent.store_experience(reward, next_obs, done)
            
            # Update metrics
            episode_rewards[current_agent.agent_id] += reward
            episode_length += 1
            
            # Update observations
            observations.update(new_observations)
        
        # Episode results
        results = {
            'episode_rewards': episode_rewards,
            'episode_length': episode_length,
            'winner': self._get_winner(agents),
            'episode_actions': dict(episode_actions)
        }
        
        return results
    
    def _get_winner(self, agents: List[CoupPPOAgent]) -> Optional[str]:
        """Get the winner of the episode."""
        winner = self.env.game.state.get_winner()
        if winner:
            # Map winner back to agent
            for i, agent in enumerate(agents):
                if self.env.game.state.players[i] == winner:
                    return agent.agent_id
        return None
    
    def update_agents(self, agents: List[CoupPPOAgent]) -> Dict[str, Dict[str, float]]:
        """Update agent policies."""
        update_results = {}
        
        for agent in agents:
            results = agent.update_policy()
            if results:
                update_results[agent.agent_id] = results
        
        return update_results
    
    def evaluate_agents(self, num_episodes: int = 50) -> Dict[str, Any]:
        """Evaluate agents against baselines and each other."""
        self.logger.info(f"Starting evaluation with {num_episodes} episodes")
        
        results = {
            'vs_baseline': {},
            'vs_self': {},
            'exploitability': {},
            'strategy_diversity': {}
        }
        
        # Evaluate against baselines
        for baseline_name in ['random', 'honest', 'aggressive', 'defensive']:
            win_rates = {}
            
            for agent_id, agent in self.agents.items():
                wins = 0
                
                for _ in range(num_episodes // 4):  # Fewer episodes per baseline
                    # Create baseline agents
                    baseline_agents = [
                        self.baselines.create_baseline_agent(baseline_name, f"baseline_{i}")
                        for i in range(self.num_players - 1)
                    ]
                    
                    # Run episode
                    episode_agents = [agent] + baseline_agents
                    random.shuffle(episode_agents)  # Randomize positions
                    
                    episode_results = self.run_episode(episode_agents, training=False)
                    
                    if episode_results['winner'] == agent_id:
                        wins += 1
                
                win_rates[agent_id] = wins / (num_episodes // 4)
            
            results['vs_baseline'][baseline_name] = win_rates
        
        # Evaluate self-play performance
        agent_win_counts = defaultdict(int)
        total_self_play_episodes = num_episodes // 2
        
        for _ in range(total_self_play_episodes):
            agents_sample = random.sample(list(self.agents.values()), self.num_players)
            episode_results = self.run_episode(agents_sample, training=False)
            
            if episode_results['winner']:
                agent_win_counts[episode_results['winner']] += 1
        
        for agent_id in self.agents:
            results['vs_self'][agent_id] = agent_win_counts[agent_id] / total_self_play_episodes
        
        # Calculate exploitability estimates
        for agent_id, agent in self.agents.items():
            belief_system = agent.belief_system
            exploitability = 0.0
            
            for opponent_id in self.agents:
                if opponent_id != agent_id:
                    exploitability += belief_system.get_exploitability_estimate(opponent_id)
            
            if len(self.agents) > 1:
                exploitability /= (len(self.agents) - 1)
            
            results['exploitability'][agent_id] = exploitability
        
        # Calculate strategy diversity
        for agent_id, agent in self.agents.items():
            action_dist = agent.action_counts
            total_actions = sum(action_dist.values())
            
            if total_actions > 0:
                probs = [count / total_actions for count in action_dist.values()]
                entropy = -sum(p * np.log2(p) for p in probs if p > 0)
                max_entropy = np.log2(len(action_dist))
                diversity = entropy / max_entropy if max_entropy > 0 else 0
            else:
                diversity = 0
            
            results['strategy_diversity'][agent_id] = diversity
        
        self.logger.info("Evaluation completed")
        return results
    
    def check_convergence(self) -> bool:
        """Check if strategies have converged to Nash equilibrium."""
        if self.episode - self.last_convergence_check < 100:
            return False
        
        self.last_convergence_check = self.episode
        
        # Collect recent strategy profiles
        current_strategies = {}
        for agent_id, agent in self.agents.items():
            action_dist = agent.action_counts
            total_actions = sum(action_dist.values())
            
            if total_actions > 0:
                strategy = [count / total_actions for count in action_dist.values()]
            else:
                strategy = [1.0 / len(action_dist) for _ in action_dist]
            
            current_strategies[agent_id] = strategy
            self.strategy_history[agent_id].append(strategy)
        
        # Check for convergence
        converged = True
        for agent_id, agent in self.agents.items():
            if len(self.strategy_history[agent_id]) < 10:
                converged = False
                break
            
            recent_strategies = list(self.strategy_history[agent_id])[-10:]
            
            # Calculate variance in recent strategies
            variances = []
            for i in range(len(recent_strategies[0])):
                values = [strategy[i] for strategy in recent_strategies]
                variance = np.var(values)
                variances.append(variance)
            
            avg_variance = np.mean(variances)
            if avg_variance > self.convergence_threshold:
                converged = False
                break
        
        if converged:
            self.logger.info(f"Strategy convergence detected at episode {self.episode}")
        
        return converged
    
    def save_checkpoint(self) -> None:
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(self.save_dir, f"checkpoint_{self.episode}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save agents
        for agent_id, agent in self.agents.items():
            agent.save_model(os.path.join(checkpoint_dir, f"{agent_id}.pt"))
        
        # Save pool agents
        pool_dir = os.path.join(checkpoint_dir, "pool")
        os.makedirs(pool_dir, exist_ok=True)
        for i, agent in enumerate(self.agent_pool):
            agent.save_model(os.path.join(pool_dir, f"pool_agent_{i}.pt"))
        
        # Save training state
        training_state = {
            'episode': self.episode,
            'total_timesteps': self.total_timesteps,
            'training_config': self.training_config,
            'metrics': self.metrics.get_summary()
        }
        
        with open(os.path.join(checkpoint_dir, "training_state.json"), 'w') as f:
            json.dump(training_state, f, indent=2)
        
        self.logger.info(f"Checkpoint saved at episode {self.episode}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load training checkpoint."""
        # Load training state
        with open(os.path.join(checkpoint_dir, "training_state.json"), 'r') as f:
            training_state = json.load(f)
        
        self.episode = training_state['episode']
        self.total_timesteps = training_state['total_timesteps']
        self.training_config.update(training_state.get('training_config', {}))
        
        # Load agents
        for agent_id, agent in self.agents.items():
            agent_path = os.path.join(checkpoint_dir, f"{agent_id}.pt")
            if os.path.exists(agent_path):
                agent.load_model(agent_path)
        
        # Load pool agents
        pool_dir = os.path.join(checkpoint_dir, "pool")
        if os.path.exists(pool_dir):
            for i, agent in enumerate(self.agent_pool):
                pool_path = os.path.join(pool_dir, f"pool_agent_{i}.pt")
                if os.path.exists(pool_path):
                    agent.load_model(pool_path)
        
        self.logger.info(f"Checkpoint loaded from episode {self.episode}")
    
    def train(self, max_episodes: Optional[int] = None, 
              progress_callback: Optional[Callable[[Dict], None]] = None) -> None:
        """Main training loop."""
        if not self.agents:
            self.initialize_agents()
        
        max_episodes = max_episodes or self.training_config['max_episodes']
        start_time = time.time()
        
        self.logger.info(f"Starting training for {max_episodes} episodes")
        
        try:
            while self.episode < max_episodes:
                # Select training agents
                training_agents = list(self.agents.values())
                random.shuffle(training_agents)
                
                # Run episode
                episode_results = self.run_episode(training_agents, training=True)
                
                # Update metrics
                self.metrics.update_episode(episode_results)
                
                # Update agents periodically
                if self.episode % self.training_config['episodes_per_agent_update'] == 0:
                    update_results = self.update_agents(training_agents)
                    self.metrics.update_training(update_results)
                
                # Rotate opponents
                if self.episode % self.training_config['opponent_rotation_frequency'] == 0:
                    self._rotate_agent_pool()
                
                # Evaluation
                if self.episode % self.training_config['evaluation_frequency'] == 0:
                    eval_results = self.evaluate_agents()
                    self.metrics.update_evaluation(eval_results)
                    
                    # Check convergence
                    if self.check_convergence():
                        self.logger.info("Training converged - stopping early")
                        break
                
                # Save checkpoint
                if self.episode % self.training_config['save_frequency'] == 0:
                    self.save_checkpoint()
                
                # Progress callback
                if progress_callback and self.episode % 10 == 0:
                    progress_data = {
                        'episode': self.episode,
                        'elapsed_time': time.time() - start_time,
                        'recent_metrics': self.metrics.get_recent_summary()
                    }
                    progress_callback(progress_data)
                
                self.episode += 1
                self.total_timesteps += episode_results['episode_length']
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
        
        finally:
            # Final save
            self.save_checkpoint()
            
            # Final evaluation
            final_eval = self.evaluate_agents(num_episodes=100)
            self.metrics.update_evaluation(final_eval)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Training completed in {elapsed_time:.2f} seconds")
            self.logger.info(f"Total episodes: {self.episode}")
            self.logger.info(f"Total timesteps: {self.total_timesteps}")
    
    def _rotate_agent_pool(self) -> None:
        """Rotate agents in the pool to maintain diversity."""
        # Replace some pool agents with copies of main agents (with noise)
        num_replacements = min(2, len(self.agent_pool))
        replacement_indices = random.sample(range(len(self.agent_pool)), num_replacements)
        
        for idx in replacement_indices:
            # Select a main agent to copy
            source_agent = random.choice(list(self.agents.values()))
            
            # Create new agent and copy weights with noise
            new_agent = self.create_agent(f"pool_agent_{idx}_gen_{self.episode}")
            new_agent.policy_net.load_state_dict(source_agent.policy_net.state_dict())
            
            # Add noise to weights
            with torch.no_grad():
                for param in new_agent.policy_net.parameters():
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)
            
            self.agent_pool[idx] = new_agent
        
        self.logger.debug(f"Rotated {num_replacements} agents in pool")


class CoupTrainer:
    """Main trainer class with additional utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.self_play_trainer = SelfPlayTrainer(**config.get('trainer_args', {}))
    
    def train(self, progress_callback: Optional[Callable] = None) -> None:
        """Train agents with self-play."""
        self.self_play_trainer.train(
            max_episodes=self.config.get('max_episodes', 10000),
            progress_callback=progress_callback
        )
    
    def evaluate(self, model_dir: str, num_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate trained models."""
        # Load checkpoint
        self.self_play_trainer.load_checkpoint(model_dir)
        
        # Run evaluation
        return self.self_play_trainer.evaluate_agents(num_episodes)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return self.self_play_trainer.metrics.get_summary()