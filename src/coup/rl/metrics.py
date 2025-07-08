"""Metrics tracking and evaluation for Coup RL training."""

import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class MetricsTracker:
    """Comprehensive metrics tracking for Coup RL training."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Episode metrics
        self.episode_rewards = defaultdict(lambda: deque(maxlen=window_size))
        self.episode_lengths = deque(maxlen=window_size)
        self.win_rates = defaultdict(lambda: deque(maxlen=window_size))
        
        # Action metrics
        self.action_distributions = defaultdict(lambda: defaultdict(int))
        self.action_success_rates = defaultdict(lambda: defaultdict(list))
        
        # Bluffing metrics
        self.bluff_attempts = defaultdict(int)
        self.bluff_successes = defaultdict(int)
        self.challenge_attempts = defaultdict(int)
        self.challenge_successes = defaultdict(int)
        self.block_attempts = defaultdict(int)
        self.block_successes = defaultdict(int)
        
        # Training metrics
        self.policy_losses = defaultdict(lambda: deque(maxlen=window_size))
        self.value_losses = defaultdict(lambda: deque(maxlen=window_size))
        self.entropy_losses = defaultdict(lambda: deque(maxlen=window_size))
        self.bluff_losses = defaultdict(lambda: deque(maxlen=window_size))
        
        # Evaluation metrics
        self.baseline_win_rates = defaultdict(dict)
        self.self_play_win_rates = defaultdict(lambda: deque(maxlen=50))
        self.exploitability_scores = defaultdict(lambda: deque(maxlen=50))
        self.strategy_diversity_scores = defaultdict(lambda: deque(maxlen=50))
        
        # Nash equilibrium metrics
        self.strategy_profiles = defaultdict(lambda: deque(maxlen=100))
        self.convergence_metrics = deque(maxlen=100)
        self.exploitability_estimates = deque(maxlen=100)
        
        # Game flow metrics
        self.game_length_distribution = defaultdict(int)
        self.character_usage = defaultdict(lambda: defaultdict(int))
        self.endgame_patterns = defaultdict(int)
        
        # Timestamps
        self.episode_timestamps = []
        self.evaluation_timestamps = []
    
    def update_episode(self, episode_results: Dict[str, Any]) -> None:
        """Update metrics with episode results."""
        episode_rewards = episode_results.get('episode_rewards', {})
        episode_length = episode_results.get('episode_length', 0)
        winner = episode_results.get('winner')
        episode_actions = episode_results.get('episode_actions', {})
        
        # Store timestamp
        self.episode_timestamps.append(datetime.now())
        
        # Episode length
        self.episode_lengths.append(episode_length)
        self.game_length_distribution[episode_length // 10 * 10] += 1
        
        # Rewards and win rates
        for agent_id, reward in episode_rewards.items():
            self.episode_rewards[agent_id].append(reward)
            
            # Win rate (1 if winner, 0 otherwise)
            win = 1 if agent_id == winner else 0
            self.win_rates[agent_id].append(win)
        
        # Action distributions
        for agent_id, actions in episode_actions.items():
            for action in actions:
                self.action_distributions[agent_id][action] += 1
    
    def update_action_result(self, agent_id: str, action: int, success: bool, 
                           action_type: str = 'normal') -> None:
        """Update action-specific metrics."""
        self.action_success_rates[agent_id][action].append(1 if success else 0)
        
        # Special action types
        if action_type == 'bluff':
            self.bluff_attempts[agent_id] += 1
            if success:
                self.bluff_successes[agent_id] += 1
        elif action_type == 'challenge':
            self.challenge_attempts[agent_id] += 1
            if success:
                self.challenge_successes[agent_id] += 1
        elif action_type == 'block':
            self.block_attempts[agent_id] += 1
            if success:
                self.block_successes[agent_id] += 1
    
    def update_training(self, training_results: Dict[str, Dict[str, float]]) -> None:
        """Update training-specific metrics."""
        for agent_id, losses in training_results.items():
            if 'policy_loss' in losses:
                self.policy_losses[agent_id].append(losses['policy_loss'])
            if 'value_loss' in losses:
                self.value_losses[agent_id].append(losses['value_loss'])
            if 'entropy_loss' in losses:
                self.entropy_losses[agent_id].append(losses['entropy_loss'])
            if 'bluff_loss' in losses:
                self.bluff_losses[agent_id].append(losses['bluff_loss'])
    
    def update_evaluation(self, eval_results: Dict[str, Any]) -> None:
        """Update evaluation metrics."""
        self.evaluation_timestamps.append(datetime.now())
        
        # Baseline win rates
        vs_baseline = eval_results.get('vs_baseline', {})
        for baseline_name, win_rates in vs_baseline.items():
            for agent_id, win_rate in win_rates.items():
                if baseline_name not in self.baseline_win_rates[agent_id]:
                    self.baseline_win_rates[agent_id][baseline_name] = deque(maxlen=50)
                self.baseline_win_rates[agent_id][baseline_name].append(win_rate)
        
        # Self-play win rates
        vs_self = eval_results.get('vs_self', {})
        for agent_id, win_rate in vs_self.items():
            self.self_play_win_rates[agent_id].append(win_rate)
        
        # Exploitability
        exploitability = eval_results.get('exploitability', {})
        for agent_id, score in exploitability.items():
            self.exploitability_scores[agent_id].append(score)
        
        # Strategy diversity
        strategy_diversity = eval_results.get('strategy_diversity', {})
        for agent_id, score in strategy_diversity.items():
            self.strategy_diversity_scores[agent_id].append(score)
    
    def update_strategy_profile(self, agent_id: str, strategy: List[float]) -> None:
        """Update strategy profile for Nash equilibrium detection."""
        self.strategy_profiles[agent_id].append(strategy.copy())
    
    def calculate_convergence_metric(self) -> float:
        """Calculate strategy convergence metric."""
        if not self.strategy_profiles:
            return 1.0  # High variance indicates no convergence
        
        # Calculate variance in recent strategy profiles
        total_variance = 0.0
        num_agents = 0
        
        for agent_id, profiles in self.strategy_profiles.items():
            if len(profiles) < 10:
                continue
            
            recent_profiles = list(profiles)[-10:]
            
            # Calculate variance for each action dimension
            action_variances = []
            for action_idx in range(len(recent_profiles[0])):
                values = [profile[action_idx] for profile in recent_profiles]
                variance = np.var(values)
                action_variances.append(variance)
            
            total_variance += np.mean(action_variances)
            num_agents += 1
        
        if num_agents == 0:
            return 1.0
        
        avg_variance = total_variance / num_agents
        self.convergence_metrics.append(avg_variance)
        
        return avg_variance
    
    def calculate_exploitability_estimate(self) -> float:
        """Calculate overall exploitability estimate."""
        if not self.exploitability_scores:
            return 0.5
        
        # Average exploitability across all agents
        recent_scores = []
        for agent_scores in self.exploitability_scores.values():
            if agent_scores:
                recent_scores.append(agent_scores[-1])
        
        if not recent_scores:
            return 0.5
        
        avg_exploitability = np.mean(recent_scores)
        self.exploitability_estimates.append(avg_exploitability)
        
        return avg_exploitability
    
    def get_recent_summary(self, window: int = 20) -> Dict[str, Any]:
        """Get summary of recent metrics."""
        summary = {
            'episode_count': len(self.episode_lengths),
            'avg_episode_length': np.mean(list(self.episode_lengths)[-window:]) if self.episode_lengths else 0,
            'agent_performance': {},
            'training_progress': {},
            'bluffing_stats': {}
        }
        
        # Agent performance
        for agent_id in self.episode_rewards:
            recent_rewards = list(self.episode_rewards[agent_id])[-window:]
            recent_wins = list(self.win_rates[agent_id])[-window:]
            
            summary['agent_performance'][agent_id] = {
                'avg_reward': np.mean(recent_rewards) if recent_rewards else 0,
                'win_rate': np.mean(recent_wins) if recent_wins else 0,
                'reward_std': np.std(recent_rewards) if recent_rewards else 0
            }
        
        # Training progress
        for agent_id in self.policy_losses:
            recent_policy_loss = list(self.policy_losses[agent_id])[-window:]
            recent_value_loss = list(self.value_losses[agent_id])[-window:]
            
            summary['training_progress'][agent_id] = {
                'policy_loss': np.mean(recent_policy_loss) if recent_policy_loss else 0,
                'value_loss': np.mean(recent_value_loss) if recent_value_loss else 0
            }
        
        # Bluffing statistics
        for agent_id in self.bluff_attempts:
            bluff_rate = (self.bluff_successes[agent_id] / self.bluff_attempts[agent_id] 
                         if self.bluff_attempts[agent_id] > 0 else 0)
            challenge_rate = (self.challenge_successes[agent_id] / self.challenge_attempts[agent_id] 
                            if self.challenge_attempts[agent_id] > 0 else 0)
            
            summary['bluffing_stats'][agent_id] = {
                'bluff_success_rate': bluff_rate,
                'challenge_success_rate': challenge_rate,
                'bluff_frequency': self.bluff_attempts[agent_id] / max(1, len(self.episode_lengths))
            }
        
        return summary
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete metrics summary."""
        return {
            'episode_metrics': {
                'total_episodes': len(self.episode_lengths),
                'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
                'episode_length_std': np.std(self.episode_lengths) if self.episode_lengths else 0
            },
            'agent_performance': self._get_agent_performance_summary(),
            'action_statistics': self._get_action_statistics(),
            'bluffing_analysis': self._get_bluffing_analysis(),
            'training_metrics': self._get_training_metrics(),
            'evaluation_results': self._get_evaluation_results(),
            'nash_equilibrium': self._get_nash_metrics(),
            'game_analysis': self._get_game_analysis()
        }
    
    def _get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get agent performance summary."""
        performance = {}
        
        for agent_id in self.episode_rewards:
            rewards = list(self.episode_rewards[agent_id])
            wins = list(self.win_rates[agent_id])
            
            performance[agent_id] = {
                'avg_reward': np.mean(rewards) if rewards else 0,
                'reward_std': np.std(rewards) if rewards else 0,
                'win_rate': np.mean(wins) if wins else 0,
                'total_games': len(rewards),
                'reward_trend': self._calculate_trend(rewards) if len(rewards) > 10 else 0
            }
        
        return performance
    
    def _get_action_statistics(self) -> Dict[str, Any]:
        """Get action usage statistics."""
        action_stats = {}
        
        for agent_id, actions in self.action_distributions.items():
            total_actions = sum(actions.values())
            
            if total_actions > 0:
                action_probs = {action: count / total_actions 
                              for action, count in actions.items()}
                
                # Calculate entropy (strategy diversity)
                probs = list(action_probs.values())
                entropy = -sum(p * np.log2(p) for p in probs if p > 0)
                max_entropy = np.log2(len(action_probs))
                diversity = entropy / max_entropy if max_entropy > 0 else 0
                
                action_stats[agent_id] = {
                    'action_distribution': action_probs,
                    'strategy_diversity': diversity,
                    'most_used_action': max(actions.items(), key=lambda x: x[1])[0],
                    'total_actions': total_actions
                }
        
        return action_stats
    
    def _get_bluffing_analysis(self) -> Dict[str, Any]:
        """Get bluffing behavior analysis."""
        bluffing = {}
        
        for agent_id in self.bluff_attempts:
            bluff_success_rate = (self.bluff_successes[agent_id] / self.bluff_attempts[agent_id] 
                                if self.bluff_attempts[agent_id] > 0 else 0)
            
            challenge_success_rate = (self.challenge_successes[agent_id] / self.challenge_attempts[agent_id] 
                                    if self.challenge_attempts[agent_id] > 0 else 0)
            
            block_success_rate = (self.block_successes[agent_id] / self.block_attempts[agent_id] 
                                if self.block_attempts[agent_id] > 0 else 0)
            
            bluffing[agent_id] = {
                'bluff_attempts': self.bluff_attempts[agent_id],
                'bluff_success_rate': bluff_success_rate,
                'challenge_attempts': self.challenge_attempts[agent_id],
                'challenge_success_rate': challenge_success_rate,
                'block_attempts': self.block_attempts[agent_id],
                'block_success_rate': block_success_rate,
                'overall_deception_skill': (bluff_success_rate + challenge_success_rate) / 2
            }
        
        return bluffing
    
    def _get_training_metrics(self) -> Dict[str, Any]:
        """Get training progress metrics."""
        training = {}
        
        for agent_id in self.policy_losses:
            policy_losses = list(self.policy_losses[agent_id])
            value_losses = list(self.value_losses[agent_id])
            
            training[agent_id] = {
                'avg_policy_loss': np.mean(policy_losses) if policy_losses else 0,
                'avg_value_loss': np.mean(value_losses) if value_losses else 0,
                'policy_loss_trend': self._calculate_trend(policy_losses) if len(policy_losses) > 10 else 0,
                'training_stability': np.std(policy_losses) if policy_losses else 0
            }
        
        return training
    
    def _get_evaluation_results(self) -> Dict[str, Any]:
        """Get evaluation results summary."""
        evaluation = {
            'baseline_performance': {},
            'self_play_performance': {},
            'relative_strength': {}
        }
        
        # Baseline performance
        for agent_id, baselines in self.baseline_win_rates.items():
            agent_baseline_perf = {}
            for baseline_name, win_rates in baselines.items():
                recent_rates = list(win_rates)[-10:]  # Last 10 evaluations
                agent_baseline_perf[baseline_name] = {
                    'win_rate': np.mean(recent_rates) if recent_rates else 0,
                    'consistency': 1 - np.std(recent_rates) if recent_rates else 0
                }
            evaluation['baseline_performance'][agent_id] = agent_baseline_perf
        
        # Self-play performance
        for agent_id, win_rates in self.self_play_win_rates.items():
            recent_rates = list(win_rates)[-10:]
            evaluation['self_play_performance'][agent_id] = {
                'win_rate': np.mean(recent_rates) if recent_rates else 0,
                'trend': self._calculate_trend(recent_rates) if len(recent_rates) > 5 else 0
            }
        
        return evaluation
    
    def _get_nash_metrics(self) -> Dict[str, Any]:
        """Get Nash equilibrium metrics."""
        convergence_metric = self.calculate_convergence_metric()
        exploitability_metric = self.calculate_exploitability_estimate()
        
        return {
            'strategy_convergence': convergence_metric,
            'estimated_exploitability': exploitability_metric,
            'convergence_trend': self._calculate_trend(list(self.convergence_metrics)) if len(self.convergence_metrics) > 5 else 0,
            'is_converged': convergence_metric < 0.05,  # Threshold for convergence
            'equilibrium_quality': 1 - exploitability_metric  # Lower exploitability = better equilibrium
        }
    
    def _get_game_analysis(self) -> Dict[str, Any]:
        """Get game flow analysis."""
        return {
            'game_length_distribution': dict(self.game_length_distribution),
            'avg_game_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'character_usage': dict(self.character_usage),
            'endgame_patterns': dict(self.endgame_patterns)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]  # Slope
    
    def save_metrics(self, filepath: str) -> None:
        """Save metrics to file."""
        metrics_data = self.get_summary()
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
    
    def plot_training_progress(self, save_path: Optional[str] = None) -> None:
        """Plot training progress visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Win rates over time
        axes[0, 0].set_title('Win Rates Over Time')
        for agent_id, win_rates in self.win_rates.items():
            if win_rates:
                # Calculate rolling average
                window = min(20, len(win_rates))
                rolling_avg = np.convolve(win_rates, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(rolling_avg, label=agent_id)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Episode rewards
        axes[0, 1].set_title('Episode Rewards')
        for agent_id, rewards in self.episode_rewards.items():
            if rewards:
                window = min(20, len(rewards))
                rolling_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                axes[0, 1].plot(rolling_avg, label=agent_id)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Policy loss
        axes[0, 2].set_title('Policy Loss')
        for agent_id, losses in self.policy_losses.items():
            if losses:
                axes[0, 2].plot(losses, label=agent_id, alpha=0.7)
        axes[0, 2].set_xlabel('Update')
        axes[0, 2].set_ylabel('Policy Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Action distribution
        axes[1, 0].set_title('Action Distribution')
        for agent_id, actions in self.action_distributions.items():
            if actions:
                total = sum(actions.values())
                action_probs = [actions.get(i, 0) / total for i in range(16)]
                axes[1, 0].bar(range(16), action_probs, alpha=0.7, label=agent_id)
        axes[1, 0].set_xlabel('Action')
        axes[1, 0].set_ylabel('Probability')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Bluffing success rates
        axes[1, 1].set_title('Bluffing Success Rates')
        agent_ids = list(self.bluff_attempts.keys())
        bluff_rates = [self.bluff_successes[aid] / max(1, self.bluff_attempts[aid]) 
                      for aid in agent_ids]
        challenge_rates = [self.challenge_successes[aid] / max(1, self.challenge_attempts[aid]) 
                          for aid in agent_ids]
        
        x_pos = np.arange(len(agent_ids))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, bluff_rates, width, label='Bluff Success')
        axes[1, 1].bar(x_pos + width/2, challenge_rates, width, label='Challenge Success')
        axes[1, 1].set_xlabel('Agent')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(agent_ids, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Convergence metrics
        axes[1, 2].set_title('Strategy Convergence')
        if self.convergence_metrics:
            axes[1, 2].plot(self.convergence_metrics, label='Convergence Metric')
            axes[1, 2].axhline(y=0.05, color='r', linestyle='--', label='Convergence Threshold')
        if self.exploitability_estimates:
            ax2 = axes[1, 2].twinx()
            ax2.plot(self.exploitability_estimates, color='orange', label='Exploitability')
            ax2.set_ylabel('Exploitability')
        axes[1, 2].set_xlabel('Evaluation')
        axes[1, 2].set_ylabel('Convergence Metric')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()