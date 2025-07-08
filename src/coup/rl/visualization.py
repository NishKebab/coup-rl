"""Visualization and logging callbacks for Coup RL training."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import logging


class TrainingVisualizer:
    """Real-time visualization for training progress."""
    
    def __init__(self, save_dir: str = "visualizations", update_frequency: int = 10):
        self.save_dir = save_dir
        self.update_frequency = update_frequency
        self.episode_count = 0
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Data storage for plotting
        self.training_data = defaultdict(list)
        self.evaluation_data = defaultdict(list)
        self.strategy_data = defaultdict(lambda: defaultdict(list))
        
        # Plot style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def log_episode(self, episode_data: Dict[str, Any]) -> None:
        """Log episode data for visualization."""
        self.episode_count += 1
        
        # Store episode metrics
        for agent_id, reward in episode_data.get('episode_rewards', {}).items():
            self.training_data[f'{agent_id}_reward'].append(reward)
        
        self.training_data['episode_length'].append(episode_data.get('episode_length', 0))
        
        # Store action distributions
        for agent_id, actions in episode_data.get('episode_actions', {}).items():
            action_counts = defaultdict(int)
            for action in actions:
                action_counts[action] += 1
            
            for action, count in action_counts.items():
                self.strategy_data[agent_id][f'action_{action}'].append(count)
    
    def log_training_update(self, training_data: Dict[str, Dict[str, float]]) -> None:
        """Log training update metrics."""
        for agent_id, metrics in training_data.items():
            for metric_name, value in metrics.items():
                self.training_data[f'{agent_id}_{metric_name}'].append(value)
    
    def log_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """Log evaluation results."""
        timestamp = datetime.now()
        
        # Baseline performance
        vs_baseline = eval_data.get('vs_baseline', {})
        for baseline_name, agent_results in vs_baseline.items():
            for agent_id, win_rate in agent_results.items():
                self.evaluation_data[f'{agent_id}_vs_{baseline_name}'].append(win_rate)
        
        # Self-play performance
        vs_self = eval_data.get('vs_self', {})
        for agent_id, win_rate in vs_self.items():
            self.evaluation_data[f'{agent_id}_self_play'].append(win_rate)
        
        # Exploitability
        exploitability = eval_data.get('exploitability', {})
        for agent_id, score in exploitability.items():
            self.evaluation_data[f'{agent_id}_exploitability'].append(score)
        
        # Strategy diversity
        strategy_diversity = eval_data.get('strategy_diversity', {})
        for agent_id, score in strategy_diversity.items():
            self.evaluation_data[f'{agent_id}_diversity'].append(score)
        
        self.evaluation_data['timestamps'].append(timestamp)
    
    def create_training_dashboard(self) -> None:
        """Create comprehensive training dashboard."""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Episode Rewards', 'Policy Loss', 'Win Rates vs Baselines',
                'Action Distributions', 'Strategy Diversity', 'Exploitability',
                'Episode Length', 'Bluffing Success', 'Nash Convergence'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Episode Rewards
        agent_ids = self._get_agent_ids()
        for agent_id in agent_ids:
            if f'{agent_id}_reward' in self.training_data:
                rewards = self.training_data[f'{agent_id}_reward']
                rolling_avg = self._rolling_average(rewards, window=20)
                fig.add_trace(
                    go.Scatter(y=rolling_avg, name=f'{agent_id} Reward', line=dict(width=2)),
                    row=1, col=1
                )
        
        # Plot 2: Policy Loss
        for agent_id in agent_ids:
            if f'{agent_id}_policy_loss' in self.training_data:
                losses = self.training_data[f'{agent_id}_policy_loss']
                fig.add_trace(
                    go.Scatter(y=losses, name=f'{agent_id} Policy Loss', line=dict(width=1)),
                    row=1, col=2
                )
        
        # Plot 3: Win Rates vs Baselines
        baselines = ['random', 'honest', 'aggressive', 'defensive']
        for agent_id in agent_ids:
            for baseline in baselines:
                key = f'{agent_id}_vs_{baseline}'
                if key in self.evaluation_data:
                    win_rates = self.evaluation_data[key]
                    fig.add_trace(
                        go.Scatter(y=win_rates, name=f'{agent_id} vs {baseline}', 
                                 line=dict(width=2, dash='dash')),
                        row=1, col=3
                    )
        
        # Plot 4: Action Distributions (latest)
        self._add_action_distribution_plot(fig, row=2, col=1)
        
        # Plot 5: Strategy Diversity
        for agent_id in agent_ids:
            if f'{agent_id}_diversity' in self.evaluation_data:
                diversity = self.evaluation_data[f'{agent_id}_diversity']
                fig.add_trace(
                    go.Scatter(y=diversity, name=f'{agent_id} Diversity', line=dict(width=2)),
                    row=2, col=2
                )
        
        # Plot 6: Exploitability
        for agent_id in agent_ids:
            if f'{agent_id}_exploitability' in self.evaluation_data:
                exploitability = self.evaluation_data[f'{agent_id}_exploitability']
                fig.add_trace(
                    go.Scatter(y=exploitability, name=f'{agent_id} Exploitability', 
                             line=dict(width=2)),
                    row=2, col=3
                )
        
        # Plot 7: Episode Length
        if 'episode_length' in self.training_data:
            lengths = self.training_data['episode_length']
            rolling_avg = self._rolling_average(lengths, window=50)
            fig.add_trace(
                go.Scatter(y=rolling_avg, name='Avg Episode Length', 
                         line=dict(width=2, color='purple')),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text="Coup RL Training Dashboard",
            showlegend=True
        )
        
        # Save as HTML
        fig.write_html(os.path.join(self.save_dir, "training_dashboard.html"))
        
        return fig
    
    def _add_action_distribution_plot(self, fig, row: int, col: int) -> None:
        """Add action distribution plot to subplot."""
        agent_ids = self._get_agent_ids()
        
        # Create stacked bar chart of action distributions
        action_data = []
        
        for agent_id in agent_ids:
            if agent_id in self.strategy_data:
                total_actions = 0
                action_counts = defaultdict(int)
                
                # Sum up all actions for this agent
                for action_key, counts in self.strategy_data[agent_id].items():
                    if action_key.startswith('action_'):
                        action_num = int(action_key.split('_')[1])
                        action_counts[action_num] = sum(counts)
                        total_actions += sum(counts)
                
                # Calculate percentages
                if total_actions > 0:
                    for action_num in range(16):  # 16 possible actions
                        percentage = (action_counts[action_num] / total_actions) * 100
                        action_data.append({
                            'Agent': agent_id,
                            'Action': f'Action {action_num}',
                            'Percentage': percentage
                        })
        
        if action_data:
            df = pd.DataFrame(action_data)
            
            # Create stacked bar chart
            for action in df['Action'].unique():
                action_df = df[df['Action'] == action]
                fig.add_trace(
                    go.Bar(x=action_df['Agent'], y=action_df['Percentage'], 
                          name=action, showlegend=False),
                    row=row, col=col
                )
    
    def create_strategy_evolution_plot(self) -> go.Figure:
        """Create plot showing strategy evolution over time."""
        fig = go.Figure()
        
        agent_ids = self._get_agent_ids()
        
        for agent_id in agent_ids:
            if agent_id in self.strategy_data:
                # Calculate action probabilities over time
                action_probs_over_time = defaultdict(list)
                
                for episode in range(len(self.training_data.get(f'{agent_id}_reward', []))):
                    total_actions = 0
                    action_counts = defaultdict(int)
                    
                    # Count actions up to this episode
                    for action_key, counts in self.strategy_data[agent_id].items():
                        if action_key.startswith('action_') and len(counts) > episode:
                            action_num = int(action_key.split('_')[1])
                            action_counts[action_num] += counts[episode]
                            total_actions += counts[episode]
                    
                    # Calculate probabilities
                    if total_actions > 0:
                        for action_num in range(16):
                            prob = action_counts[action_num] / total_actions
                            action_probs_over_time[f'{agent_id}_action_{action_num}'].append(prob)
                    else:
                        for action_num in range(16):
                            action_probs_over_time[f'{agent_id}_action_{action_num}'].append(0)
                
                # Plot top 5 most used actions for each agent
                top_actions = {}
                for action_num in range(16):
                    key = f'{agent_id}_action_{action_num}'
                    if action_probs_over_time[key]:
                        avg_prob = np.mean(action_probs_over_time[key])
                        top_actions[action_num] = avg_prob
                
                top_5_actions = sorted(top_actions.items(), key=lambda x: x[1], reverse=True)[:5]
                
                for action_num, _ in top_5_actions:
                    key = f'{agent_id}_action_{action_num}'
                    probs = action_probs_over_time[key]
                    
                    fig.add_trace(go.Scatter(
                        y=self._rolling_average(probs, window=10),
                        name=f'{agent_id} Action {action_num}',
                        line=dict(width=2)
                    ))
        
        fig.update_layout(
            title="Strategy Evolution Over Time",
            xaxis_title="Episode",
            yaxis_title="Action Probability",
            height=600
        )
        
        fig.write_html(os.path.join(self.save_dir, "strategy_evolution.html"))
        return fig
    
    def create_nash_equilibrium_analysis(self) -> go.Figure:
        """Create Nash equilibrium convergence analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Strategy Convergence', 'Exploitability Over Time',
                'Mutual Best Response', 'Policy Gradient Norms'
            ]
        )
        
        # Strategy convergence (variance in strategy profiles)
        convergence_data = []
        agent_ids = self._get_agent_ids()
        
        for episode in range(min(100, len(self.training_data.get('episode_length', [])))):
            episode_variance = 0
            
            for agent_id in agent_ids:
                if agent_id in self.strategy_data:
                    # Calculate strategy variance for this episode window
                    window_start = max(0, episode - 10)
                    window_strategies = []
                    
                    for ep in range(window_start, episode + 1):
                        strategy = []
                        total_actions = 0
                        
                        for action_num in range(16):
                            key = f'action_{action_num}'
                            if key in self.strategy_data[agent_id] and len(self.strategy_data[agent_id][key]) > ep:
                                count = self.strategy_data[agent_id][key][ep]
                                strategy.append(count)
                                total_actions += count
                            else:
                                strategy.append(0)
                        
                        if total_actions > 0:
                            strategy = [s / total_actions for s in strategy]
                            window_strategies.append(strategy)
                    
                    if len(window_strategies) > 1:
                        # Calculate variance across action dimensions
                        strategy_array = np.array(window_strategies)
                        variances = np.var(strategy_array, axis=0)
                        episode_variance += np.mean(variances)
            
            if len(agent_ids) > 0:
                episode_variance /= len(agent_ids)
            
            convergence_data.append(episode_variance)
        
        fig.add_trace(
            go.Scatter(y=convergence_data, name='Strategy Variance', line=dict(width=2)),
            row=1, col=1
        )
        
        # Exploitability over time
        for agent_id in agent_ids:
            if f'{agent_id}_exploitability' in self.evaluation_data:
                exploitability = self.evaluation_data[f'{agent_id}_exploitability']
                fig.add_trace(
                    go.Scatter(y=exploitability, name=f'{agent_id} Exploitability', 
                             line=dict(width=2)),
                    row=1, col=2
                )
        
        # Policy gradient norms (if available)
        for agent_id in agent_ids:
            if f'{agent_id}_policy_loss' in self.training_data:
                losses = self.training_data[f'{agent_id}_policy_loss']
                # Use loss as proxy for gradient norm
                fig.add_trace(
                    go.Scatter(y=losses, name=f'{agent_id} Policy Loss', 
                             line=dict(width=1)),
                    row=2, col=2
                )
        
        fig.update_layout(
            title="Nash Equilibrium Analysis",
            height=800,
            showlegend=True
        )
        
        fig.write_html(os.path.join(self.save_dir, "nash_analysis.html"))
        return fig
    
    def create_bluffing_analysis(self) -> go.Figure:
        """Create analysis of bluffing behavior."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Bluff Success Rates', 'Challenge Accuracy',
                'Bluff Frequency Over Time', 'Deception Skill Matrix'
            ]
        )
        
        # This would require more detailed bluffing data
        # For now, create placeholder structure
        
        fig.update_layout(
            title="Bluffing Behavior Analysis",
            height=800
        )
        
        fig.write_html(os.path.join(self.save_dir, "bluffing_analysis.html"))
        return fig
    
    def _get_agent_ids(self) -> List[str]:
        """Extract agent IDs from training data."""
        agent_ids = set()
        
        for key in self.training_data.keys():
            if '_reward' in key:
                agent_id = key.replace('_reward', '')
                agent_ids.add(agent_id)
        
        for key in self.evaluation_data.keys():
            if '_self_play' in key:
                agent_id = key.replace('_self_play', '')
                agent_ids.add(agent_id)
        
        return list(agent_ids)
    
    def _rolling_average(self, data: List[float], window: int = 20) -> List[float]:
        """Calculate rolling average."""
        if len(data) < window:
            return data
        
        rolling_avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            avg = np.mean(data[start_idx:i + 1])
            rolling_avg.append(avg)
        
        return rolling_avg
    
    def save_training_data(self) -> None:
        """Save all training data to JSON."""
        data = {
            'training_data': dict(self.training_data),
            'evaluation_data': dict(self.evaluation_data),
            'strategy_data': dict(self.strategy_data),
            'episode_count': self.episode_count,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.save_dir, "training_data.json"), 'w') as f:
            json.dump(data, f, indent=2, default=str)


class ProgressCallback:
    """Callback for tracking training progress."""
    
    def __init__(self, visualizer: TrainingVisualizer, log_frequency: int = 100):
        self.visualizer = visualizer
        self.log_frequency = log_frequency
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, progress_data: Dict[str, Any]) -> None:
        """Called during training to log progress."""
        episode = progress_data.get('episode', 0)
        elapsed_time = progress_data.get('elapsed_time', 0)
        recent_metrics = progress_data.get('recent_metrics', {})
        
        # Log progress
        if episode % self.log_frequency == 0:
            self.logger.info(f"Episode {episode} - Elapsed: {elapsed_time:.2f}s")
            
            # Log agent performance
            agent_performance = recent_metrics.get('agent_performance', {})
            for agent_id, metrics in agent_performance.items():
                win_rate = metrics.get('win_rate', 0)
                avg_reward = metrics.get('avg_reward', 0)
                self.logger.info(f"  {agent_id}: Win Rate: {win_rate:.3f}, Avg Reward: {avg_reward:.3f}")
            
            # Update visualizations
            if episode % (self.log_frequency * 5) == 0:
                self.visualizer.create_training_dashboard()
                self.visualizer.save_training_data()


class RealTimeMonitor:
    """Real-time monitoring system for training."""
    
    def __init__(self, update_interval: int = 60):
        self.update_interval = update_interval
        self.is_monitoring = False
        self.metrics_history = []
    
    def start_monitoring(self, trainer) -> None:
        """Start real-time monitoring."""
        self.is_monitoring = True
        # This would set up real-time monitoring
        # For now, just log that monitoring started
        logging.info("Real-time monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.is_monitoring = False
        logging.info("Real-time monitoring stopped")
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update metrics for real-time display."""
        timestamp = datetime.now()
        metrics_with_time = {
            'timestamp': timestamp,
            **metrics
        }
        self.metrics_history.append(metrics_with_time)
        
        # Keep only recent history
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]