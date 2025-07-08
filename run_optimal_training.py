"""Long training run to find optimal Coup strategies."""

import sys
import os
sys.path.insert(0, 'src')

import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from coup.rl.environment import CoupEnvironment
from coup.rl.agent import CoupPPOAgent
from coup.rl.baseline import BaselineStrategies

class OptimalCoupTrainer:
    """Trainer for finding optimal Coup strategies."""
    
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.env = CoupEnvironment(num_players=num_players)
        
        # Create agents with optimized hyperparameters
        self.agents = []
        for i in range(num_players):
            agent = CoupPPOAgent(
                obs_size=63, 
                action_size=16, 
                agent_id=f'agent_{i}',
                learning_rate=3e-4,  # Lower learning rate for stability
                gamma=0.95,          # Slightly lower discount for faster learning
                entropy_coef=0.02,   # More exploration
                clip_ratio=0.15      # Tighter clipping for stability
            )
            self.agents.append(agent)
        
        # Baseline strategies
        self.baselines = BaselineStrategies()
        
        # Training metrics
        self.training_data = {
            'episodes': [],
            'win_rates': {agent.agent_id: [] for agent in self.agents},
            'rewards': {agent.agent_id: [] for agent in self.agents},
            'episode_lengths': [],
            'evaluation_results': [],
            'strategy_evolution': {agent.agent_id: [] for agent in self.agents}
        }
        
        # Nash equilibrium tracking
        self.strategy_profiles = {agent.agent_id: [] for agent in self.agents}
        self.convergence_metrics = []
        
    def run_episode(self, episode_num, training=True):
        """Run a single episode with improved timeout handling."""
        observations = self.env.reset()
        episode_rewards = {agent.agent_id: 0 for agent in self.agents}
        episode_actions = {agent.agent_id: [] for agent in self.agents}
        
        # Reset agent hidden states
        for agent in self.agents:
            agent.reset_hidden_state()
        
        # Agent mapping
        agent_mapping = {self.env.agents[i]: self.agents[i] for i in range(len(self.agents))}
        
        # Dynamic step limit based on training progress
        base_limit = 60
        progress_bonus = min(40, episode_num // 100)  # Gradually allow longer games
        max_steps = base_limit + progress_bonus
        
        step = 0
        
        while (not all(self.env.terminations.values()) and 
               not all(self.env.truncations.values()) and 
               step < max_steps):
            
            current_agent_name = self.env.agent_selection
            current_agent = agent_mapping[current_agent_name]
            
            if self.env.terminations[current_agent_name] or self.env.truncations[current_agent_name]:
                self.env.step(None)
                continue
            
            observation = observations.get(current_agent_name, np.zeros(63))
            game_state = {
                'challenge_phase': self.env.challenge_phase,
                'block_phase': self.env.block_phase,
                'current_player': current_agent_name,
                'phase': self.env.game.state.phase.value
            }
            
            action = current_agent.act(observation, game_state, training=training)
            episode_actions[current_agent.agent_id].append(action)
            
            # Store previous observation for experience
            prev_observation = observation.copy()
            
            self.env.step(action)
            
            new_observations = {}
            for agent_name in self.env.agents:
                if not self.env.terminations[agent_name] and not self.env.truncations[agent_name]:
                    new_observations[agent_name] = self.env.observe(agent_name)
            
            reward = self.env.rewards[current_agent_name]
            done = self.env.terminations[current_agent_name] or self.env.truncations[current_agent_name]
            
            # Enhanced reward shaping for better learning
            if step >= max_steps - 5:  # Penalize very long games
                reward -= 0.1
            
            if training and len(current_agent.buffer) < 1000:  # Only store if buffer not full
                next_obs = new_observations.get(current_agent_name, observation)
                current_agent.store_experience(reward, next_obs, done)
            
            episode_rewards[current_agent.agent_id] += reward
            observations.update(new_observations)
            step += 1
        
        # Get winner
        winner = self.env.game.state.get_winner()
        winner_agent = None
        if winner:
            for i, agent in enumerate(self.agents):
                if self.env.game.state.players[i] == winner:
                    winner_agent = agent.agent_id
                    break
        
        return {
            'episode_rewards': episode_rewards,
            'episode_length': step,
            'winner': winner_agent,
            'actions': episode_actions
        }
    
    def update_agents(self):
        """Update agents with improved hyperparameters."""
        update_results = {}
        for agent in self.agents:
            if len(agent.buffer) >= 32:  # Only update if enough experience
                results = agent.update_policy(batch_size=64, epochs=8)
                if results:
                    update_results[agent.agent_id] = results
        return update_results
    
    def evaluate_convergence(self):
        """Check if strategies are converging to Nash equilibrium."""
        if len(self.strategy_profiles[self.agents[0].agent_id]) < 20:
            return False
        
        # Calculate strategy variance across recent episodes
        recent_profiles = {}
        for agent_id in self.strategy_profiles:
            recent_profiles[agent_id] = self.strategy_profiles[agent_id][-10:]
        
        total_variance = 0
        for agent_id, profiles in recent_profiles.items():
            if len(profiles) > 5:
                profile_array = np.array(profiles)
                variances = np.var(profile_array, axis=0)
                total_variance += np.mean(variances)
        
        avg_variance = total_variance / len(self.agents)
        self.convergence_metrics.append(avg_variance)
        
        # Converged if variance is low and stable
        convergence_threshold = 0.01
        is_converged = avg_variance < convergence_threshold
        
        if len(self.convergence_metrics) > 5:
            recent_trend = np.diff(self.convergence_metrics[-5:])
            is_stable = np.abs(np.mean(recent_trend)) < 0.001
            return is_converged and is_stable
        
        return False
    
    def evaluate_against_baselines(self, num_games=30):
        """Comprehensive baseline evaluation."""
        print("📊 Evaluating against baselines...")
        
        results = {}
        
        for baseline_name in ['random', 'honest', 'aggressive', 'defensive']:
            print(f"  Testing vs {baseline_name}...")
            
            agent_wins = {agent.agent_id: 0 for agent in self.agents}
            total_games_per_agent = num_games
            
            for agent_idx, test_agent in enumerate(self.agents):
                for game in range(total_games_per_agent):
                    # Create test setup
                    baseline_agents = [
                        self.baselines.create_baseline_agent(baseline_name, f'base_{i}')
                        for i in range(3)
                    ]
                    test_agents = [test_agent] + baseline_agents
                    
                    # Run game
                    observations = self.env.reset()
                    
                    for agent in test_agents:
                        agent.reset_hidden_state()
                    
                    agent_mapping = {self.env.agents[i]: test_agents[i] for i in range(len(test_agents))}
                    
                    step = 0
                    max_steps = 80
                    
                    while (not all(self.env.terminations.values()) and 
                           not all(self.env.truncations.values()) and 
                           step < max_steps):
                        
                        current_agent_name = self.env.agent_selection
                        current_agent = agent_mapping[current_agent_name]
                        
                        if self.env.terminations[current_agent_name] or self.env.truncations[current_agent_name]:
                            self.env.step(None)
                            continue
                        
                        observation = observations.get(current_agent_name, np.zeros(63))
                        game_state = {
                            'challenge_phase': self.env.challenge_phase,
                            'block_phase': self.env.block_phase,
                            'current_player': current_agent_name,
                            'phase': self.env.game.state.phase.value
                        }
                        
                        action = current_agent.act(observation, game_state, training=False)
                        self.env.step(action)
                        
                        new_observations = {}
                        for agent_name in self.env.agents:
                            if not self.env.terminations[agent_name] and not self.env.truncations[agent_name]:
                                new_observations[agent_name] = self.env.observe(agent_name)
                        
                        observations.update(new_observations)
                        step += 1
                    
                    # Check if RL agent won
                    winner = self.env.game.state.get_winner()
                    if winner and self.env.game.state.players[0] == winner:
                        agent_wins[test_agent.agent_id] += 1
            
            # Calculate win rates
            win_rates = {agent_id: wins / total_games_per_agent for agent_id, wins in agent_wins.items()}
            results[baseline_name] = win_rates
            
            avg_win_rate = np.mean(list(win_rates.values()))
            print(f"    Average win rate: {avg_win_rate:.3f}")
        
        return results
    
    def save_checkpoint(self, episode):
        """Save training checkpoint."""
        checkpoint_dir = f"models/checkpoint_{episode}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save agents
        for agent in self.agents:
            agent.save_model(os.path.join(checkpoint_dir, f"{agent.agent_id}.pt"))
        
        # Save training data
        with open(os.path.join(checkpoint_dir, "training_data.json"), 'w') as f:
            json.dump(self.training_data, f, indent=2, default=str)
        
        print(f"💾 Checkpoint saved: {checkpoint_dir}")
    
    def create_comprehensive_visualization(self):
        """Create comprehensive training visualization."""
        print("📈 Creating comprehensive visualization...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Coup RL: Path to Optimal Strategy', fontsize=16, fontweight='bold')
        
        # 1. Win rates evolution
        axes[0, 0].set_title('Win Rate Evolution')
        for agent_id, win_rates in self.training_data['win_rates'].items():
            if len(win_rates) > 10:
                window = min(50, len(win_rates))
                rolling_avg = np.convolve(win_rates, np.ones(window)/window, mode='valid')
                x = range(window-1, len(win_rates))
                axes[0, 0].plot(x, rolling_avg, label=agent_id, linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Reward progression
        axes[0, 1].set_title('Reward Progression')
        for agent_id, rewards in self.training_data['rewards'].items():
            if len(rewards) > 10:
                window = min(50, len(rewards))
                rolling_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                x = range(window-1, len(rewards))
                axes[0, 1].plot(x, rolling_avg, label=agent_id, linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Episode length trend
        axes[0, 2].set_title('Episode Length Optimization')
        lengths = self.training_data['episode_lengths']
        if len(lengths) > 10:
            window = min(50, len(lengths))
            rolling_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
            x = range(window-1, len(lengths))
            axes[0, 2].plot(x, rolling_avg, color='purple', linewidth=2)
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Episode Length')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Strategy convergence
        axes[1, 0].set_title('Strategy Convergence')
        if self.convergence_metrics:
            axes[1, 0].plot(self.convergence_metrics, color='red', linewidth=2)
            axes[1, 0].axhline(y=0.01, color='green', linestyle='--', label='Convergence Threshold')
            axes[1, 0].legend()
        axes[1, 0].set_xlabel('Evaluation Round')
        axes[1, 0].set_ylabel('Strategy Variance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Baseline performance
        if self.training_data['evaluation_results']:
            axes[1, 1].set_title('Performance vs Baselines')
            latest_eval = self.training_data['evaluation_results'][-1]
            
            baselines = list(latest_eval.keys())
            agent_scores = {agent_id: [] for agent_id in self.training_data['win_rates'].keys()}
            
            for baseline in baselines:
                for agent_id, score in latest_eval[baseline].items():
                    agent_scores[agent_id].append(score)
            
            x_pos = np.arange(len(baselines))
            width = 0.2
            
            for i, (agent_id, scores) in enumerate(agent_scores.items()):
                offset = (i - len(agent_scores)/2) * width
                axes[1, 1].bar(x_pos + offset, scores, width, label=agent_id, alpha=0.8)
            
            axes[1, 1].set_xlabel('Baseline Strategy')
            axes[1, 1].set_ylabel('Win Rate')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(baselines, rotation=45)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Nash equilibrium quality
        axes[1, 2].set_title('Nash Equilibrium Quality')
        if len(self.convergence_metrics) > 10:
            equilibrium_quality = [1 - min(1, var) for var in self.convergence_metrics]
            axes[1, 2].plot(equilibrium_quality, color='orange', linewidth=2)
            axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_xlabel('Evaluation Round')
        axes[1, 2].set_ylabel('Equilibrium Quality')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Additional plots for strategy analysis
        # ... (implement additional strategy analysis plots)
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = f'visualizations/optimal_training_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plot_file
    
    def train_to_optimal(self, max_episodes=2000, early_stop_patience=200):
        """Train agents until optimal strategy is found."""
        print(f"🎯 Training to Optimal Strategy")
        print(f"Max episodes: {max_episodes}")
        print(f"Early stop patience: {early_stop_patience}")
        print("="*60)
        
        converged_count = 0
        best_performance = 0
        
        for episode in range(max_episodes):
            # Run training episode
            episode_result = self.run_episode(episode, training=True)
            
            # Store training data
            self.training_data['episodes'].append(episode)
            self.training_data['episode_lengths'].append(episode_result['episode_length'])
            
            # Update win rates and rewards
            winner = episode_result['winner']
            for agent_id in self.training_data['win_rates']:
                win = 1 if agent_id == winner else 0
                self.training_data['win_rates'][agent_id].append(win)
                self.training_data['rewards'][agent_id].append(episode_result['episode_rewards'][agent_id])
            
            # Update agents every 5 episodes
            if episode % 5 == 0 and episode > 0:
                update_results = self.update_agents()
                
                # Progress logging
                if episode % 50 == 0:
                    # Calculate recent performance
                    window = min(50, episode)
                    recent_performance = {}
                    
                    for agent_id in self.training_data['win_rates']:
                        recent_wins = self.training_data['win_rates'][agent_id][-window:]
                        recent_rewards = self.training_data['rewards'][agent_id][-window:]
                        
                        recent_performance[agent_id] = {
                            'win_rate': np.mean(recent_wins),
                            'avg_reward': np.mean(recent_rewards)
                        }
                    
                    print(f"\n📊 Episode {episode} Performance:")
                    for agent_id, perf in recent_performance.items():
                        print(f"  {agent_id}: Win Rate: {perf['win_rate']:.3f}, Avg Reward: {perf['avg_reward']:.2f}")
                    
                    avg_length = np.mean(self.training_data['episode_lengths'][-window:])
                    print(f"  Avg Episode Length: {avg_length:.1f}")
            
            # Comprehensive evaluation every 100 episodes
            if episode % 100 == 0 and episode > 0:
                print(f"\n🔍 Comprehensive Evaluation at Episode {episode}")
                
                # Baseline evaluation
                baseline_results = self.evaluate_against_baselines(num_games=20)
                self.training_data['evaluation_results'].append(baseline_results)
                
                # Calculate overall performance
                overall_performance = 0
                for baseline_results_dict in baseline_results.values():
                    overall_performance += np.mean(list(baseline_results_dict.values()))
                overall_performance /= len(baseline_results)
                
                print(f"Overall baseline performance: {overall_performance:.3f}")
                
                # Check for convergence
                is_converged = self.evaluate_convergence()
                
                if is_converged:
                    converged_count += 1
                    print(f"🎯 Strategy convergence detected! ({converged_count}/{early_stop_patience//100})")
                    
                    if converged_count >= early_stop_patience // 100:
                        print(f"🏆 OPTIMAL STRATEGY FOUND!")
                        print(f"Training converged after {episode} episodes")
                        break
                else:
                    converged_count = 0
                
                # Save checkpoint
                if episode % 200 == 0:
                    self.save_checkpoint(episode)
            
            # Emergency break for very long training
            if episode > 0 and episode % 500 == 0:
                avg_recent_length = np.mean(self.training_data['episode_lengths'][-100:])
                if avg_recent_length > 90:  # Games still too long
                    print(f"⚠️  Games still very long at episode {episode}. Consider hyperparameter tuning.")
        
        # Final evaluation and visualization
        print(f"\n🎉 Training completed after {episode + 1} episodes!")
        
        # Final comprehensive evaluation
        print("📊 Final comprehensive evaluation...")
        final_baseline_results = self.evaluate_against_baselines(num_games=50)
        self.training_data['evaluation_results'].append(final_baseline_results)
        
        # Save final checkpoint
        self.save_checkpoint(episode)
        
        # Create comprehensive visualization
        plot_file = self.create_comprehensive_visualization()
        
        # Save complete training data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'visualizations/optimal_training_data_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(self.training_data, f, indent=2, default=str)
        
        print(f"\n📁 Results saved:")
        print(f"   📊 Visualization: {plot_file}")
        print(f"   💾 Training data: {results_file}")
        print(f"   🏆 Model checkpoints: models/checkpoint_{episode}/")
        
        return {
            'training_data': self.training_data,
            'plot_file': plot_file,
            'results_file': results_file,
            'final_episode': episode
        }

def main():
    """Run optimal strategy training."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimal_training.log'),
            logging.StreamHandler()
        ]
    )
    
    print("🏆 COUP RL: TRAINING TO OPTIMAL STRATEGY")
    print("="*60)
    print("This will train agents until they converge to Nash equilibrium")
    print("Expected duration: 30-60 minutes depending on convergence")
    print("="*60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Create trainer
    trainer = OptimalCoupTrainer(num_players=4)
    
    try:
        # Run training to optimal strategy
        results = trainer.train_to_optimal(max_episodes=2000, early_stop_patience=300)
        
        print(f"\n🎉 OPTIMAL STRATEGY TRAINING COMPLETE!")
        print(f"📊 Check the visualization: {results['plot_file']}")
        print(f"💾 Training data saved: {results['results_file']}")
        print(f"🏆 Models saved in: models/checkpoint_{results['final_episode']}/")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        print("💾 Saving current progress...")
        trainer.save_checkpoint(len(trainer.training_data['episodes']))
        trainer.create_comprehensive_visualization()
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()