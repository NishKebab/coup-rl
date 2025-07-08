#!/usr/bin/env python3
"""Full training run with the FIXED environment - no more 0% win rates!"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from coup.rl.environment import CoupEnvironment
from coup.rl.agent import CoupPPOAgent
from coup.rl.baseline import BaselineStrategies

class FixedCoupTrainer:
    """Trainer using the fixed environment for proper game completion."""
    
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.env = CoupEnvironment(num_players=num_players, max_cycles=100)  # Shorter cycles
        
        # Create agents with optimized hyperparameters
        self.agents = []
        for i in range(num_players):
            agent = CoupPPOAgent(
                obs_size=63, 
                action_size=16, 
                agent_id=f'agent_{i}',
                learning_rate=1e-3,
                gamma=0.98,
                entropy_coef=0.01,
                clip_ratio=0.2
            )
            self.agents.append(agent)
        
        # Training metrics
        self.training_data = {
            'episodes': [],
            'win_rates': {agent.agent_id: [] for agent in self.agents},
            'rewards': {agent.agent_id: [] for agent in self.agents},
            'episode_lengths': [],
            'completion_rates': []
        }
    
    def run_episode(self, episode_num, training=True):
        """Run a single episode with the fixed environment."""
        observations = self.env.reset()
        episode_rewards = {agent.agent_id: 0 for agent in self.agents}
        
        # Reset agent hidden states
        for agent in self.agents:
            agent.reset_hidden_state()
        
        # Agent mapping
        agent_mapping = {self.env.agents[i]: self.agents[i] for i in range(len(self.agents))}
        
        step = 0
        max_steps = 50  # Reasonable limit for fixed environment
        
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
            
            # Get valid actions and ensure agent picks valid one
            valid_actions = self.env.get_valid_actions(current_agent_name)
            
            # Let agent choose, but force valid action
            preferred_action = current_agent.act(observation, game_state, training=training)
            if preferred_action in valid_actions:
                action = preferred_action
            else:
                action = np.random.choice(valid_actions)
            
            self.env.step(action)
            
            # Update observations and rewards
            new_observations = {}
            for agent_name in self.env.agents:
                if not self.env.terminations[agent_name] and not self.env.truncations[agent_name]:
                    new_observations[agent_name] = self.env.observe(agent_name)
            
            reward = self.env.rewards[current_agent_name]
            done = self.env.terminations[current_agent_name] or self.env.truncations[current_agent_name]
            
            # Store experience for training
            if training and len(current_agent.buffer) < 2000:
                next_obs = new_observations.get(current_agent_name, observation)
                current_agent.store_experience(reward, next_obs, done)
            
            episode_rewards[current_agent.agent_id] += reward
            observations.update(new_observations)
            step += 1
            
            # Check for natural game completion
            if self.env.game.state.is_game_over():
                break
        
        # Get winner
        winner = self.env.game.state.get_winner()
        winner_agent = None
        if winner:
            for i, agent in enumerate(self.agents):
                if self.env.game.state.players[i] == winner:
                    winner_agent = agent.agent_id
                    break
        
        # Check if game completed naturally (not timeout)
        completed_naturally = self.env.game.state.is_game_over()
        
        return {
            'episode_rewards': episode_rewards,
            'episode_length': step,
            'winner': winner_agent,
            'completed_naturally': completed_naturally
        }
    
    def train(self, num_episodes=500):
        """Run training with the fixed environment."""
        print(f"🏆 FIXED COUP TRAINING - NO MORE 0% WIN RATES!")
        print(f"Episodes: {num_episodes}")
        print("="*60)
        
        games_completed = 0
        
        for episode in range(num_episodes):
            # Run training episode
            episode_result = self.run_episode(episode, training=True)
            
            # Track completion
            if episode_result['completed_naturally']:
                games_completed += 1
            
            # Store training data
            self.training_data['episodes'].append(episode)
            self.training_data['episode_lengths'].append(episode_result['episode_length'])
            completion_rate = games_completed / (episode + 1)
            self.training_data['completion_rates'].append(completion_rate)
            
            # Update win rates and rewards
            winner = episode_result['winner']
            for agent_id in self.training_data['win_rates']:
                win = 1 if agent_id == winner else 0
                self.training_data['win_rates'][agent_id].append(win)
                self.training_data['rewards'][agent_id].append(episode_result['episode_rewards'][agent_id])
            
            # Update agents every 10 episodes
            if episode % 10 == 0 and episode > 0:
                for agent in self.agents:
                    if len(agent.buffer) >= 32:
                        agent.update_policy(batch_size=64, epochs=4)
            
            # Progress update every 50 episodes
            if episode % 50 == 0 and episode > 0:
                recent_wins = {}
                recent_completion = np.mean(self.training_data['completion_rates'][-50:])
                recent_length = np.mean(self.training_data['episode_lengths'][-50:])
                
                for agent_id in self.training_data['win_rates']:
                    recent_wins[agent_id] = np.mean(self.training_data['win_rates'][agent_id][-50:])
                
                print(f"\\n📊 Episode {episode}:")
                print(f"  Completion rate: {recent_completion:.3f}")
                print(f"  Avg episode length: {recent_length:.1f}")
                print(f"  Win rates: {recent_wins}")
                
                if recent_completion > 0.8:
                    print("  ✅ High completion rate achieved!")
                
                if max(recent_wins.values()) > 0.1:
                    print("  🎯 Non-zero win rates achieved!")
        
        return self.training_data
    
    def create_results_visualization(self):
        """Create comprehensive results visualization."""
        print("\\n📈 Creating results visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fixed Coup RL Training - GAMES ACTUALLY COMPLETE!', fontsize=14, fontweight='bold')
        
        # 1. Win rates over time
        axes[0, 0].set_title('Win Rates Evolution (No More 0%!)')
        for agent_id, wins in self.training_data['win_rates'].items():
            if len(wins) > 20:
                window = min(20, len(wins))
                rolling_avg = np.convolve(wins, np.ones(window)/window, mode='valid')
                x = range(window-1, len(wins))
                axes[0, 0].plot(x, rolling_avg, label=agent_id, linewidth=2)
        
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Game completion rate
        axes[0, 1].set_title('Game Completion Rate')
        completion_rates = self.training_data['completion_rates']
        axes[0, 1].plot(completion_rates, color='green', linewidth=2)
        axes[0, 1].axhline(y=0.8, color='orange', linestyle='--', label='80% Target')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Completion Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # 3. Episode lengths
        axes[1, 0].set_title('Episode Lengths (Much Shorter!)')
        lengths = self.training_data['episode_lengths']
        axes[1, 0].plot(lengths, color='purple', alpha=0.6)
        
        if len(lengths) > 20:
            window = min(20, len(lengths))
            rolling_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
            x = range(window-1, len(lengths))
            axes[1, 0].plot(x, rolling_avg, color='red', linewidth=2, label='Rolling Average')
            axes[1, 0].legend()
        
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Episode Length')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Final win distribution
        axes[1, 1].set_title('Total Wins Distribution')
        total_wins = [np.sum(wins) for wins in self.training_data['win_rates'].values()]
        agent_names = list(self.training_data['win_rates'].keys())
        
        bars = axes[1, 1].bar(agent_names, total_wins, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
        axes[1, 1].set_ylabel('Total Wins')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, wins in zip(bars, total_wins):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{wins}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('visualizations', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = f'visualizations/fixed_training_results_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plot_file

def main():
    """Run the full training with fixed environment."""
    print("🚀 STARTING FULL TRAINING WITH FIXED ENVIRONMENT")
    print("="*60)
    print("Expected improvements:")
    print("✅ Games will complete with actual winners")
    print("✅ Win rates will be non-zero")
    print("✅ Episodes will be shorter and more meaningful")
    print("✅ Agents will learn effective strategies")
    print("="*60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    try:
        # Create trainer with fixed environment
        trainer = FixedCoupTrainer(num_players=4)
        
        # Run training
        training_data = trainer.train(num_episodes=1000)
        
        # Create visualization
        plot_file = trainer.create_results_visualization()
        
        # Save training data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'visualizations/fixed_training_data_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        # Final results summary
        final_completion_rate = np.mean(training_data['completion_rates'][-50:])
        final_win_rates = {}
        for agent_id, wins in training_data['win_rates'].items():
            final_win_rates[agent_id] = np.mean(wins[-50:]) if wins else 0.0
        
        avg_episode_length = np.mean(training_data['episode_lengths'][-50:])
        
        print(f"\\n🎉 TRAINING COMPLETED!")
        print(f"📊 Final Results:")
        print(f"  Game completion rate: {final_completion_rate:.3f}")
        print(f"  Average episode length: {avg_episode_length:.1f}")
        print(f"  Final win rates:")
        for agent_id, rate in final_win_rates.items():
            print(f"    {agent_id}: {rate:.3f}")
        
        print(f"\\n📁 Results saved:")
        print(f"  📊 Visualization: {plot_file}")
        print(f"  💾 Training data: {results_file}")
        
        # Check if we achieved the fix
        max_win_rate = max(final_win_rates.values())
        if max_win_rate > 0.1:
            print(f"\\n🏆 SUCCESS! Win rates are now non-zero (max: {max_win_rate:.3f})")
            print("The 0% win rate issue has been SOLVED! 🎯")
        else:
            print(f"\\n⚠️  Win rates still low, may need more training")
        
        if final_completion_rate > 0.8:
            print(f"✅ High completion rate achieved: {final_completion_rate:.3f}")
        
    except KeyboardInterrupt:
        print(f"\\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()