"""Demo training script for Coup RL agents."""

import sys
import os
sys.path.insert(0, 'src')

import logging
import numpy as np
from typing import Dict, Any

from coup.rl.environment import CoupEnvironment
from coup.rl.agent import CoupPPOAgent
from coup.rl.baseline import BaselineStrategies

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCoupTrainer:
    """Simplified trainer for demonstration."""
    
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.env = CoupEnvironment(num_players=num_players)
        
        # Create agents
        self.agents = []
        for i in range(num_players):
            agent = CoupPPOAgent(
                obs_size=63, 
                action_size=16, 
                agent_id=f'agent_{i}',
                learning_rate=1e-3
            )
            self.agents.append(agent)
        
        # Baseline strategies for comparison
        self.baselines = BaselineStrategies()
        
        # Training metrics
        self.episode_rewards = []
        self.win_rates = {agent.agent_id: [] for agent in self.agents}
        
    def run_episode(self, training=True):
        """Run a single episode."""
        observations = self.env.reset()
        episode_rewards = {agent.agent_id: 0 for agent in self.agents}
        episode_length = 0
        
        # Reset agent hidden states
        for agent in self.agents:
            agent.reset_hidden_state()
        
        # Agent mapping
        agent_mapping = {self.env.agents[i]: self.agents[i] for i in range(len(self.agents))}
        
        max_steps = 200  # Prevent infinite loops
        step = 0
        
        while (not all(self.env.terminations.values()) and 
               not all(self.env.truncations.values()) and 
               step < max_steps):
            
            current_agent_name = self.env.agent_selection
            current_agent = agent_mapping[current_agent_name]
            
            # Skip if agent is terminated
            if self.env.terminations[current_agent_name] or self.env.truncations[current_agent_name]:
                self.env.step(None)
                continue
            
            # Get observation and game state
            observation = observations.get(current_agent_name, np.zeros(63))
            game_state = {
                'challenge_phase': self.env.challenge_phase,
                'block_phase': self.env.block_phase,
                'current_player': current_agent_name,
                'phase': self.env.game.state.phase.value
            }
            
            # Select action
            action = current_agent.act(observation, game_state, training=training)
            
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
            'episode_length': episode_length,
            'winner': winner_agent,
            'step_count': step
        }
    
    def update_agents(self):
        """Update all agents."""
        update_results = {}
        for agent in self.agents:
            results = agent.update_policy(batch_size=32, epochs=5)
            if results:
                update_results[agent.agent_id] = results
        return update_results
    
    def evaluate_against_baseline(self, baseline_strategy='random', num_episodes=10):
        """Evaluate agents against baseline strategy."""
        baseline_agent = self.baselines.create_baseline_agent(baseline_strategy, 'baseline')
        
        wins = {agent.agent_id: 0 for agent in self.agents}
        
        for episode in range(num_episodes):
            # Replace one agent with baseline
            test_agents = self.agents[:-1] + [baseline_agent]
            
            # Run episode (non-training)
            observations = self.env.reset()
            
            # Reset agent hidden states
            for agent in test_agents:
                agent.reset_hidden_state()
            
            # Agent mapping
            agent_mapping = {self.env.agents[i]: test_agents[i] for i in range(len(test_agents))}
            
            max_steps = 200
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
                
                action = current_agent.act(observation, game_state, training=False)
                self.env.step(action)
                
                new_observations = {}
                for agent_name in self.env.agents:
                    if not self.env.terminations[agent_name] and not self.env.truncations[agent_name]:
                        new_observations[agent_name] = self.env.observe(agent_name)
                
                observations.update(new_observations)
                step += 1
            
            # Check winner
            winner = self.env.game.state.get_winner()
            if winner:
                for i, agent in enumerate(test_agents):
                    if self.env.game.state.players[i] == winner and hasattr(agent, 'agent_id'):
                        if agent.agent_id in wins:
                            wins[agent.agent_id] += 1
                        break
        
        # Calculate win rates
        win_rates = {agent_id: wins[agent_id] / num_episodes for agent_id in wins}
        return win_rates
    
    def train(self, num_episodes=100, update_frequency=10, eval_frequency=25):
        """Train the agents."""
        logger.info(f"Starting training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            # Run training episode
            results = self.run_episode(training=True)
            
            # Store metrics
            self.episode_rewards.append(results['episode_rewards'])
            
            # Update win rates
            winner = results['winner']
            for agent_id in self.win_rates:
                win = 1 if agent_id == winner else 0
                self.win_rates[agent_id].append(win)
            
            # Update agents periodically
            if episode % update_frequency == 0 and episode > 0:
                update_results = self.update_agents()
                logger.info(f"Episode {episode}: Updated agents")
                
                # Log recent performance
                recent_rewards = {}
                recent_wins = {}
                window = min(20, episode)
                
                for agent_id in self.win_rates:
                    recent_reward_list = [r[agent_id] for r in self.episode_rewards[-window:]]
                    recent_win_list = self.win_rates[agent_id][-window:]
                    
                    recent_rewards[agent_id] = np.mean(recent_reward_list)
                    recent_wins[agent_id] = np.mean(recent_win_list)
                
                logger.info(f"Recent performance (last {window} episodes):")
                for agent_id in recent_rewards:
                    logger.info(f"  {agent_id}: Win Rate: {recent_wins[agent_id]:.3f}, Avg Reward: {recent_rewards[agent_id]:.3f}")
            
            # Evaluate against baseline
            if episode % eval_frequency == 0 and episode > 0:
                baseline_results = self.evaluate_against_baseline('random', num_episodes=5)
                logger.info(f"Episode {episode}: Evaluation vs Random baseline:")
                for agent_id, win_rate in baseline_results.items():
                    logger.info(f"  {agent_id}: {win_rate:.3f} win rate vs random")
            
            # Progress logging
            if episode % 10 == 0:
                avg_length = results['episode_length']
                logger.info(f"Episode {episode}: Length: {avg_length}, Winner: {winner or 'None'}")
        
        logger.info("Training completed!")
        
        # Final evaluation
        logger.info("\n" + "="*50)
        logger.info("FINAL EVALUATION")
        logger.info("="*50)
        
        # Self-play performance
        final_window = min(50, len(self.episode_rewards))
        logger.info(f"\nSelf-play performance (last {final_window} episodes):")
        
        for agent_id in self.win_rates:
            final_win_rate = np.mean(self.win_rates[agent_id][-final_window:])
            final_rewards = [r[agent_id] for r in self.episode_rewards[-final_window:]]
            final_avg_reward = np.mean(final_rewards)
            
            logger.info(f"  {agent_id}:")
            logger.info(f"    Win Rate: {final_win_rate:.3f}")
            logger.info(f"    Avg Reward: {final_avg_reward:.3f}")
        
        # Baseline comparisons
        logger.info("\nPerformance vs Baselines:")
        for baseline in ['random', 'honest', 'aggressive', 'defensive']:
            try:
                baseline_results = self.evaluate_against_baseline(baseline, num_episodes=20)
                logger.info(f"\n  vs {baseline}:")
                for agent_id, win_rate in baseline_results.items():
                    logger.info(f"    {agent_id}: {win_rate:.3f}")
            except Exception as e:
                logger.error(f"Error evaluating vs {baseline}: {e}")

def main():
    """Main training function."""
    print("🎯 Starting Coup RL Agent Demo Training")
    print("="*50)
    
    # Create trainer
    trainer = SimpleCoupTrainer(num_players=4)
    
    # Train agents
    try:
        trainer.train(num_episodes=200, update_frequency=10, eval_frequency=50)
        print("\n🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()