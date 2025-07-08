"""Quick evaluation and visualization script."""

import sys
import os
sys.path.insert(0, 'src')

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from coup.rl.environment import CoupEnvironment
from coup.rl.agent import CoupPPOAgent
from coup.rl.baseline import BaselineStrategies

def quick_training_and_eval(num_episodes=100):
    """Run quick training and evaluation."""
    print("🎯 Quick Training and Evaluation")
    print("="*50)
    
    # Create environment and agents
    env = CoupEnvironment(num_players=4)
    agents = []
    
    for i in range(4):
        agent = CoupPPOAgent(
            obs_size=63, 
            action_size=16, 
            agent_id=f'agent_{i}',
            learning_rate=1e-3
        )
        agents.append(agent)
    
    # Training metrics
    episode_data = []
    win_rates = {agent.agent_id: [] for agent in agents}
    action_distributions = {agent.agent_id: [] for agent in agents}
    
    print(f"🚀 Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Run episode
        observations = env.reset()
        episode_rewards = {agent.agent_id: 0 for agent in agents}
        episode_actions = {agent.agent_id: [] for agent in agents}
        
        # Reset agent hidden states
        for agent in agents:
            agent.reset_hidden_state()
        
        # Agent mapping
        agent_mapping = {env.agents[i]: agents[i] for i in range(len(agents))}
        
        max_steps = 100  # Shorter episodes for speed
        step = 0
        
        while (not all(env.terminations.values()) and 
               not all(env.truncations.values()) and 
               step < max_steps):
            
            current_agent_name = env.agent_selection
            current_agent = agent_mapping[current_agent_name]
            
            if env.terminations[current_agent_name] or env.truncations[current_agent_name]:
                env.step(None)
                continue
            
            observation = observations.get(current_agent_name, np.zeros(63))
            game_state = {
                'challenge_phase': env.challenge_phase,
                'block_phase': env.block_phase,
                'current_player': current_agent_name,
                'phase': env.game.state.phase.value
            }
            
            action = current_agent.act(observation, game_state, training=True)
            episode_actions[current_agent.agent_id].append(action)
            
            env.step(action)
            
            new_observations = {}
            for agent_name in env.agents:
                if not env.terminations[agent_name] and not env.truncations[agent_name]:
                    new_observations[agent_name] = env.observe(agent_name)
            
            reward = env.rewards[current_agent_name]
            done = env.terminations[current_agent_name] or env.truncations[current_agent_name]
            
            # Store experience
            next_obs = new_observations.get(current_agent_name, observation)
            current_agent.store_experience(reward, next_obs, done)
            
            episode_rewards[current_agent.agent_id] += reward
            observations.update(new_observations)
            step += 1
        
        # Get winner
        winner = env.game.state.get_winner()
        winner_agent = None
        if winner:
            for i, agent in enumerate(agents):
                if env.game.state.players[i] == winner:
                    winner_agent = agent.agent_id
                    break
        
        # Store episode data
        episode_data.append({
            'episode': episode,
            'rewards': episode_rewards,
            'winner': winner_agent,
            'length': step,
            'actions': episode_actions
        })
        
        # Update win rates
        for agent_id in win_rates:
            win = 1 if agent_id == winner_agent else 0
            win_rates[agent_id].append(win)
        
        # Update agents every 10 episodes
        if episode % 10 == 0 and episode > 0:
            for agent in agents:
                agent.update_policy(batch_size=min(32, len(agent.buffer)), epochs=3)
        
        if episode % 20 == 0:
            recent_wins = {aid: np.mean(win_rates[aid][-20:]) for aid in win_rates}
            recent_rewards = {aid: np.mean([ep['rewards'][aid] for ep in episode_data[-20:]]) for aid in win_rates}
            print(f"Episode {episode}: Win rates: {recent_wins}")
    
    return episode_data, win_rates, agents

def evaluate_against_baselines(agents, num_games=20):
    """Evaluate trained agents against baselines."""
    print("\n📊 Evaluating against baselines...")
    
    baselines = BaselineStrategies()
    results = {}
    
    for baseline_name in ['random', 'honest', 'aggressive', 'defensive']:
        print(f"  Testing vs {baseline_name}...")
        baseline_agent = baselines.create_baseline_agent(baseline_name, 'baseline')
        
        wins = {agent.agent_id: 0 for agent in agents}
        
        for game in range(num_games):
            env = CoupEnvironment(num_players=4)
            test_agents = agents[:3] + [baseline_agent]  # Replace one agent with baseline
            
            observations = env.reset()
            
            for agent in test_agents:
                agent.reset_hidden_state()
            
            agent_mapping = {env.agents[i]: test_agents[i] for i in range(len(test_agents))}
            
            max_steps = 100
            step = 0
            
            while (not all(env.terminations.values()) and 
                   not all(env.truncations.values()) and 
                   step < max_steps):
                
                current_agent_name = env.agent_selection
                current_agent = agent_mapping[current_agent_name]
                
                if env.terminations[current_agent_name] or env.truncations[current_agent_name]:
                    env.step(None)
                    continue
                
                observation = observations.get(current_agent_name, np.zeros(63))
                game_state = {
                    'challenge_phase': env.challenge_phase,
                    'block_phase': env.block_phase,
                    'current_player': current_agent_name,
                    'phase': env.game.state.phase.value
                }
                
                action = current_agent.act(observation, game_state, training=False)
                env.step(action)
                
                new_observations = {}
                for agent_name in env.agents:
                    if not env.terminations[agent_name] and not env.truncations[agent_name]:
                        new_observations[agent_name] = env.observe(agent_name)
                
                observations.update(new_observations)
                step += 1
            
            # Check winner
            winner = env.game.state.get_winner()
            if winner:
                for i, agent in enumerate(test_agents):
                    if env.game.state.players[i] == winner and hasattr(agent, 'agent_id'):
                        if agent.agent_id in wins:
                            wins[agent.agent_id] += 1
                        break
        
        # Calculate win rates
        win_rates = {agent_id: wins[agent_id] / num_games for agent_id in wins}
        results[baseline_name] = win_rates
        
        print(f"    Results: {win_rates}")
    
    return results

def create_visualizations(episode_data, win_rates, baseline_results):
    """Create comprehensive visualizations."""
    print("\n📈 Creating visualizations...")
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Coup RL Training Results', fontsize=16, fontweight='bold')
    
    # 1. Win rates over time
    axes[0, 0].set_title('Win Rates Over Time')
    for agent_id, rates in win_rates.items():
        if len(rates) > 10:
            # Calculate rolling average
            window = min(20, len(rates))
            rolling_avg = np.convolve(rates, np.ones(window)/window, mode='valid')
            x = range(window-1, len(rates))
            axes[0, 0].plot(x, rolling_avg, label=agent_id, linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Win Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Episode rewards
    axes[0, 1].set_title('Average Episode Rewards')
    agent_ids = list(win_rates.keys())
    for agent_id in agent_ids:
        rewards = [ep['rewards'][agent_id] for ep in episode_data]
        if len(rewards) > 10:
            window = min(20, len(rewards))
            rolling_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            x = range(window-1, len(rewards))
            axes[0, 1].plot(x, rolling_avg, label=agent_id, linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Episode lengths
    axes[0, 2].set_title('Episode Lengths')
    lengths = [ep['length'] for ep in episode_data]
    if len(lengths) > 10:
        window = min(20, len(lengths))
        rolling_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        x = range(window-1, len(lengths))
        axes[0, 2].plot(x, rolling_avg, color='purple', linewidth=2)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Episode Length')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Action distributions
    axes[1, 0].set_title('Action Distributions')
    action_counts = {agent_id: [0] * 16 for agent_id in agent_ids}
    
    for ep in episode_data[-20:]:  # Last 20 episodes
        for agent_id, actions in ep['actions'].items():
            for action in actions:
                if action < 16:
                    action_counts[agent_id][action] += 1
    
    # Plot as stacked bar chart
    x_pos = np.arange(16)
    bottom = np.zeros(16)
    
    for i, agent_id in enumerate(agent_ids):
        total = sum(action_counts[agent_id])
        if total > 0:
            proportions = [count / total for count in action_counts[agent_id]]
            axes[1, 0].bar(x_pos, proportions, bottom=bottom, label=agent_id, alpha=0.7)
            bottom = [b + p for b, p in zip(bottom, proportions)]
    
    axes[1, 0].set_xlabel('Action')
    axes[1, 0].set_ylabel('Proportion')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Baseline comparison
    axes[1, 1].set_title('Performance vs Baselines')
    baseline_names = list(baseline_results.keys())
    agent_baseline_scores = {agent_id: [] for agent_id in agent_ids}
    
    for baseline_name, results in baseline_results.items():
        for agent_id in agent_ids:
            agent_baseline_scores[agent_id].append(results.get(agent_id, 0))
    
    x_pos = np.arange(len(baseline_names))
    width = 0.2
    
    for i, agent_id in enumerate(agent_ids):
        offset = (i - len(agent_ids)/2) * width
        axes[1, 1].bar(x_pos + offset, agent_baseline_scores[agent_id], 
                       width, label=agent_id, alpha=0.8)
    
    axes[1, 1].set_xlabel('Baseline Strategy')
    axes[1, 1].set_ylabel('Win Rate')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(baseline_names, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Final performance summary
    axes[1, 2].set_title('Final Performance Summary')
    final_win_rates = [np.mean(win_rates[agent_id][-20:]) for agent_id in agent_ids]
    final_rewards = [np.mean([ep['rewards'][agent_id] for ep in episode_data[-20:]]) for agent_id in agent_ids]
    
    scatter = axes[1, 2].scatter(final_win_rates, final_rewards, 
                                s=100, alpha=0.7, c=range(len(agent_ids)), cmap='viridis')
    
    for i, agent_id in enumerate(agent_ids):
        axes[1, 2].annotate(agent_id, (final_win_rates[i], final_rewards[i]), 
                           xytext=(5, 5), textcoords='offset points')
    
    axes[1, 2].set_xlabel('Final Win Rate')
    axes[1, 2].set_ylabel('Final Average Reward')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f'visualizations/coup_training_results_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {plot_filename}")
    
    # Also show the plot
    plt.show()
    
    return plot_filename

def save_results(episode_data, win_rates, baseline_results):
    """Save all results to JSON files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare data for JSON serialization
    results = {
        'timestamp': timestamp,
        'episode_data': episode_data,
        'win_rates': {k: list(v) for k, v in win_rates.items()},
        'baseline_results': baseline_results,
        'summary': {
            'total_episodes': len(episode_data),
            'final_win_rates': {k: float(np.mean(v[-20:])) if len(v) >= 20 else float(np.mean(v)) 
                               for k, v in win_rates.items()},
            'final_avg_rewards': {k: float(np.mean([ep['rewards'][k] for ep in episode_data[-20:]])) 
                                 for k in win_rates.keys()},
            'avg_episode_length': float(np.mean([ep['length'] for ep in episode_data]))
        }
    }
    
    # Save results
    results_filename = f'visualizations/coup_results_{timestamp}.json'
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {results_filename}")
    
    # Print summary
    print(f"\n📋 TRAINING SUMMARY")
    print("="*50)
    print(f"Total Episodes: {results['summary']['total_episodes']}")
    print(f"Avg Episode Length: {results['summary']['avg_episode_length']:.1f}")
    print(f"\nFinal Performance:")
    for agent_id, win_rate in results['summary']['final_win_rates'].items():
        avg_reward = results['summary']['final_avg_rewards'][agent_id]
        print(f"  {agent_id}: Win Rate: {win_rate:.3f}, Avg Reward: {avg_reward:.2f}")
    
    print(f"\nBaseline Comparison:")
    for baseline, agent_results in baseline_results.items():
        print(f"  vs {baseline}:")
        for agent_id, win_rate in agent_results.items():
            print(f"    {agent_id}: {win_rate:.3f}")
    
    return results_filename

def main():
    """Main evaluation function."""
    print("🎮 Coup RL Quick Evaluation & Visualization")
    print("="*60)
    
    try:
        # Run training
        episode_data, win_rates, agents = quick_training_and_eval(num_episodes=150)
        
        # Evaluate against baselines
        baseline_results = evaluate_against_baselines(agents, num_games=15)
        
        # Create visualizations
        plot_file = create_visualizations(episode_data, win_rates, baseline_results)
        
        # Save results
        results_file = save_results(episode_data, win_rates, baseline_results)
        
        print(f"\n🎉 Evaluation Complete!")
        print(f"📁 Files generated:")
        print(f"   📊 Visualization: {plot_file}")
        print(f"   💾 Results data: {results_file}")
        print(f"   📂 Directory: visualizations/")
        
        print(f"\n💡 To view results:")
        print(f"   - Open the PNG file to see training progress")
        print(f"   - Check the JSON file for detailed metrics")
        print(f"   - Run longer training for better convergence")
        
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()