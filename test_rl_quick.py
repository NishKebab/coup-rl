"""Quick test of the RL training system."""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import torch
from coup.rl.environment import CoupEnvironment
from coup.rl.agent import CoupPPOAgent
from coup.rl.baseline import BaselineStrategies

def test_environment():
    """Test basic environment functionality."""
    print("Testing Environment...")
    
    env = CoupEnvironment(num_players=4)
    observations = env.reset()
    
    print(f"✅ Environment created with {len(env.agents)} agents")
    print(f"✅ Observation space: {env.observation_spaces[env.agents[0]].shape}")
    print(f"✅ Action space: {env.action_spaces[env.agents[0]].n}")
    
    # Test a few steps
    for step in range(5):
        agent_id = env.agent_selection
        if not env.terminations[agent_id] and not env.truncations[agent_id]:
            action = np.random.choice(16)  # Random action
            env.step(action)
            
            if env.agent_selection in observations:
                new_obs = env.observe(env.agent_selection)
                print(f"Step {step}: Agent {agent_id} took action {action}")
    
    print("✅ Environment test completed\n")

def test_agents():
    """Test agent functionality."""
    print("Testing Agents...")
    
    # Test RL agent
    agent = CoupPPOAgent(obs_size=63, action_size=16, agent_id='test_rl_agent')
    obs = np.random.rand(63)
    game_state = {'challenge_phase': False, 'block_phase': False}
    
    action = agent.act(obs, game_state, training=False)
    print(f"✅ RL Agent action: {action}")
    
    # Test baseline agents
    baselines = BaselineStrategies()
    for strategy in ['random', 'honest', 'aggressive', 'defensive']:
        baseline_agent = baselines.create_baseline_agent(strategy, f'test_{strategy}')
        action = baseline_agent.act(obs, game_state, training=False)
        print(f"✅ {strategy.capitalize()} agent action: {action}")
    
    print("✅ Agent test completed\n")

def test_training_episode():
    """Test a single training episode."""
    print("Testing Training Episode...")
    
    # Create environment and agents
    env = CoupEnvironment(num_players=4)
    agents = []
    
    for i in range(4):
        agent = CoupPPOAgent(obs_size=63, action_size=16, agent_id=f'agent_{i}')
        agents.append(agent)
    
    # Run one episode
    observations = env.reset()
    episode_length = 0
    
    # Reset agent hidden states
    for agent in agents:
        agent.reset_hidden_state()
    
    # Agent mapping
    agent_mapping = {env.agents[i]: agents[i] for i in range(len(agents))}
    
    print("Starting episode...")
    
    max_steps = 100  # Prevent infinite loops
    step = 0
    
    while (not all(env.terminations.values()) and 
           not all(env.truncations.values()) and 
           step < max_steps):
        
        current_agent_name = env.agent_selection
        current_agent = agent_mapping[current_agent_name]
        
        # Skip if agent is terminated
        if env.terminations[current_agent_name] or env.truncations[current_agent_name]:
            env.step(None)
            continue
        
        # Get observation and game state
        observation = observations.get(current_agent_name, np.zeros(63))
        game_state = {
            'challenge_phase': env.challenge_phase,
            'block_phase': env.block_phase,
            'current_player': current_agent_name,
            'phase': env.game.state.phase.value
        }
        
        # Select action
        action = current_agent.act(observation, game_state, training=False)
        
        # Execute action
        env.step(action)
        
        # Update observations
        new_observations = {}
        for agent_name in env.agents:
            if not env.terminations[agent_name] and not env.truncations[agent_name]:
                new_observations[agent_name] = env.observe(agent_name)
        
        observations.update(new_observations)
        episode_length += 1
        step += 1
        
        if step % 20 == 0:
            print(f"  Step {step}, Current player: {current_agent_name}, Action: {action}")
    
    # Get winner
    winner = env.game.state.get_winner()
    winner_name = None
    if winner:
        for i, agent in enumerate(agents):
            if env.game.state.players[i] == winner:
                winner_name = agent.agent_id
                break
    
    print(f"✅ Episode completed in {episode_length} steps")
    print(f"✅ Winner: {winner_name if winner_name else 'No winner'}")
    print("✅ Training episode test completed\n")

def main():
    """Run all tests."""
    print("🎯 Starting Coup RL System Tests")
    print("=" * 50)
    
    try:
        test_environment()
        test_agents()
        test_training_episode()
        
        print("🎉 All tests passed! The RL system is ready for training.")
        print("\nNext steps:")
        print("1. Run: python examples/train_coup_agents.py")
        print("2. Or create a custom training script")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()