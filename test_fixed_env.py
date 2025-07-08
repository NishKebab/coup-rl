#!/usr/bin/env python3
"""Quick test of the fixed environment."""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
from coup.rl.environment import CoupEnvironment
from coup.rl.agent import CoupPPOAgent

def test_fixed_environment():
    """Test that the fixed environment produces winners."""
    print("🔧 Testing Fixed Environment")
    print("="*40)
    
    env = CoupEnvironment(num_players=4, max_cycles=50)
    agents = []
    
    for i in range(4):
        agent = CoupPPOAgent(obs_size=63, action_size=16, agent_id=f'agent_{i}')
        agents.append(agent)
    
    wins = 0
    total_games = 10
    
    print(f"🎮 Running {total_games} test games...")
    
    for game_num in range(total_games):
        observations = env.reset()
        
        # Reset agent hidden states
        for agent in agents:
            agent.reset_hidden_state()
        
        # Agent mapping
        agent_mapping = {env.agents[i]: agents[i] for i in range(len(agents))}
        
        step = 0
        max_steps = 30  # Short test
        
        while (not all(env.terminations.values()) and 
               not all(env.truncations.values()) and 
               step < max_steps):
            
            current_agent_name = env.agent_selection
            current_agent = agent_mapping[current_agent_name]
            
            if env.terminations[current_agent_name] or env.truncations[current_agent_name]:
                env.step(None)
                continue
            
            # Get valid actions
            valid_actions = env.get_valid_actions(current_agent_name)
            
            # Force valid action selection
            action = np.random.choice(valid_actions)
            
            # Execute action
            env.step(action)
            
            # Update observations
            new_observations = {}
            for agent_name in env.agents:
                if not env.terminations[agent_name] and not env.truncations[agent_name]:
                    new_observations[agent_name] = env.observe(agent_name)
            
            observations.update(new_observations)
            step += 1
            
            # Check for game over
            if env.game.state.is_game_over():
                winner = env.game.state.get_winner()
                if winner:
                    wins += 1
                    print(f"  Game {game_num + 1}: Winner found in {step} steps! 🏆")
                break
        
        if step >= max_steps:
            print(f"  Game {game_num + 1}: Timeout after {step} steps")
    
    print(f"\\n📊 Test Results:")
    print(f"  Games with winners: {wins}/{total_games} ({wins/total_games*100:.1f}%)")
    
    if wins > 0:
        print(f"  🎉 SUCCESS! Non-zero win rate achieved!")
        print(f"  ✅ The 0% win rate issue is FIXED!")
    else:
        print(f"  ❌ Still no winners - may need more investigation")
    
    return wins > 0

def main():
    """Run the test."""
    print("🎯 TESTING FIXED ENVIRONMENT")
    print("="*50)
    
    try:
        success = test_fixed_environment()
        
        if success:
            print("\\n🏆 Environment fix validated!")
            print("Ready for full training run!")
        else:
            print("\\n⚠️  Environment may need further fixes")
            
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()