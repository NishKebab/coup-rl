"""Debug script to identify why games don't complete properly."""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
from coup.rl.environment import CoupEnvironment
from coup.rl.agent import CoupPPOAgent

def debug_single_episode():
    """Debug a single episode to see what's happening."""
    print("🔍 Debugging Game Completion Issues")
    print("="*50)
    
    # Create environment and agents
    env = CoupEnvironment(num_players=4)
    agents = []
    
    for i in range(4):
        agent = CoupPPOAgent(obs_size=63, action_size=16, agent_id=f'agent_{i}')
        agents.append(agent)
    
    # Run detailed episode
    observations = env.reset()
    
    # Reset agent hidden states
    for agent in agents:
        agent.reset_hidden_state()
    
    # Agent mapping
    agent_mapping = {env.agents[i]: agents[i] for i in range(len(agents))}
    
    print(f"🎮 Starting episode with {len(agents)} agents")
    print(f"Initial game state: {env.game.state.phase}")
    
    # Check initial player states
    for i, player in enumerate(env.game.state.players):
        print(f"Player {i}: {player.coins} coins, {len(player.cards)} cards, eliminated: {player.is_eliminated}")
    
    step = 0
    max_steps = 50  # Short debug run
    
    while (not all(env.terminations.values()) and 
           not all(env.truncations.values()) and 
           step < max_steps):
        
        current_agent_name = env.agent_selection
        current_agent = agent_mapping[current_agent_name]
        
        print(f"\nStep {step}:")
        print(f"  Current player: {current_agent_name}")
        print(f"  Challenge phase: {env.challenge_phase}")
        print(f"  Block phase: {env.block_phase}")
        
        # Check termination states
        print(f"  Terminations: {env.terminations}")
        print(f"  Truncations: {env.truncations}")
        
        if env.terminations[current_agent_name] or env.truncations[current_agent_name]:
            print(f"  Agent {current_agent_name} is terminated/truncated, skipping")
            env.step(None)
            continue
        
        # Check player states
        for i, player in enumerate(env.game.state.players):
            agent_name = env.agents[i]
            print(f"  Player {i} ({agent_name}): {player.coins} coins, {len(player.cards)} cards, eliminated: {player.is_eliminated}")
        
        # Get observation
        observation = observations.get(current_agent_name, np.zeros(63))
        game_state = {
            'challenge_phase': env.challenge_phase,
            'block_phase': env.block_phase,
            'current_player': current_agent_name,
            'phase': env.game.state.phase.value
        }
        
        # Get action
        action = current_agent.act(observation, game_state, training=False)
        print(f"  Action taken: {action}")
        
        # Execute action
        env.step(action)
        
        # Check what happened
        print(f"  After action - Challenge phase: {env.challenge_phase}, Block phase: {env.block_phase}")
        print(f"  Game phase: {env.game.state.phase}")
        
        # Update observations
        new_observations = {}
        for agent_name in env.agents:
            if not env.terminations[agent_name] and not env.truncations[agent_name]:
                new_observations[agent_name] = env.observe(agent_name)
        
        observations.update(new_observations)
        step += 1
        
        # Check for game over
        if env.game.state.is_game_over():
            print(f"  🎯 GAME OVER detected!")
            winner = env.game.state.get_winner()
            if winner:
                print(f"  🏆 Winner: {winner.name}")
            else:
                print(f"  ❌ No winner found")
            break
    
    print(f"\n📊 Episode Summary:")
    print(f"  Steps taken: {step}")
    print(f"  Max steps: {max_steps}")
    print(f"  Game over: {env.game.state.is_game_over()}")
    
    # Final player states
    print(f"\n👥 Final Player States:")
    for i, player in enumerate(env.game.state.players):
        agent_name = env.agents[i]
        print(f"  Player {i} ({agent_name}): {player.coins} coins, {len(player.cards)} cards, eliminated: {player.is_eliminated}")
    
    # Check why game didn't end
    active_players = [p for p in env.game.state.players if not p.is_eliminated]
    print(f"\n🔍 Active players remaining: {len(active_players)}")
    
    if len(active_players) > 1:
        print("❌ Game should continue - multiple players still active")
        
        # Check if players actually have cards
        for player in active_players:
            if len(player.cards) == 0:
                print(f"⚠️  Player {player.name} has 0 cards but is not eliminated!")
    
    elif len(active_players) == 1:
        print(f"🎯 Game should end - winner: {active_players[0].name}")
    
    else:
        print("❌ No active players - this shouldn't happen!")

def test_card_loss_mechanism():
    """Test the card loss mechanism specifically."""
    print("\n🔍 Testing Card Loss Mechanism")
    print("="*40)
    
    # Create a simple game state
    from coup.game import CoupGame
    from coup.types import Character
    from coup.card import Card
    
    game = CoupGame()
    game.add_player("TestPlayer1")
    game.add_player("TestPlayer2")
    game.start_game()
    
    player1 = game.state.players[0]
    player2 = game.state.players[1]
    
    print(f"Player 1 initial: {len(player1.cards)} cards, eliminated: {player1.is_eliminated}")
    print(f"Player 2 initial: {len(player2.cards)} cards, eliminated: {player2.is_eliminated}")
    
    # Test losing one card
    if len(player1.cards) > 0:
        removed_card = player1.lose_card_by_choice(0)
        print(f"Player 1 after losing 1 card: {len(player1.cards)} cards, eliminated: {player1.is_eliminated}")
        print(f"Removed card: {removed_card}")
    
    # Test losing second card
    if len(player1.cards) > 0:
        removed_card = player1.lose_card_by_choice(0)
        print(f"Player 1 after losing 2nd card: {len(player1.cards)} cards, eliminated: {player1.is_eliminated}")
        print(f"Removed card: {removed_card}")
    
    # Test game over condition
    print(f"Game over: {game.state.is_game_over()}")
    winner = game.state.get_winner()
    if winner:
        print(f"Winner: {winner.name}")

def main():
    """Run debugging tests."""
    try:
        debug_single_episode()
        test_card_loss_mechanism()
        
        print(f"\n🎯 Key Issues Found:")
        print(f"1. Check if coup/assassinate actions actually cause card loss")
        print(f"2. Verify game over detection works when players reach 0 cards")
        print(f"3. Ensure environment properly updates termination states")
        print(f"4. Check if challenge/block phases get stuck in loops")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()