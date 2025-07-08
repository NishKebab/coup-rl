"""Example script for evaluating trained Coup agents."""

import os
import sys
import json
import numpy as np
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from coup.rl.training import CoupTrainer
from coup.rl.baseline import BaselineStrategies
from coup.rl.agent import CoupPPOAgent
from coup.rl.environment import CoupEnvironment


def evaluate_against_baselines(model_dir: str, num_episodes: int = 200) -> Dict[str, Any]:
    """Evaluate trained agents against baseline strategies."""
    print("Evaluating agents against baseline strategies...")
    
    # Create trainer and load models
    config = {'trainer_args': {'num_players': 4}}
    trainer = CoupTrainer(config)
    
    # Load checkpoint
    evaluation_results = trainer.evaluate(model_dir, num_episodes)
    
    return evaluation_results


def tournament_evaluation(model_dirs: list, num_episodes: int = 100) -> Dict[str, Any]:
    """Run tournament between different trained models."""
    print("Running tournament evaluation...")
    
    results = {
        'tournament_results': {},
        'head_to_head': {},
        'skill_rankings': {}
    }
    
    # Load all models
    agents = {}
    for i, model_dir in enumerate(model_dirs):
        agent_id = f"model_{i}"
        # Load agent from checkpoint
        # This would require implementing model loading logic
        print(f"Loading model from {model_dir}")
    
    # Run tournament games
    # Implementation would depend on specific tournament format
    
    return results


def analyze_strategy_diversity(model_dir: str) -> Dict[str, Any]:
    """Analyze strategy diversity of trained agents."""
    print("Analyzing strategy diversity...")
    
    # Load training data
    data_file = os.path.join(model_dir, "training_data.json")
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            training_data = json.load(f)
    else:
        print("No training data found")
        return {}
    
    strategy_data = training_data.get('strategy_data', {})
    
    diversity_metrics = {}
    
    for agent_id, actions in strategy_data.items():
        # Calculate action distribution
        total_actions = 0
        action_counts = {}
        
        for action_key, counts in actions.items():
            if action_key.startswith('action_'):
                action_num = int(action_key.split('_')[1])
                action_counts[action_num] = sum(counts)
                total_actions += sum(counts)
        
        if total_actions > 0:
            # Calculate entropy (diversity measure)
            probs = [action_counts.get(i, 0) / total_actions for i in range(16)]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            max_entropy = np.log2(16)  # Maximum possible entropy
            normalized_entropy = entropy / max_entropy
            
            diversity_metrics[agent_id] = {
                'entropy': entropy,
                'normalized_entropy': normalized_entropy,
                'action_distribution': {i: probs[i] for i in range(16) if probs[i] > 0.01},
                'most_used_actions': sorted(
                    [(i, probs[i]) for i in range(16)], 
                    key=lambda x: x[1], reverse=True
                )[:5]
            }
    
    return {
        'strategy_diversity': diversity_metrics,
        'summary': {
            'avg_entropy': np.mean([m['entropy'] for m in diversity_metrics.values()]),
            'avg_normalized_entropy': np.mean([m['normalized_entropy'] for m in diversity_metrics.values()]),
            'most_diverse_agent': max(diversity_metrics.items(), key=lambda x: x[1]['entropy'])[0] if diversity_metrics else None
        }
    }


def nash_equilibrium_analysis(model_dir: str) -> Dict[str, Any]:
    """Analyze Nash equilibrium properties of trained strategies."""
    print("Analyzing Nash equilibrium properties...")
    
    # Load training data
    data_file = os.path.join(model_dir, "training_data.json")
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            training_data = json.load(f)
    else:
        return {}
    
    evaluation_data = training_data.get('evaluation_data', {})
    
    # Extract exploitability data
    exploitability_data = {}
    for key, values in evaluation_data.items():
        if 'exploitability' in key:
            agent_id = key.replace('_exploitability', '')
            exploitability_data[agent_id] = values
    
    # Calculate convergence metrics
    convergence_analysis = {}
    
    for agent_id, exploitability_values in exploitability_data.items():
        if len(exploitability_values) > 10:
            recent_values = exploitability_values[-10:]
            variance = np.var(recent_values)
            trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            
            convergence_analysis[agent_id] = {
                'final_exploitability': recent_values[-1],
                'exploitability_variance': variance,
                'exploitability_trend': trend,
                'is_converging': variance < 0.01 and abs(trend) < 0.001
            }
    
    # Overall analysis
    if convergence_analysis:
        avg_exploitability = np.mean([data['final_exploitability'] for data in convergence_analysis.values()])
        converged_agents = sum(1 for data in convergence_analysis.values() if data['is_converging'])
        
        nash_quality = {
            'average_exploitability': avg_exploitability,
            'convergence_rate': converged_agents / len(convergence_analysis),
            'equilibrium_quality': 1 - avg_exploitability,  # Higher is better
            'is_approximate_nash': avg_exploitability < 0.1 and converged_agents / len(convergence_analysis) > 0.8
        }
    else:
        nash_quality = {}
    
    return {
        'agent_convergence': convergence_analysis,
        'nash_equilibrium_quality': nash_quality
    }


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained Coup agents')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing trained models')
    parser.add_argument('--num_episodes', type=int, default=200, help='Number of evaluation episodes')
    parser.add_argument('--baseline_eval', action='store_true', help='Run baseline evaluation')
    parser.add_argument('--strategy_analysis', action='store_true', help='Run strategy analysis')
    parser.add_argument('--nash_analysis', action='store_true', help='Run Nash equilibrium analysis')
    parser.add_argument('--full_eval', action='store_true', help='Run all evaluations')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"Model directory {args.model_dir} does not exist")
        return
    
    print(f"Evaluating models from: {args.model_dir}")
    print("="*50)
    
    results = {}
    
    # Baseline evaluation
    if args.baseline_eval or args.full_eval:
        try:
            baseline_results = evaluate_against_baselines(args.model_dir, args.num_episodes)
            results['baseline_evaluation'] = baseline_results
            
            print("\nBaseline Evaluation Results:")
            vs_baseline = baseline_results.get('vs_baseline', {})
            for baseline, agent_results in vs_baseline.items():
                print(f"\nVs {baseline}:")
                for agent_id, win_rate in agent_results.items():
                    print(f"  {agent_id}: {win_rate:.3f} win rate")
        
        except Exception as e:
            print(f"Baseline evaluation failed: {e}")
    
    # Strategy diversity analysis
    if args.strategy_analysis or args.full_eval:
        try:
            strategy_results = analyze_strategy_diversity(args.model_dir)
            results['strategy_analysis'] = strategy_results
            
            print("\nStrategy Diversity Analysis:")
            summary = strategy_results.get('summary', {})
            print(f"Average entropy: {summary.get('avg_entropy', 0):.3f}")
            print(f"Average normalized entropy: {summary.get('avg_normalized_entropy', 0):.3f}")
            print(f"Most diverse agent: {summary.get('most_diverse_agent', 'N/A')}")
            
            diversity = strategy_results.get('strategy_diversity', {})
            for agent_id, metrics in diversity.items():
                print(f"\n{agent_id}:")
                print(f"  Entropy: {metrics['entropy']:.3f}")
                print(f"  Top actions: {metrics['most_used_actions'][:3]}")
        
        except Exception as e:
            print(f"Strategy analysis failed: {e}")
    
    # Nash equilibrium analysis
    if args.nash_analysis or args.full_eval:
        try:
            nash_results = nash_equilibrium_analysis(args.model_dir)
            results['nash_analysis'] = nash_results
            
            print("\nNash Equilibrium Analysis:")
            nash_quality = nash_results.get('nash_equilibrium_quality', {})
            print(f"Average exploitability: {nash_quality.get('average_exploitability', 0):.4f}")
            print(f"Convergence rate: {nash_quality.get('convergence_rate', 0):.3f}")
            print(f"Equilibrium quality: {nash_quality.get('equilibrium_quality', 0):.3f}")
            print(f"Is approximate Nash: {nash_quality.get('is_approximate_nash', False)}")
        
        except Exception as e:
            print(f"Nash analysis failed: {e}")
    
    # Save results
    output_file = os.path.join(args.model_dir, "evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nEvaluation results saved to: {output_file}")


if __name__ == "__main__":
    main()