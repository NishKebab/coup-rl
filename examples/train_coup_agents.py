"""Example script for training Coup RL agents."""

import os
import sys
import logging
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from coup.rl.training import CoupTrainer
from coup.rl.visualization import TrainingVisualizer, ProgressCallback
from coup.rl.metrics import MetricsTracker


def create_training_config() -> Dict[str, Any]:
    """Create training configuration."""
    return {
        'trainer_args': {
            'num_players': 4,
            'save_dir': 'models',
            'log_dir': 'logs'
        },
        'max_episodes': 5000,
        'evaluation_frequency': 100,
        'save_frequency': 500
    }


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main training function."""
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Coup RL Agent Training")
    
    # Create trainer
    config = create_training_config()
    trainer = CoupTrainer(config)
    
    # Setup visualization
    visualizer = TrainingVisualizer(save_dir="visualizations", update_frequency=10)
    progress_callback = ProgressCallback(visualizer, log_frequency=50)
    
    try:
        # Train agents
        logger.info("Beginning training...")
        trainer.train(progress_callback=progress_callback)
        
        logger.info("Training completed successfully!")
        
        # Final evaluation
        logger.info("Running final evaluation...")
        final_metrics = trainer.get_metrics()
        
        # Create final visualizations
        visualizer.create_training_dashboard()
        visualizer.create_strategy_evolution_plot()
        visualizer.create_nash_equilibrium_analysis()
        visualizer.save_training_data()
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        
        agent_performance = final_metrics.get('agent_performance', {})
        for agent_id, performance in agent_performance.items():
            print(f"\n{agent_id}:")
            print(f"  Win Rate: {performance.get('win_rate', 0):.3f}")
            print(f"  Avg Reward: {performance.get('avg_reward', 0):.3f}")
            print(f"  Total Games: {performance.get('total_games', 0)}")
        
        nash_metrics = final_metrics.get('nash_equilibrium', {})
        print(f"\nNash Equilibrium Metrics:")
        print(f"  Strategy Convergence: {nash_metrics.get('strategy_convergence', 0):.4f}")
        print(f"  Estimated Exploitability: {nash_metrics.get('estimated_exploitability', 0):.4f}")
        print(f"  Is Converged: {nash_metrics.get('is_converged', False)}")
        
        print(f"\nVisualization files saved to: visualizations/")
        print(f"Model checkpoints saved to: models/")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    logger.info("Training script completed")


if __name__ == "__main__":
    main()