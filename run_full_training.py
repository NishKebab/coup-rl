"""Run full training session with visualization."""

import sys
import os
sys.path.insert(0, 'src')

import logging
from coup.rl.training import CoupTrainer
from coup.rl.visualization import TrainingVisualizer, ProgressCallback

def main():
    """Run full training with visualization."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('full_training.log'),
            logging.StreamHandler()
        ]
    )
    
    print("🎯 Starting Full Coup RL Training with Visualization")
    print("="*60)
    
    # Create trainer with full configuration
    config = {
        'trainer_args': {
            'num_players': 4,
            'save_dir': 'models',
            'log_dir': 'logs'
        },
        'max_episodes': 1000,  # More episodes for better learning
        'evaluation_frequency': 100,
        'save_frequency': 200
    }
    
    trainer = CoupTrainer(config)
    
    # Setup visualization
    visualizer = TrainingVisualizer(save_dir="visualizations", update_frequency=10)
    
    # Create progress callback that updates visualizations
    def enhanced_callback(progress_data):
        episode = progress_data.get('episode', 0)
        
        # Log progress
        if episode % 50 == 0:
            print(f"\n📊 Episode {episode} Progress:")
            recent_metrics = progress_data.get('recent_metrics', {})
            
            # Agent performance
            agent_perf = recent_metrics.get('agent_performance', {})
            for agent_id, metrics in agent_perf.items():
                win_rate = metrics.get('win_rate', 0)
                avg_reward = metrics.get('avg_reward', 0)
                print(f"  {agent_id}: Win Rate: {win_rate:.3f}, Avg Reward: {avg_reward:.3f}")
        
        # Update visualizations every 100 episodes
        if episode % 100 == 0 and episode > 0:
            try:
                print(f"📈 Generating visualizations at episode {episode}...")
                visualizer.create_training_dashboard()
                visualizer.create_strategy_evolution_plot()
                visualizer.save_training_data()
                print(f"✅ Visualizations saved to: visualizations/")
            except Exception as e:
                print(f"⚠️  Visualization error: {e}")
    
    try:
        # Train agents
        print("🚀 Starting training...")
        trainer.train(progress_callback=enhanced_callback)
        
        print("\n🎉 Training completed successfully!")
        
        # Final visualization
        print("📊 Creating final visualizations...")
        visualizer.create_training_dashboard()
        visualizer.create_strategy_evolution_plot()
        visualizer.create_nash_equilibrium_analysis()
        visualizer.save_training_data()
        
        print(f"\n📁 Results saved to:")
        print(f"   Models: models/")
        print(f"   Logs: logs/")
        print(f"   Visualizations: visualizations/")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        print("💾 Saving current progress...")
        
        # Save what we have so far
        try:
            visualizer.save_training_data()
            print("✅ Progress saved")
        except:
            print("⚠️  Could not save progress")
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()