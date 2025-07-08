# Coup Reinforcement Learning System

This directory contains a complete reinforcement learning system for training agents to play the board game Coup using multi-agent self-play to approximate Nash equilibrium strategies.

## Features

### 🎯 Core Components
- **PettingZoo Environment**: Multi-agent environment wrapper for Coup game
- **PPO Agents with LSTM**: Policy networks with memory for handling partial observability
- **Opponent Modeling**: Belief systems for tracking opponent behavior patterns
- **Self-Play Training**: Multi-agent training with opponent rotation
- **Nash Equilibrium Detection**: Convergence metrics and exploitability analysis

### 🧠 Advanced Features
- **Bluffing Mechanics**: Explicit modeling of bluffing as part of policy output
- **Challenge/Block Systems**: Dynamic action spaces for interactive gameplay
- **Strategy Diversity**: Entropy-based measures of strategy complexity
- **Baseline Comparisons**: Multiple baseline strategies (random, honest, aggressive, defensive)

### 📊 Evaluation & Visualization
- **Real-time Metrics**: Win rates, action distributions, bluff success rates
- **Interactive Dashboards**: Plotly-based training progress visualization
- **Strategy Evolution**: Tracking how strategies change over time
- **Exploitability Analysis**: Measuring distance from Nash equilibrium

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

### Basic Training

```python
from coup.rl.training import CoupTrainer
from coup.rl.visualization import TrainingVisualizer, ProgressCallback

# Create trainer
config = {
    'trainer_args': {
        'num_players': 4,
        'save_dir': 'models',
        'log_dir': 'logs'
    },
    'max_episodes': 5000
}

trainer = CoupTrainer(config)

# Setup visualization
visualizer = TrainingVisualizer()
callback = ProgressCallback(visualizer)

# Train agents
trainer.train(progress_callback=callback)
```

### Command Line Training

```bash
# Run training example
python examples/train_coup_agents.py

# Evaluate trained models
python examples/evaluate_agents.py --model_dir models/checkpoint_5000 --full_eval
```

## Architecture Overview

### Environment (`coup/rl/environment.py`)
- **CoupEnvironment**: PettingZoo-compatible multi-agent environment
- **Observation Space**: 
  - Own coins and cards
  - Opponent information (coins, card counts)
  - Game phase and action history
  - Bluff detection features
- **Action Space**: 
  - Basic actions (Income, Foreign Aid, Coup, etc.)
  - Character actions (Tax, Assassinate, Exchange, Steal)
  - Challenge and Block decisions
  - Target selection for targeted actions

### Agent (`coup/rl/agent.py`)
- **CoupPPOAgent**: PPO agent with LSTM memory
- **Policy Network**: 
  - LSTM for sequential decision making
  - Separate heads for policy, value, and bluffing probability
  - Opponent modeling integration
- **Bluffing System**: Explicit bluff probability output and strategic bluffing

### Belief System (`coup/rl/belief_system.py`)
- **Opponent Modeling**: Track character probabilities and behavioral patterns
- **Bluff Detection**: Success rates and frequency analysis
- **Challenge Confidence**: Adaptive challenge/block decision making
- **Exploitability Estimation**: Measure opponent predictability

### Training (`coup/rl/training.py`)
- **Self-Play Trainer**: Multi-agent training with opponent rotation
- **Agent Pool**: Diverse opponent pool to prevent overfitting
- **Nash Convergence**: Strategy variance tracking and equilibrium detection
- **Evaluation**: Regular assessment against baselines and self-play

## Configuration

### Training Parameters

```python
training_config = {
    'episodes_per_agent_update': 10,    # PPO update frequency
    'agent_pool_size': 8,               # Diverse opponent pool
    'opponent_rotation_frequency': 50,   # Opponent mixing rate
    'evaluation_frequency': 100,        # Baseline evaluation
    'save_frequency': 500,              # Checkpoint saving
    'max_episodes': 10000               # Total training episodes
}
```

### Agent Parameters

```python
agent_config = {
    'hidden_size': 256,      # Policy network hidden size
    'lstm_size': 128,        # LSTM memory size
    'learning_rate': 3e-4,   # PPO learning rate
    'gamma': 0.99,           # Discount factor
    'gae_lambda': 0.95,      # GAE lambda
    'clip_ratio': 0.2,       # PPO clip ratio
    'entropy_coef': 0.01,    # Entropy regularization
    'bluff_coef': 0.1        # Bluffing loss weight
}
```

## Evaluation Metrics

### Performance Metrics
- **Win Rate**: Against baselines and in self-play
- **Episode Length**: Average game duration
- **Reward Distribution**: Agent performance over time

### Strategy Metrics
- **Action Distribution**: Frequency of different actions
- **Strategy Diversity**: Entropy of action policies
- **Bluff Success Rate**: Effectiveness of deception
- **Challenge Accuracy**: Ability to detect bluffs

### Nash Equilibrium Metrics
- **Strategy Convergence**: Variance in strategy profiles over time
- **Exploitability**: Estimated exploitability by best response
- **Equilibrium Quality**: Distance from theoretical Nash equilibrium

## Baseline Strategies

The system includes several baseline strategies for evaluation:

1. **Random**: Uniform random action selection
2. **Honest**: Never bluffs, only uses owned characters
3. **Aggressive**: Frequent character actions and challenges
4. **Defensive**: Conservative play, avoids risky actions
5. **Always Challenge**: Challenges all character actions
6. **Adaptive**: Learns and adapts strategy during play

## Visualization Features

### Training Dashboard
- Real-time win rates and episode rewards
- Policy loss and training stability
- Action distributions and strategy evolution
- Bluffing success rates and challenge accuracy

### Strategy Analysis
- Strategy evolution over time
- Action probability trends
- Character usage patterns
- Bluffing behavior analysis

### Nash Equilibrium Analysis
- Convergence metrics visualization
- Exploitability tracking
- Strategy variance plots
- Equilibrium quality assessment

## Advanced Usage

### Custom Baseline Strategies

```python
from coup.rl.baseline import BaselineAgent

class CustomStrategy(BaselineAgent):
    def act(self, observation, game_state, training=False):
        # Implement custom strategy logic
        return action

# Register with baseline factory
baselines = BaselineStrategies()
baselines.strategies['custom'] = CustomStrategy
```

### Custom Metrics

```python
from coup.rl.metrics import MetricsTracker

class CustomMetrics(MetricsTracker):
    def update_episode(self, episode_results):
        super().update_episode(episode_results)
        # Add custom metric tracking
        
    def get_custom_analysis(self):
        # Return custom analysis
        return analysis
```

### Training Callbacks

```python
def custom_callback(progress_data):
    episode = progress_data['episode']
    metrics = progress_data['recent_metrics']
    
    # Custom logging or analysis
    if episode % 100 == 0:
        print(f"Custom analysis at episode {episode}")

trainer.train(progress_callback=custom_callback)
```

## Files Structure

```
coup/rl/
├── __init__.py              # Package initialization
├── environment.py           # PettingZoo environment wrapper
├── agent.py                 # PPO agent with LSTM
├── belief_system.py         # Opponent modeling system
├── training.py              # Multi-agent training loop
├── baseline.py              # Baseline strategy implementations
├── metrics.py               # Metrics tracking and analysis
└── visualization.py         # Visualization and logging

examples/
├── train_coup_agents.py     # Training script example
└── evaluate_agents.py       # Evaluation script example
```

## Research Applications

This system is designed for research into:

1. **Multi-Agent Learning**: Self-play dynamics and convergence
2. **Game Theory**: Nash equilibrium approximation in complex games
3. **Deception and Bluffing**: Strategic deception in multi-agent systems
4. **Opponent Modeling**: Learning and exploiting opponent patterns
5. **Strategy Diversity**: Maintaining diverse strategies in self-play

## Performance Tips

### Training Efficiency
- Use GPU acceleration for larger networks
- Implement parallel environment execution
- Tune hyperparameters based on convergence metrics
- Monitor exploitability to detect overfitting

### Memory Usage
- Limit experience buffer size for long training runs
- Use gradient checkpointing for large networks
- Regular checkpoint saving to prevent data loss

### Convergence
- Monitor strategy variance for convergence detection
- Use diverse opponent pools to prevent cycling
- Regular evaluation against fixed baselines
- Early stopping based on exploitability thresholds

## Contributing

When contributing to the RL system:

1. Maintain compatibility with PettingZoo interface
2. Add comprehensive metrics for new features
3. Include baseline comparisons for new algorithms
4. Document hyperparameter sensitivity
5. Provide visualization for new metrics

## Citation

If you use this RL system in research, please cite:

```bibtex
@software{coup_rl_system,
  title={Multi-Agent Reinforcement Learning for Coup},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/coup}
}
```