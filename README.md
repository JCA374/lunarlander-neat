# NEAT LunarLander - Visual Learning Project

A pedagogical implementation of **NEAT** (NeuroEvolution of Augmenting Topologies) solving the **LunarLander-v2** environment, with a strong focus on **visualization** and **understanding** the algorithm.

## 🎯 Project Goals

This project prioritizes **pedagogy over performance**:

- 🔍 **Beautiful visualizations** of neural network evolution
- 📊 **Comprehensive statistics** tracking
- 🎓 **Educational code** with clear comments
- 🎨 **Publication-quality graphics**
- 🎮 **Interactive demos** to see the agent in action

Perfect for learning how NEAT evolves neural networks from simple to complex!

## 🧬 What is NEAT?

NEAT (NeuroEvolution of Augmenting Topologies) is a genetic algorithm that evolves neural networks. Unlike traditional neural networks:

- **Starts simple**: Begins with minimal networks (just inputs → outputs)
- **Grows structure**: Adds nodes and connections through mutation
- **Evolves topology**: Both weights AND network structure evolve
- **Maintains diversity**: Uses speciation to protect innovation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# On Ubuntu/Debian, you may also need graphviz:
sudo apt-get install graphviz
```

### 2. Train the Agent

```bash
python train.py
```

This will:
- Train for up to 5000 generations (or until fitness reaches 270)
- Save best genome every 10 generations (configurable)
- Log detailed statistics
- Create a timestamped run directory in `runs/`

Expected training time: varies; for quick smoke tests lower `max_generations` (e.g., 100) or raise `save_best_every` to reduce I/O.

### 3. Visualize Results

After training completes, you'll see commands to visualize the results. For example:

```bash
# Replace with your actual run directory
RUN_DIR="runs/run_20241126_120000"

# Visualize all network topologies
python visualize_network.py $RUN_DIR

# Plot training statistics
python plot_fitness.py $RUN_DIR

# Create animated GIF of network evolution
python visualize_evolution.py $RUN_DIR

# Watch the best agent play!
python demo_best.py $RUN_DIR
```

## 📁 Project Structure

```
lunarlander-neat/
├── train.py                  # Main training script
├── visualize_network.py      # Create network diagrams
├── plot_fitness.py           # Plot training statistics
├── visualize_evolution.py    # Create evolution GIF
├── demo_best.py             # Watch the trained agent
├── neat-config.ini          # NEAT configuration
├── requirements.txt         # Python dependencies
└── runs/                    # Training outputs
    └── run_YYYYMMDD_HHMM/
        ├── winner.pkl              # Best genome overall
        ├── best_genome_g{N}.pkl    # Best genome per generation
        ├── config.ini              # Copy of config used
        ├── stats.pkl               # Statistics object
        ├── fitness_history.csv     # Fitness data
        ├── complexity_history.csv  # Network complexity data
        ├── training_plots.png      # Comprehensive plots
        ├── fitness_plot.png        # Simple fitness plot
        ├── neat_evolution.gif      # Animated network evolution
        └── networks/               # Network diagrams
            ├── net_g000.png
            ├── net_g001.png
            └── ...
```

## 🎨 Visualizations Explained

### Network Diagrams

The network visualizations show:

- **Node colors**:
  - 🔵 **Blue**: Input nodes (x/y position & velocity, angle, angular velocity, leg contacts)
  - 🔴 **Red**: Output nodes (discrete thrusters)
  - 🟢 **Green**: Hidden nodes (evolved by NEAT)

- **Connection colors**:
  - **Green**: Positive weights (excitatory)
  - **Red**: Negative weights (inhibitory)
  - **Darker = stronger**: Color intensity shows weight magnitude

- **Connection style**:
  - **Solid**: Enabled connection
  - **Dashed**: Disabled connection
  - **Thickness**: Proportional to weight strength

### Training Plots

The fitness plots show:

1. **Fitness Evolution**: How the best, average, and worst fitness improve
2. **Standard Deviation**: Population diversity
3. **Species Diversity**: Number of species over time
4. **Network Complexity**: Average nodes and connections

### Evolution GIF

Watch the network grow from generation 0 to the final solution! You'll see:

- Initial simple connections (inputs → output)
- Emergence of hidden nodes
- Complexification as the task demands
- Optimization of connection weights

## 🎮 LunarLander Environment

The LunarLander-v2 problem:

- **Goal**: Soft-land the lunar module on the landing pad
- **Inputs (8)**:
  1. X position
  2. Y position
  3. X velocity
  4. Y velocity
  5. Angle
  6. Angular velocity
  7. Left leg ground contact (boolean)
  8. Right leg ground contact (boolean)
- **Actions (4)**:
  0. Do nothing
  1. Fire left engine
  2. Fire main engine
  3. Fire right engine
- **Success**: Average reward ≥ 270 over episodes
- **Challenge**: Multi-objective control (position, velocity, angle, fuel)

## ⚙️ Configuration Explained

Key settings in `neat-config.ini`:

```ini
pop_size = 150                   # Larger population for harder task
fitness_threshold = 270.0        # LunarLander considered "solved"
num_inputs = 8                   # LunarLander observations
num_outputs = 4                  # Discrete thrusters (argmax over outputs)
activation_default = tanh        # Smooth activation function

# Mutation rates tuned for gradual evolution without bloat
node_add_prob = 0.2             # Slightly lower to prevent bloat
conn_add_prob = 0.5             # 50% chance to add connection
weight_mutate_rate = 0.8        # 80% chance to mutate weights
```

These settings prioritize **observable evolution** over fast training.

## 🔧 Advanced Usage

### Adjust Generation Count

Edit `train.py` or call directly with a custom limit (default is 5000; use a lower number for smoke tests):

```python
winner, run_dir = run_neat('neat-config.ini', max_generations=5000, save_best_every=10)
```

### Visualize Specific Generation

```bash
# Create network diagram for generation 50
python -c "
import pickle, neat
from visualize_network import visualize_network

config = neat.Config(...)  # Load config
with open('runs/run_xxx/best_genome_g050.pkl', 'rb') as f:
    genome = pickle.load(f)
visualize_network(genome, config, 'gen_50', view=True)
"
```

### Interactive Demo Mode

```bash
python demo_best.py $RUN_DIR --interactive
```

This lets you trigger episodes on-demand and explore the agent's performance.

### Create Side-by-Side Comparison

```bash
python visualize_evolution.py $RUN_DIR --compare
```

Creates a side-by-side comparison of networks from different generations.

## 📚 Learning Resources

### Understanding NEAT

1. **Original Paper**: [Evolving Neural Networks through Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
2. **NEAT-Python Docs**: [neat-python.readthedocs.io](https://neat-python.readthedocs.io/)
3. **LunarLander Environment**: [Gymnasium LunarLander-v2](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

### Key NEAT Concepts

- **Speciation**: Groups similar genomes to protect innovation
- **Historical Markings**: Tracks gene history for crossover
- **Complexification**: Networks start simple and grow
- **Structural Innovation**: Both topology and weights evolve

## 🎓 Educational Features

This implementation includes several pedagogical enhancements:

1. **Verbose Output**: Clear progress indicators during training
2. **Generation Summaries**: Detailed statistics after each generation
3. **Visual Evolution**: See networks grow in complexity
4. **Performance Analysis**: Understand what makes good networks
5. **Interactive Exploration**: Play with the trained agent

## 🐛 Troubleshooting

### Import Errors

```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# On Ubuntu/Debian, install graphviz system package
sudo apt-get install graphviz
```

### Training Not Improving

- Check `neat-config.ini` mutation rates
- Try increasing population size
- Increase max generations
- Check fitness evaluation in `train.py`

### Visualization Issues

```bash
# Make sure you ran training first
python train.py

# Then run visualization on the output directory
python visualize_network.py runs/run_YYYYMMDD_HHMM/
```

### GIF Creation Fails

Make sure Pillow is installed:
```bash
pip install pillow
```

## 🎯 Next Steps

After mastering this project:

1. **Try harder environments**: Lunar Lander, Mountain Car, Acrobot
2. **Experiment with config**: Different mutation rates, population sizes
3. **Add recurrence**: Change `feed_forward = False` to allow cycles
4. **Compare with other methods**: Try DQN, PPO, A2C on same problem
5. **Visualize activation patterns**: Show neuron activations during play

## 📖 Code Structure

### `train.py`

- `eval_genome()`: Evaluates a single genome
- `eval_genomes()`: Evaluates entire population
- `FitnessLogger`: Custom reporter for detailed logging
- `run_neat()`: Main training loop

### `visualize_network.py`

- `visualize_network()`: Creates single network diagram
- `visualize_all_generations()`: Processes all saved genomes

### `plot_fitness.py`

- Loads CSV files with statistics
- Creates comprehensive multi-plot figure
- Shows fitness, diversity, and complexity

### `visualize_evolution.py`

- Combines network PNGs into animated GIF
- Creates side-by-side comparisons

### `demo_best.py`

- `run_winner()`: Runs trained agent
- `analyze_network()`: Analyzes network structure
- `demo_interactive()`: Interactive demo mode

## 🤝 Contributing

This is an educational project! Feel free to:

- Add more visualization options
- Try different environments
- Improve documentation
- Share your results!

## 📝 License

Educational project - free to use and modify for learning purposes.

## 🌟 Acknowledgments

- **NEAT Algorithm**: Kenneth O. Stanley and Risto Miikkulainen
- **neat-python**: CodeReclaimers
- **Gymnasium**: Farama Foundation
- **Graphviz**: AT&T Labs Research

---

**Happy Learning! 🎓🧬🤖**

Remember: The goal is to **understand** NEAT, not just to solve LunarLander quickly. Take time to explore the visualizations and see how evolution finds solutions!
