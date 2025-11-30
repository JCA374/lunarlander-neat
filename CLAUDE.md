# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **pedagogical implementation of NEAT** (NeuroEvolution of Augmenting Topologies) solving **LunarLander-v2** (previously CartPole). The project prioritizes **visualization and understanding** over raw performance. The goal is to help learners see how NEAT evolves neural networks from simple to complex topologies.

## Core Architecture

### Training Pipeline (train.py)

The training system has three main components:

1. **Fitness Evaluation** (`eval_genome`, `eval_genomes`)
   - Each genome → FeedForwardNetwork → LunarLander episode (averaged over 3 runs)
   - Action selection: 4 outputs → `np.argmax` for discrete thrusters
   - Fitness = average reward (target ≥ 200)

2. **Custom Reporter** (`FitnessLogger`)
   - Saves **periodic best genomes** as `g{N}.pkl` (every 5 generations by default)
   - Logs to THREE CSV files: `fitness.csv`, `complexity.csv`, and `species.csv`
   - This enables full evolution replay and detailed analysis

3. **Output Structure**
   - Creates timestamped `runs/run_YYYYMMDD_HHMM/` directory
   - Saves config copy, winner, stats, checkpoints, and CSV logs

### Visualization Pipeline

All visualization scripts take a `run_dir` as argument and operate on saved artifacts:

- **visualize_network.py**: Loads `.pkl` genomes → Graphviz diagrams
  - Color codes weights (green=positive, red=negative)
  - Line thickness = weight magnitude
  - Creates `networks/net_g{N}.png` for each generation

- **plot_fitness.py**: Reads CSV files → matplotlib plots
  - Creates both comprehensive 4-panel and simple plots
  - Shows fitness, diversity, complexity evolution

- **plot_complexity_fitness.py**: Analyzes relationship between network size and performance
  - Creates 3 synchronized time-series plots (fitness, nodes, connections)
  - Creates scatter plots showing complexity vs fitness correlation
  - Calculates correlation coefficients to show if bigger = better
  - Teaches that NEAT optimizes for minimal effective structure

- **visualize_evolution.py**: Combines PNGs → animated GIF
  - Uses Pillow to create `neat_evolution.gif`
  - Optional side-by-side comparison

- **demo_best.py**: Loads winner.pkl or specific genome → plays LunarLander
  - Accepts run directory OR specific genome file path
  - Falls back to main config if training still running
  - Renders with `gymnasium` environment
  - Interactive mode available

- **watch_progress.py**: Quick helper to test latest genome during training
  - Automatically finds most recent saved genome
  - Perfect for monitoring training progress
  - Shows generation number and performance

### Data Flow

```
train.py
  ↓ (creates)
runs/run_YYYYMMDD_HHMM/
  ├── data/
  │   ├── fitness.csv        ← Used by plot_fitness.py
  │   ├── complexity.csv     ← Used by plot_complexity_fitness.py
  │   └── species.csv        ← Used for diversity analysis
  ├── genomes/
  │   ├── g000.pkl, g005.pkl ... ← Used by visualize_network.py
  │   └── winner.pkl         ← Used by demo_best.py
  └── config.ini             ← Used by all visualization scripts
```

## Common Commands

### Setup
```bash
./setup.sh
# or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo apt-get install graphviz  # Ubuntu/Debian only
```

### Training
```bash
source venv/bin/activate
python train.py  # Creates runs/run_YYYYMMDD_HHMM/
```

### Visualization Workflow
```bash
RUN_DIR="runs/run_20241126_120000"  # Replace with actual

# Generate all visualizations
python visualize_network_v2.py $RUN_DIR
python plot_fitness.py $RUN_DIR
python plot_complexity_fitness.py $RUN_DIR  # NEW: Complexity analysis
python visualize_evolution.py $RUN_DIR
python demo_best.py $RUN_DIR

# Or generate everything at once
python visualize_all.py $RUN_DIR

# Useful options
python visualize_network_v2.py $RUN_DIR --view  # Open last network
python demo_best.py $RUN_DIR --interactive      # Manual episode triggering
python visualize_evolution.py $RUN_DIR --fast --compare
```

### Quick Access to Latest Run
```bash
RUN_DIR=$(ls -td runs/run_* | head -1)
```

## Configuration Philosophy

The `neat-config.ini` is tuned for **observable evolution**, not speed:

- `pop_size = 150` - Larger population to tackle LunarLander complexity
- `node_add_prob = 0.2` and `conn_add_prob = 0.5` - Gradual complexification without bloat
- `num_hidden = 0` - Networks start minimal (inputs → outputs only)
- `num_inputs = 8`, `num_outputs = 4` - LunarLander obs and discrete thrusters
- `fitness_threshold = 200.0` - LunarLander "solved" target (average reward)

Key insight: Networks evolve from 12 nodes + sparse connections → ~15-20 nodes + 30-50 connections.

## Modifying Training Behavior

To change training parameters, edit `train.py`'s `run_neat()` call or modify `neat-config.ini`:

```python
# In train.py, change max_generations
winner, run_dir = run_neat('neat-config.ini', max_generations=150)

# In neat-config.ini, adjust mutation rates to see different evolution patterns
node_add_prob = 0.3  # More aggressive structure evolution
pop_size = 120       # Smaller population for faster iteration (less stable)
```

## Adding New Visualizations

All visualization scripts follow this pattern:
1. Load config from `run_dir/config.ini`
2. Create NEAT config object
3. Load genome(s) from `.pkl` files
4. Process and save output to `run_dir/`

Example for a new visualization:
```python
import neat, pickle, os, sys

run_dir = sys.argv[1]
config = neat.Config(neat.DefaultGenome, ...,
                     os.path.join(run_dir, 'config.ini'))

with open(os.path.join(run_dir, 'winner.pkl'), 'rb') as f:
    genome = pickle.load(f)

# Your visualization logic here
```

## Testing Different Environments

To adapt to a different Gymnasium environment:
1. Change `gym.make('LunarLander-v2')` in `eval_genome()`
2. Update `num_inputs` and `num_outputs` in `neat-config.ini`
3. Adjust `fitness_threshold` to the new success criteria
4. Update visualization labels in scripts (input/output node names)

## Important Invariants

- **Every script must work with timestamped run directories** - Never hardcode paths
- **Genomes saved every 10 generations** - Keeps disk usage reasonable while enabling evolution replay
- **Config copied to run_dir** - Ensures reproducibility
- **CSV format for stats** - Human-readable, easy to analyze externally
- **Visualization scripts are idempotent** - Safe to re-run on same data

## Known Issues & Fixes

### Graphviz Color Warnings (Fixed)
Previous versions generated warnings like `Warning: "#00e100" is not a known color`. This was fixed by removing quotes from hex color strings in `visualize_network.py`. Colors are now properly formatted as `#00ff00` instead of `"#00ff00"`.

### Matplotlib Unicode/Emoji Warnings (Fixed)
Previous versions generated warnings about missing glyphs for emoji characters (🏆, 📊, 🧬, 🧠) in plot titles. This was fixed by removing emojis from all matplotlib plot titles in `plot_fitness.py`. The plots now use clean text titles without special characters.

## Troubleshooting

If visualizations fail:
- Check that training completed: `ls runs/run_YYYYMMDD_HHMM/*.pkl`
- Verify graphviz installed: `which dot`
- Ensure virtual environment active: `which python` should show venv path
- Check for CSV files: `runs/run_*/fitness_history.csv` must exist

If training plateaus:
- Increase `node_add_prob` or `conn_add_prob` in config
- Increase population size or max generations
- Check that LunarLander environment is working: `python -c "import gymnasium; gymnasium.make('LunarLander-v2')"`
