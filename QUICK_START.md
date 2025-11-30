# Quick Start Guide - Beginner-Friendly NEAT Training

## What You Have Now

This NEAT implementation is optimized for **learning and visualization**. It shows you:
- How networks evolve from simple to complex
- The relationship between network size and performance
- Clear, beginner-friendly progress tracking

## Step 1: Train a Network

```bash
source .venv/bin/activate
python train.py
```

**What happens:**
- Creates a timestamped run directory (e.g., `runs/run_20251130_153000/`)
- Trains for up to 5000 generations (stops early if target reached)
- Target: 250 (LunarLander is "solved" at 200+, we aim higher)
- Saves network snapshots every 5 generations
- Shows beginner-friendly progress with visual progress bar

**Example output:**
```
======================================================================
GENERATION 100 / 5000
======================================================================
  Best Fitness:     45.3  (+12.4 improving)
  Average Fitness: -230.5

  Species:         2
    -> Low diversity (population converging)

  Network Size:    5.2 nodes, 18.3 connections
    -> Largest:    7 nodes, 24 connections
    -> Growing (+1.2 nodes - adding complexity)

  Progress: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 18.1%
  Target:   250 (LunarLander solved at 200+)
  (Learning to land - getting positive scores!)
```

## Step 2: Generate ALL Visualizations

```bash
# Use the run directory from your training
python visualize_all.py runs/run_20251130_153000
```

This creates:
1. **Training analysis** - JSON data about the run
2. **Training summary** - Markdown report
3. **Network diagrams** - PNG images showing network evolution
4. **Fitness plots** - Standard training curves
5. **Complexity vs Fitness plots** - NEW! Shows how network size relates to performance
6. **Evolution animation** - Animated GIF of network growth
7. **Network analysis** - Detailed report on the best network

## Step 3: Explore the Results

### Watch Your Network Play

```bash
python demo_best.py runs/run_20251130_153000
```

You'll see the LunarLander environment with your trained network controlling it!

### View Training Progression

Open these files in your browser or image viewer:

**Standard Training Plots:**
- `runs/run_20251130_153000/visualizations/training_plots.png` - 4-panel comprehensive view
- `runs/run_20251130_153000/visualizations/fitness_plot.png` - Simple fitness over time

**NEW: Complexity Analysis:**
- `runs/run_20251130_153000/visualizations/complexity_vs_fitness.png` - 3-panel time series
- `runs/run_20251130_153000/visualizations/scatter_complexity_fitness.png` - Scatter plots
- `runs/run_20251130_153000/visualizations/combined_complexity_fitness.png` - Dual-axis combo

**Network Evolution:**
- `runs/run_20251130_153000/visualizations/neat_evolution.gif` - Animated network growth
- `runs/run_20251130_153000/visualizations/networks/` - Individual network diagrams

### Read the Summary

```bash
cat runs/run_20251130_153000/analysis/training_summary.md
```

## Understanding the Complexity vs Fitness Plots

### Figure 1: Three Synchronized Plots

Shows fitness, nodes, and connections over time on separate aligned plots. Look for:
- **Fitness jumps** - When did the network suddenly improve?
- **Complexity changes** - Did the network grow or shrink at those moments?
- **Patterns** - Does complexity always increase? Or does pruning happen?

### Figure 2: Scatter Plots

Each dot is a generation, color-coded from early (purple) to late (yellow):
- **Upward trend** = More complexity → better fitness
- **Horizontal spread** = Different complexities achieve similar fitness
- **Clustering** = Networks converge to similar size

### Figure 3: Combined Dual-Axis

Everything on one plot:
- Green line = Fitness (left axis)
- Blue/Red lines = Nodes/Connections (right axis)
- See the relationship at a glance!

### Correlation Coefficients

At the bottom of the output:
```
Correlations with Best Fitness:
  Nodes:        -0.016
  Connections:  -0.189
```

**What this means:**
- **+1.0** = Perfect positive correlation (more complex = better)
- **0.0** = No correlation (complexity doesn't matter)
- **-1.0** = Perfect negative correlation (simpler = better, pruning)

In the example:
- Nodes: -0.016 (basically zero - node count doesn't affect fitness much)
- Connections: -0.189 (slight negative - networks actually get simpler as they improve!)

**Key lesson:** NEAT doesn't just grow networks - it finds the **minimal effective structure**!

## Configuration Quick Reference

Current settings in [neat-config.ini](neat-config.ini):

```ini
fitness_threshold = 299.0  # Target score (200 = solved, we aim higher)
pop_size = 120             # 120 networks per generation
node_add_prob = 0.25       # 25% chance to add a node
conn_add_prob = 0.6        # 60% chance to add a connection
```

**Save frequency:** Every 5 generations (in [train.py](train.py:493))

## Tweaking for Different Goals

### Faster Training (Less Pedagogical)
```ini
pop_size = 80               # Smaller population
fitness_threshold = 220     # Easier target
```

### More Dramatic Evolution (Better Visualization)
```ini
node_add_prob = 0.3         # Add nodes more aggressively
conn_add_prob = 0.7         # Add connections more aggressively
```

### Harder Challenge (Advanced)
```ini
fitness_threshold = 280     # Higher target
pop_size = 200              # More diversity
```

## Troubleshooting

**Training seems stuck at negative fitness:**
- Normal for first 50-100 generations
- Networks are learning the basics (don't crash as hard)
- Watch for the message: "Learning to land - getting positive scores!"

**Visualizations fail:**
- Make sure you activated the virtual environment: `source .venv/bin/activate`
- Check Graphviz is installed: `which dot`
- Verify the run directory exists and has data

**Want to see more/fewer network snapshots:**
- Edit [train.py](train.py:493): change `save_best_every=5` to a different number
- Smaller number = more snapshots, smoother animation
- Larger number = fewer snapshots, faster training, less disk space

## File Structure Reference

```
runs/run_20251130_153000/
├── config.ini                    # Copy of training config
├── data/
│   ├── fitness.csv              # Fitness data (used by plots)
│   ├── complexity.csv           # Complexity data (NEW!)
│   ├── species.csv              # Species diversity data
│   └── stats.pkl                # Full statistics object
├── genomes/
│   ├── g000.pkl                 # Generation 0 best network
│   ├── g005.pkl                 # Generation 5 best network
│   ├── g010.pkl                 # ... every 5 generations
│   └── winner.pkl               # Final best network
├── checkpoints/
│   └── checkpoint-*             # NEAT checkpoints (every 10 gen)
├── analysis/
│   ├── training_summary.md      # Human-readable summary
│   └── key_moments.json         # Detected breakthroughs
└── visualizations/
    ├── training_plots.png       # Standard 4-panel plots
    ├── fitness_plot.png         # Simple fitness curve
    ├── complexity_vs_fitness.png    # NEW! 3-panel time series
    ├── scatter_complexity_fitness.png  # NEW! Scatter plots
    ├── combined_complexity_fitness.png # NEW! Dual-axis
    ├── neat_evolution.gif       # Animation
    └── networks/
        ├── net_g000.png         # Network diagram gen 0
        ├── net_g005.png         # Network diagram gen 5
        └── ...                  # One per saved generation
```

## Next Steps

1. **Run a training session** and watch the beginner-friendly output
2. **Generate all visualizations** with `visualize_all.py`
3. **Study the complexity plots** - they're the most educational!
4. **Try different config settings** and see how it affects evolution
5. **Compare multiple runs** using the scatter plots

Have fun learning about neuroevolution! 🚀
