# Visualization Files

This directory contains all generated visualizations for this training run.

## Files Overview

### Standard Training Plots

**training_plots.png** (4-panel comprehensive view)
- Fitness evolution (best/average/worst with std dev)
- Species diversity over time
- Network complexity evolution
- Best for overall training summary

**fitness_plot.png** (simple single plot)
- Clean fitness curve over generations
- Best for presentations or quick reference

### Complexity Analysis (NEW!)

**complexity_vs_fitness.png** (3-panel time series)
- Top: Fitness progression
- Middle: Node count evolution
- Bottom: Connection count evolution
- All synchronized on same X-axis
- **Best for understanding how network structure changes over time**

**scatter_complexity_fitness.png** (relationship analysis)
- Left: Nodes vs Fitness scatter plot
- Right: Connections vs Fitness scatter plot
- Color-coded by generation (early=purple, late=yellow)
- **Best for seeing if bigger networks = better performance**

**combined_complexity_fitness.png** (dual-axis combo)
- Fitness (green) on left axis
- Nodes (blue) and Connections (red) on right axis
- Everything on one plot
- **Best for quick comparison of fitness and complexity trends**

### Network Diagrams

**networks/** (directory)
- Individual network diagrams for each saved generation
- Shows actual neural network structure with nodes and connections
- Color-coded weights (green=positive, red=negative)

### Animation

**neat_evolution.gif** (if generated)
- Animated sequence of network diagrams
- Shows how network structure evolves over training
- Great for presentations and understanding evolution

## What to Look For

### In Complexity Plots:

1. **Correlation between fitness and complexity**
   - Do networks get bigger as they improve?
   - Or do they stay small (or even shrink)?

2. **Breakthroughs**
   - Large fitness jumps - did complexity change at the same time?
   - Or did the network find a better solution with same structure?

3. **Pruning events**
   - Networks getting simpler (fewer nodes/connections)
   - Usually means evolution found redundant structure to remove

4. **Plateau phases**
   - Flat fitness with changing complexity = searching for better structure
   - Flat fitness with stable complexity = fine-tuning weights

### Key Statistics (from plot_complexity_fitness.py output):

```
Correlations with Best Fitness:
  Nodes:        -0.016   (near zero - node count doesn't matter much)
  Connections:  -0.189   (slight negative - networks pruned connections)
```

**This tells you:**
- NEAT found solutions with FEWER connections as training progressed
- More complexity ≠ better performance (NEAT optimizes for efficiency!)
- The algorithm successfully identified and removed unnecessary structure

## Regenerating These Files

To recreate all visualizations:

```bash
# From project root
python visualize_all.py runs/run_20251127_080353

# Or individually:
python plot_fitness.py runs/run_20251127_080353
python plot_complexity_fitness.py runs/run_20251127_080353
python visualize_network_v2.py runs/run_20251127_080353
python visualize_evolution.py runs/run_20251127_080353
```
