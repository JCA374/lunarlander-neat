# Quick Start Guide

## 🚀 Setup (First Time Only)

```bash
# Option 1: Use setup script (recommended)
./setup.sh

# Option 2: Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🎯 Train the Agent

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Start training
python train.py
```

**Training will take 1-3 hours** and create a directory like `runs/run_20241126_120000/`

## 📊 Visualize Results

After training, replace `RUN_DIR` with your actual run directory:

```bash
RUN_DIR="runs/run_20241126_120000"  # Use your actual directory!

# 1. Create network diagrams (shows evolution of topology)
python visualize_network.py $RUN_DIR

# 2. Plot training statistics
python plot_fitness.py $RUN_DIR

# 3. Create animated GIF
python visualize_evolution.py $RUN_DIR

# 4. Watch the trained agent play!
python demo_best.py $RUN_DIR
```

## 🎮 Interactive Demo

```bash
# Watch the agent with different options
python demo_best.py $RUN_DIR --interactive    # Manual control
python demo_best.py $RUN_DIR --episodes 10    # Run 10 episodes
python demo_best.py $RUN_DIR --analyze        # Just analyze network
```

## 📁 What Gets Created

```
runs/run_20241126_120000/
├── winner.pkl                 # Best genome
├── best_genome_g*.pkl        # Genomes per generation
├── fitness_history.csv       # Training data
├── complexity_history.csv    # Network complexity
├── training_plots.png        # Comprehensive plots
├── fitness_plot.png          # Simple fitness graph
├── neat_evolution.gif        # Animated evolution!
└── networks/                 # Network diagrams
    ├── net_g000.png
    ├── net_g001.png
    └── ...
```

## 🎨 Visualization Guide

### Network Diagrams
- **Blue nodes** = Inputs (x/y position & velocity, angle, angular velocity, leg contacts)
- **Red nodes** = Outputs (4 discrete thrusters)
- **Green nodes** = Hidden neurons (evolved by NEAT)
- **Green connections** = Positive weights (excitatory)
- **Red connections** = Negative weights (inhibitory)
- **Thicker lines** = Stronger weights

### Training Plots
- **Best fitness** = Best performance in population
- **Average fitness** = Population mean
- **Species count** = Diversity measure
- **Network complexity** = Nodes and connections over time

## ⚡ Quick Commands Reference

```bash
# Full workflow
python train.py
python visualize_network.py runs/run_*/
python plot_fitness.py runs/run_*/
python visualize_evolution.py runs/run_*/
python demo_best.py runs/run_*/

# Create faster/slower GIF
python visualize_evolution.py runs/run_*/ --fast
python visualize_evolution.py runs/run_*/ --slow

# Demo with different speeds
python demo_best.py runs/run_*/ --fast
python demo_best.py runs/run_*/ --slow
```

## 🐛 Troubleshooting

**Virtual environment not activated?**
```bash
source venv/bin/activate
```

**Missing dependencies?**
```bash
pip install -r requirements.txt
```

**On Ubuntu, missing graphviz?**
```bash
sudo apt-get install graphviz
```

**Can't find run directory?**
```bash
ls -lah runs/
# Use the most recent one
```

## 📚 Learn More

See [README.md](README.md) for:
- Detailed explanations of NEAT algorithm
- Configuration options
- Advanced usage
- Educational resources

## 🎯 Expected Results

- **Generation 0-50**: Often negative rewards (crashes)
- **Generation 50-150**: Occasional hovering, scores creep toward 0-100
- **Generation 150-300**: Stable landings begin to appear, 150-250+ rewards
- **Target**: 200 average reward = SOLVED! ✅

Networks typically:
- Start with 0 hidden nodes
- End with 3-6 hidden nodes
- Use 20-40 connections

---

**Happy learning! 🧬🎓**
