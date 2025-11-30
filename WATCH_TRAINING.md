# Watching Training Progress - Quick Guide

## The Problem You Had

You tried to watch a genome during training:
```bash
python demo_best.py runs/run_20251130_163831/genomes/g100.pkl
```

And got an error:
```
❌ Config file not found: runs/run_20251130_163831/genomes/g100.pkl/config.ini
```

## The Solution - Two New Features!

### 1. `demo_best.py` Now Accepts Genome Files!

The script has been enhanced to handle **both**:
- Run directories (as before): `python demo_best.py runs/run_20251130_163831`
- **NEW:** Specific genome files: `python demo_best.py runs/run_20251130_163831/genomes/g100.pkl`

**How it works:**
- Detects if you passed a `.pkl` file
- Extracts the run directory from the path
- Falls back to `neat-config.ini` if training is still running (config not yet saved)
- Loads and tests that specific genome

**Examples:**
```bash
# Watch generation 100
python demo_best.py runs/run_20251130_163831/genomes/g100.pkl

# Watch with visualization
python demo_best.py runs/run_20251130_163831/genomes/g100.pkl --render

# Test more episodes
python demo_best.py runs/run_20251130_163831/genomes/g100.pkl --episodes 10

# Just analyze the network structure
python demo_best.py runs/run_20251130_163831/genomes/g100.pkl --analyze
```

### 2. NEW Script: `watch_progress.py`

Even easier! Automatically finds and tests the **latest** genome:

```bash
# Quick test of latest genome (no rendering, 3 episodes)
python watch_progress.py runs/run_20251130_163831

# With visualization
python watch_progress.py runs/run_20251130_163831 --render

# Test more episodes
python watch_progress.py runs/run_20251130_163831 --episodes 10

# Just analyze structure
python watch_progress.py runs/run_20251130_163831 --analyze
```

**What it does:**
1. Finds the most recent genome file in `genomes/` directory
2. Shows you which generation it's testing
3. Runs `demo_best.py` with that genome
4. Shows performance statistics

**Example output:**
```
======================================================================
WATCH TRAINING PROGRESS
======================================================================
Run:        runs/run_20251130_163831
Genome:     g110.pkl
Generation: 110
======================================================================

[Network analysis and episode results...]

📊 PERFORMANCE SUMMARY
  Average reward:  245.6
  Best reward:     286.6
  Worst reward:    224.4
  Success rate:   3/3
```

## Use Cases

### During Training

**Scenario:** Training is running, you want to see current progress

```bash
# In one terminal: training is running
python train.py

# In another terminal: watch latest progress
python watch_progress.py runs/run_20251130_163831
```

The script will:
- Find the latest saved genome (e.g., g110.pkl)
- Use the main `neat-config.ini` (since training hasn't finished)
- Test and show performance

**Repeat this every 50-100 generations** to track improvement!

### After Training

**Scenario:** Training completed, you want to compare different generations

```bash
# Test the winner (final best)
python demo_best.py runs/run_20251130_163831

# Compare to early generation
python demo_best.py runs/run_20251130_163831/genomes/g050.pkl

# Compare to mid-training
python demo_best.py runs/run_20251130_163831/genomes/g250.pkl

# Compare to late training
python demo_best.py runs/run_20251130_163831/genomes/g450.pkl
```

### Render Visualization

**Scenario:** You want to SEE the lander in action

```bash
# Watch latest genome with visualization
python watch_progress.py runs/run_20251130_163831 --render

# Watch specific generation with visualization
python demo_best.py runs/run_20251130_163831/genomes/g200.pkl --render --episodes 5
```

**Note:** Rendering is slower but more satisfying! 🚀

## Quick Reference Table

| Command | What It Does | When To Use |
|---------|--------------|-------------|
| `python watch_progress.py <run_dir>` | Test latest genome, no render | Quick progress check during training |
| `python watch_progress.py <run_dir> --render` | Test latest genome, with render | See latest behavior visually |
| `python demo_best.py <run_dir>` | Test final winner | After training completes |
| `python demo_best.py <run_dir>/genomes/g100.pkl` | Test generation 100 | Compare specific generation |
| `python demo_best.py <run_dir>/genomes/g100.pkl --analyze` | Just show network structure | Understand network complexity |

## Troubleshooting

**Q: Error about config file not found?**
- Fixed! The scripts now fall back to `neat-config.ini` if training is still running

**Q: No genomes found in directory?**
- Training just started, wait for generation 0 to complete (genomes saved every 5 generations)

**Q: Performance seems random?**
- LunarLander has randomness. Run more episodes: `--episodes 10`

**Q: Want to test without any visual output?**
- Already the default! Scripts don't render unless you use `--render`

**Q: How often should I check progress?**
- Every 50-100 generations is good
- Or whenever you see a big fitness jump in the training output!

## Tips

1. **Check progress strategically:**
   - When you see "HUGE LEAP!" in training output
   - When network complexity changes significantly
   - When fitness crosses 0, 100, 200 milestones

2. **Compare generations:**
   ```bash
   # Early (random behavior)
   python demo_best.py runs/run_20251130_163831/genomes/g000.pkl --episodes 3

   # Mid (learning)
   python demo_best.py runs/run_20251130_163831/genomes/g100.pkl --episodes 3

   # Late (skilled)
   python demo_best.py runs/run_20251130_163831/genomes/g200.pkl --episodes 3
   ```

3. **Use `watch_progress.py` like a dashboard:**
   ```bash
   # Simple loop to check every 5 minutes
   while true; do
     python watch_progress.py runs/run_20251130_163831
     sleep 300
   done
   ```

Enjoy watching your networks learn to land! 🌙
