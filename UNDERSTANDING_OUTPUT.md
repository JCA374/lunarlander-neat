# Understanding the Output - A Visual Guide

This guide matches the current **LunarLander-v2** setup (8 inputs, 4 outputs, target fitness 200). Early generations often show **negative** rewards; progress is measured by rising averages toward 200+.

## 🎓 How to Read NEAT's Evolution

### During Training
Typical generation summary:
```
======================================================================
📊 GENERATION 0 SUMMARY
======================================================================
  🏆 Best Fitness:   -120.0
  📈 Average Fitness: -240.5 (±60.2)
  📉 Worst Fitness:   -360.0
  🧬 Species Count:   6
  🧠 Network Complexity:
     - Avg Nodes:       12.0 (max: 12)
     - Avg Connections: 10.0 (max: 12)
  💾 Saved: best_genome_g000.pkl
======================================================================
```
Interpretation:
- Fitness < 0 is normal early on (crashes). Look for the **trend**, not single values.
- Complexity starts at 12 nodes (8 inputs + 4 outputs) with sparse connections.
- Species > 4 = healthy diversity; < 2 for many generations can signal convergence too soon.

### Evolution Progress (example trajectory)
- Gen 0-50: Fitness typically -400 → 0; networks add first hidden nodes.
- Gen 50-150: Hovering emerges; occasional positive episodes (50–150).
- Gen 150-300: Consistent landings; best/avg move toward 200+ (solved).

## 📊 Reading the Plots

### Fitness Plot
- Target line at **200**.
- Good signs: best and average both trend upward; gap narrows.
- Watch for reversals: a few noisy drops are fine, long flatlines mean tweak config or population size.

### Complexity Plot
- Expect growth from ~12 nodes / ~10 connections toward ~15-20 nodes / 30-50 connections.
- Healthy runs plateau in complexity once fitness stabilizes; unchecked growth with flat fitness can mean bloat.

## 🧠 Reading Network Diagrams

- Inputs (blue): X/Y position & velocity, angle, angular velocity, left/right leg contacts.
- Outputs (red): No-op, Left, Main, Right thrusters (argmax of 4).
- Hidden nodes (green): emergent feature detectors (e.g., angle damping, landing alignment).
- Early networks: direct input→output, few edges.
- Later networks: multiple hidden nodes, mixed short/long connections, leg contact signals weighted near the end of episodes.

## 📈 Success Indicators

- Best fitness approaches / exceeds **200**; average rises toward 150-200.
- Species stabilizes around 3-7.
- Complexity growth slows while fitness keeps improving.
- Demo (`demo_best.py`) shows controlled descent, brief hover, gentle touchdown.

## 🎯 LunarLander-Specific Ranges

| Fitness | Likely behavior |
|---------|-----------------|
| -400 to -200 | Immediate crashes, engines misused |
| -200 to -50  | Some control, still hard impacts |
| -50 to 50    | Hover attempts, poor landing alignment |
| 50 to 150    | Controlled descents, inconsistent touchdowns |
| 150 to 250   | Soft landings with occasional bumps |
| 200+         | ✅ SOLVED: stable, repeatable landings |

Key signals in learned networks:
- Angle & angular velocity strongly weighted (keep upright).
- Main engine connections become moderated to save fuel.
- Leg contact inputs gain influence late in training (soft touchdown).

## 🎬 Animation Interpretation

When watching `neat_evolution.gif`:
1. Early frames: sparse edges, no hidden nodes, chaotic behavior.
2. Mid frames: hidden nodes appear; more connections; colors intensify as weights strengthen.
3. Late frames: topology stabilizes; minor weight tweaks dominate.

Use this guide as you inspect CSVs, plots, and GIFs to confirm NEAT is making steady progress toward consistent landings.
