#!/usr/bin/env python3
"""
Training Report Generator
==========================
Generates a human-readable markdown report summarizing a training run.
"""

import pandas as pd
import json
import os
import sys
from datetime import datetime
from utils.file_manager import get_run_paths


def generate_training_summary(run_dir):
    """Generate a comprehensive markdown summary of the training run."""

    paths = get_run_paths(run_dir)

    # Load data
    if not os.path.exists(paths['fitness_csv']):
        print(f"❌ Fitness history not found")
        return

    fitness_df = pd.read_csv(paths['fitness_csv'])

    complexity_df = None
    if os.path.exists(paths['complexity_csv']):
        complexity_df = pd.read_csv(paths['complexity_csv'])

    # Load analysis if available
    analysis_file = os.path.join(paths.get('analysis_dir', run_dir), 'training_analysis.json')
    analysis = None
    if os.path.exists(analysis_file):
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)

    # Generate markdown report
    report_lines = []

    # Header
    run_name = os.path.basename(run_dir)
    report_lines.append(f"# Training Summary: {run_name}\n")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("---\n")

    # Overview section
    report_lines.append("## 📊 Overview\n")
    report_lines.append(f"- **Total Generations**: {len(fitness_df)}\n")
    report_lines.append(f"- **Final Best Fitness**: {fitness_df.iloc[-1]['best_fitness']:.1f}\n")
    report_lines.append(f"- **Final Average Fitness**: {fitness_df.iloc[-1]['average_fitness']:.1f}\n")
    report_lines.append(f"- **Final Species Count**: {fitness_df.iloc[-1]['num_species']}\n")

    if complexity_df is not None:
        final_nodes = complexity_df.iloc[-1]['avg_nodes']
        final_conns = complexity_df.iloc[-1]['avg_connections']
        report_lines.append(f"- **Final Complexity**: {final_nodes:.1f} nodes, {final_conns:.1f} connections\n")

    report_lines.append("\n")

    # Fitness progression
    report_lines.append("## 📈 Fitness Progression\n")
    report_lines.append(f"- **Starting Best**: {fitness_df.iloc[0]['best_fitness']:.1f}\n")
    report_lines.append(f"- **Ending Best**: {fitness_df.iloc[-1]['best_fitness']:.1f}\n")
    improvement = fitness_df.iloc[-1]['best_fitness'] - fitness_df.iloc[0]['best_fitness']
    report_lines.append(f"- **Total Improvement**: {improvement:.1f}\n")
    report_lines.append(f"- **Peak Fitness**: {fitness_df['best_fitness'].max():.1f} (Gen {fitness_df['best_fitness'].idxmax()})\n")
    report_lines.append("\n")

    # Key moments
    if analysis and 'key_moments' in analysis:
        moments = analysis['key_moments']

        report_lines.append("## 🎯 Key Moments\n")

        if moments.get('first_positive'):
            fp = moments['first_positive']
            report_lines.append(f"\n### First Positive Fitness\n")
            report_lines.append(f"- **Generation**: {fp['generation']}\n")
            report_lines.append(f"- **Fitness**: {fp['fitness']:.1f}\n")
            report_lines.append(f"- **Significance**: The network achieved positive reward for the first time!\n")

        if moments.get('breakthroughs'):
            report_lines.append(f"\n### Breakthroughs\n")
            for i, bt in enumerate(moments['breakthroughs'], 1):
                report_lines.append(f"\n**Breakthrough #{i}**\n")
                report_lines.append(f"- **Generation**: {bt['generation']}\n")
                report_lines.append(f"- **Jump**: +{bt['fitness_jump']:.1f} (from {bt['from']:.1f} to {bt['to']:.1f})\n")

        if complexity_df is not None and moments.get('complexity_peak'):
            peak = moments['complexity_peak']
            report_lines.append(f"\n### Complexity Peak\n")
            report_lines.append(f"- **Generation**: {peak['generation']}\n")
            report_lines.append(f"- **Nodes**: {peak['nodes']:.1f}\n")
            report_lines.append(f"- **Connections**: {peak['connections']:.1f}\n")
            report_lines.append(f"- **What happened**: Networks reached maximum structural complexity\n")

        if complexity_df is not None and moments.get('pruning'):
            report_lines.append(f"\n### Pruning Phases\n")
            for pr in moments['pruning']:
                report_lines.append(f"\n- **Generations {pr['start_gen']}-{pr['end_gen']}**: ")
                report_lines.append(f"Networks simplified from {pr['start_nodes']:.1f} to {pr['end_nodes']:.1f} nodes ")
                report_lines.append(f"({pr['nodes_removed']:.1f} removed)\n")

        report_lines.append("\n")

    # Evolution story
    report_lines.append("## 🧬 Evolution Story\n")

    # Divide into phases
    total_gens = len(fitness_df)
    early_phase = fitness_df.iloc[:total_gens//3]
    mid_phase = fitness_df.iloc[total_gens//3:2*total_gens//3]
    late_phase = fitness_df.iloc[2*total_gens//3:]

    report_lines.append(f"\n### Phase 1: Early Evolution (Generations 0-{total_gens//3})\n")
    early_avg = early_phase['best_fitness'].mean()
    report_lines.append(f"- Average best fitness: {early_avg:.1f}\n")
    if early_avg < 0:
        report_lines.append(f"- Networks were struggling, crashing frequently\n")
        report_lines.append(f"- Random exploration of the fitness landscape\n")
    else:
        report_lines.append(f"- Networks beginning to find successful strategies\n")

    report_lines.append(f"\n### Phase 2: Mid Evolution (Generations {total_gens//3}-{2*total_gens//3})\n")
    mid_avg = mid_phase['best_fitness'].mean()
    mid_improvement = mid_avg - early_avg
    report_lines.append(f"- Average best fitness: {mid_avg:.1f} ({'+' if mid_improvement > 0 else ''}{mid_improvement:.1f} from Phase 1)\n")

    report_lines.append(f"\n### Phase 3: Late Evolution (Generations {2*total_gens//3}-{total_gens})\n")
    late_avg = late_phase['best_fitness'].mean()
    late_improvement = late_avg - mid_avg
    report_lines.append(f"- Average best fitness: {late_avg:.1f} ({'+' if late_improvement > 0 else ''}{late_improvement:.1f} from Phase 2)\n")

    if complexity_df is not None:
        final_nodes = complexity_df.iloc[-1]['avg_nodes']
        initial_nodes = complexity_df.iloc[0]['avg_nodes']
        if final_nodes < initial_nodes:
            report_lines.append(f"- Networks simplified through pruning\n")
        elif final_nodes > initial_nodes * 1.5:
            report_lines.append(f"- Networks grew more complex to handle task demands\n")

    report_lines.append("\n")

    # Recommendations
    report_lines.append("## 💡 Recommendations\n")

    final_fitness = fitness_df.iloc[-1]['best_fitness']
    target = 270.0  # LunarLander target

    if final_fitness < target * 0.5:
        report_lines.append(f"\n### Performance: Below 50% of target\n")
        report_lines.append(f"- **Current**: {final_fitness:.1f} / {target} ({final_fitness/target*100:.1f}%)\n")
        report_lines.append(f"- **Suggestions**:\n")
        report_lines.append(f"  - Increase max_generations (try {total_gens * 2})\n")
        report_lines.append(f"  - Increase population size (try 200-300)\n")
        report_lines.append(f"  - Adjust mutation rates for more exploration\n")
    elif final_fitness < target:
        report_lines.append(f"\n### Performance: Close to target!\n")
        report_lines.append(f"- **Current**: {final_fitness:.1f} / {target} ({final_fitness/target*100:.1f}%)\n")
        report_lines.append(f"- **Suggestions**:\n")
        report_lines.append(f"  - Run for more generations to fine-tune\n")
        report_lines.append(f"  - Consider lowering mutation rates for refinement\n")
    else:
        report_lines.append(f"\n### Performance: Target reached! 🎉\n")
        report_lines.append(f"- **Achievement**: {final_fitness:.1f} / {target} ({final_fitness/target*100:.1f}%)\n")
        report_lines.append(f"- Excellent performance!\n")

    report_lines.append("\n---\n")
    report_lines.append(f"\n*Generated by NEAT LunarLander Training System*\n")

    # Save report
    analysis_dir = paths.get('analysis_dir', run_dir)
    os.makedirs(analysis_dir, exist_ok=True)

    output_file = os.path.join(analysis_dir, 'training_summary.md')
    with open(output_file, 'w') as f:
        f.writelines(report_lines)

    print(f"\n✅ Report generated!")
    print(f"💾 Saved to: {output_file}\n")

    return output_file


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\n❌ Usage: python generate_report.py <run_directory>\n")
        sys.exit(1)

    run_dir = sys.argv[1]

    if not os.path.exists(run_dir):
        print(f"\n❌ Error: Directory '{run_dir}' not found!\n")
        sys.exit(1)

    generate_training_summary(run_dir)
