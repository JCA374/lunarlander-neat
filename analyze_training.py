#!/usr/bin/env python3
"""
Training Analysis Script
========================
Analyzes a completed training run and detects key moments.
"""

import pandas as pd
import json
import os
import sys
from utils.file_manager import get_run_paths
from utils.analysis_helpers import (
    detect_key_moments,
    identify_evolution_phases,
    calculate_fitness_improvement
)


def analyze_run(run_dir):
    """Analyze a training run and generate key moments JSON."""

    paths = get_run_paths(run_dir)

    # Check for required files
    if not os.path.exists(paths['fitness_csv']):
        print(f"❌ Fitness history not found: {paths['fitness_csv']}")
        return

    print("\n" + "="*70)
    print("🔍 ANALYZING TRAINING RUN")
    print("="*70)
    print(f"📁 Run directory: {run_dir}")
    print("="*70 + "\n")

    # Load data
    fitness_df = pd.read_csv(paths['fitness_csv'])

    complexity_df = None
    if os.path.exists(paths['complexity_csv']):
        complexity_df = pd.read_csv(paths['complexity_csv'])

    # Detect key moments
    print("🔎 Detecting key moments...")
    moments = detect_key_moments(fitness_df, complexity_df)

    # Calculate improvement metrics
    print("📊 Calculating improvement metrics...")
    improvements = calculate_fitness_improvement(fitness_df)

    # Identify evolution phases
    print("🧬 Identifying evolution phases...")
    phases = identify_evolution_phases(fitness_df, complexity_df)

    # Combine results
    analysis = {
        'key_moments': moments,
        'improvements': improvements,
        'phases': phases,
        'summary': {
            'total_generations': int(len(fitness_df)),
            'final_best_fitness': float(fitness_df.iloc[-1]['best_fitness']),
            'final_avg_fitness': float(fitness_df.iloc[-1]['average_fitness']),
            'num_species_final': int(fitness_df.iloc[-1]['num_species'])
        }
    }

    # Save to analysis directory
    analysis_dir = paths.get('analysis_dir', os.path.join(run_dir, 'analysis'))
    os.makedirs(analysis_dir, exist_ok=True)

    output_file = os.path.join(analysis_dir, 'training_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n✅ Analysis complete!")
    print(f"💾 Saved to: {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("📊 ANALYSIS SUMMARY")
    print("="*70)

    if moments.get('first_positive'):
        fp = moments['first_positive']
        print(f"🎯 First positive fitness: Gen {fp['generation']} ({fp['fitness']:.1f})")

    if moments.get('breakthroughs'):
        print(f"🚀 Breakthroughs detected: {len(moments['breakthroughs'])}")
        for bt in moments['breakthroughs'][:3]:  # Show first 3
            print(f"   - Gen {bt['generation']}: +{bt['fitness_jump']:.1f}")

    if complexity_df is not None and moments.get('complexity_peak'):
        peak = moments['complexity_peak']
        print(f"🧠 Complexity peak: Gen {peak['generation']} ({peak['nodes']:.1f} nodes)")

    if complexity_df is not None and moments.get('pruning'):
        for pr in moments['pruning']:
            print(f"✂️  Pruning: Gen {pr['start_gen']}-{pr['end_gen']} ({pr['nodes_removed']:.1f} nodes removed)")

    print(f"\n📈 Total improvement: {improvements['total_best_improvement']:.1f}")
    print(f"🎯 Final fitness: {improvements['final_best_fitness']:.1f} ({improvements['percent_of_target']:.1f}% of target)")

    print("="*70 + "\n")

    return analysis


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\n❌ Usage: python analyze_training.py <run_directory>\n")
        sys.exit(1)

    run_dir = sys.argv[1]

    if not os.path.exists(run_dir):
        print(f"\n❌ Error: Directory '{run_dir}' not found!\n")
        sys.exit(1)

    analyze_run(run_dir)
