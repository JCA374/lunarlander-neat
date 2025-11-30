#!/usr/bin/env python3
"""
Watch Training Progress
=======================
Quickly test the latest saved genome from a training run.
Great for checking progress while training is still running!
"""

import os
import sys
import subprocess
import glob


def find_latest_genome(run_dir):
    """Find the most recent genome file in the run directory."""

    genome_dir = os.path.join(run_dir, 'genomes')

    if not os.path.exists(genome_dir):
        return None

    # Find all generation files (not winner.pkl)
    genome_files = glob.glob(os.path.join(genome_dir, 'g*.pkl'))

    if not genome_files:
        return None

    # Sort by modification time (most recent first)
    genome_files.sort(key=os.path.getmtime, reverse=True)

    return genome_files[0]


def extract_generation(genome_path):
    """Extract generation number from genome filename."""
    basename = os.path.basename(genome_path)
    # Extract number from 'g000.pkl' format
    if basename.startswith('g') and basename.endswith('.pkl'):
        try:
            return int(basename[1:-4])
        except ValueError:
            return None
    return None


def main():
    if len(sys.argv) < 2:
        print("\nUsage: python watch_progress.py <run_directory> [--render] [--episodes N]")
        print("\nQuickly test the latest genome from a training run")
        print("\nExamples:")
        print("  python watch_progress.py runs/run_20241130_163831")
        print("  python watch_progress.py runs/run_20241130_163831 --render")
        print("  python watch_progress.py runs/run_20241130_163831 --episodes 10")
        print("\nOptions:")
        print("  --render          Show the visualization (default: no render)")
        print("  --episodes N      Number of episodes to test (default: 3)")
        print("  --analyze         Just show network structure, don't run")
        print()
        sys.exit(1)

    run_dir = sys.argv[1]

    if not os.path.exists(run_dir):
        print(f"\n❌ Error: Directory '{run_dir}' not found!\n")
        sys.exit(1)

    # Find latest genome
    latest_genome = find_latest_genome(run_dir)

    if not latest_genome:
        print(f"\n❌ No genomes found in {run_dir}/genomes/")
        print("   Training may have just started. Wait for generation 0 to complete.\n")
        sys.exit(1)

    generation = extract_generation(latest_genome)

    print("\n" + "="*70)
    print("WATCH TRAINING PROGRESS")
    print("="*70)
    print(f"Run:        {run_dir}")
    print(f"Genome:     {os.path.basename(latest_genome)}")
    if generation is not None:
        print(f"Generation: {generation}")
    print("="*70 + "\n")

    # Build command
    cmd = ['python', 'demo_best.py', latest_genome]

    # Default: no render, 3 episodes
    episodes = 3
    render = False

    # Parse options
    if '--render' in sys.argv:
        render = True

    if '--episodes' in sys.argv:
        try:
            idx = sys.argv.index('--episodes')
            episodes = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("❌ Invalid episodes value, using default (3)")

    if '--analyze' in sys.argv:
        cmd.append('--analyze')
    else:
        if not render:
            cmd.append('--no-render')
        cmd.extend(['--episodes', str(episodes)])

    # Run demo
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
