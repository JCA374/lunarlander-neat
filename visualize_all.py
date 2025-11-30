#!/usr/bin/env python3
"""
Master Visualization Script
============================
Runs all visualization and analysis scripts in sequence.

One command to generate everything!
"""

import os
import sys
import subprocess
import time


def run_command(script_name, run_dir, extra_args=None):
    """Run a visualization script and report status."""

    cmd = ['python', script_name, run_dir]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*70}")
    print(f"▶️  Running: {script_name}")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"\n✅ {script_name} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {script_name} failed after {elapsed:.1f}s")
        print(f"Error code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n⚠️  {script_name} not found, skipping")
        return False


def main(run_dir):
    """Run all visualization and analysis scripts."""

    if not os.path.exists(run_dir):
        print(f"\n❌ Error: Directory '{run_dir}' not found!\n")
        sys.exit(1)

    print("\n" + "="*70)
    print("🎨 GENERATING ALL VISUALIZATIONS AND ANALYSIS")
    print("="*70)
    print(f"📁 Run directory: {run_dir}")
    print(f"⏱️  This will take a few minutes...")
    print("="*70)

    start_time = time.time()
    results = {}

    # Step 1: Analyze training data
    print("\n" + "🔍 Step 1/6: Analyzing training data")
    results['analyze'] = run_command('analyze_training.py', run_dir)

    # Step 2: Generate report
    print("\n" + "📝 Step 2/6: Generating training summary")
    results['report'] = run_command('generate_report.py', run_dir)

    # Step 3: Visualize networks with fixed positions
    print("\n" + "🧠 Step 3/6: Creating network diagrams (fixed positions)")
    results['networks'] = run_command('visualize_network_v2.py', run_dir)

    # Step 4: Plot fitness curves
    print("\n" + "📈 Step 4/6: Plotting training curves")
    results['fitness'] = run_command('plot_fitness.py', run_dir)

    # Step 5: Create evolution GIF
    print("\n" + "🎬 Step 5/6: Creating evolution animation")
    results['evolution'] = run_command('visualize_evolution.py', run_dir)

    # Step 6: Analyze network structure
    print("\n" + "🔬 Step 6/6: Analyzing best network")
    results['demo'] = run_command('demo_best.py', run_dir, ['--analyze'])

    # Summary
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print("\n" + "="*70)
    print("📊 GENERATION COMPLETE!")
    print("="*70)
    print(f"⏱️  Total time: {minutes}m {seconds}s\n")

    # Show results
    print("Results:")
    for task, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {task}")

    # List generated files
    print("\n" + "="*70)
    print("📁 Generated files:")
    print("="*70)

    output_dirs = [
        ('Analysis', 'analysis'),
        ('Networks', 'visualizations/networks' if os.path.exists(os.path.join(run_dir, 'visualizations')) else 'networks'),
        ('Plots', 'visualizations/plots' if os.path.exists(os.path.join(run_dir, 'visualizations')) else '.'),
    ]

    for name, subdir in output_dirs:
        full_path = os.path.join(run_dir, subdir)
        if os.path.exists(full_path):
            files = [f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]
            if files:
                print(f"\n{name} ({len(files)} files):")
                print(f"  {full_path}/")
                for f in sorted(files)[:5]:  # Show first 5
                    print(f"    - {f}")
                if len(files) > 5:
                    print(f"    ... and {len(files) - 5} more")

    print("\n" + "="*70)
    print("💡 What to do next:")
    print("="*70)
    print(f"  1. View training summary:")
    print(f"     cat {run_dir}/analysis/training_summary.md")
    print(f"\n  2. Watch best agent play:")
    print(f"     python demo_best.py {run_dir}")
    print(f"\n  3. Open network diagrams:")
    networks_path = os.path.join(run_dir, 'visualizations/networks' if os.path.exists(os.path.join(run_dir, 'visualizations')) else 'networks')
    print(f"     Open files in: {networks_path}/")
    print(f"\n  4. View evolution GIF:")
    gif_path = os.path.join(run_dir, 'visualizations' if os.path.exists(os.path.join(run_dir, 'visualizations')) else '.', 'neat_evolution.gif')
    if os.path.exists(gif_path):
        print(f"     xdg-open {gif_path}")
    print("="*70 + "\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\n❌ Usage: python visualize_all.py <run_directory>")
        print("\nExample:")
        print("  python visualize_all.py runs/run_20241126_120000")
        print("\nThis will generate:")
        print("  - Training analysis (JSON)")
        print("  - Training summary (Markdown)")
        print("  - Network diagrams (PNG)")
        print("  - Fitness plots (PNG)")
        print("  - Evolution animation (GIF)")
        print("  - Network analysis report\n")
        sys.exit(1)

    run_dir = sys.argv[1]
    main(run_dir)
