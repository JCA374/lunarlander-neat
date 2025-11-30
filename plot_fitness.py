#!/usr/bin/env python3
"""
Fitness Plotting Script
=======================
Creates comprehensive visualizations of training progress:
- Fitness over generations (best/avg/worst)
- Species diversity over time
- Network complexity evolution
- Multiple subplots for detailed analysis
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from glob import glob
import configparser


def plot_fitness_history(run_dir, show=False):
    """
    Create comprehensive training visualizations.

    Args:
        run_dir: Directory containing the training run
        show: Whether to display the plots
    """

    # Support both old and new directory structures
    fitness_file = os.path.join(run_dir, 'data', 'fitness.csv')
    complexity_file = os.path.join(run_dir, 'data', 'complexity.csv')

    # Fallback to old structure if new structure doesn't exist
    if not os.path.exists(fitness_file):
        fitness_file = os.path.join(run_dir, 'fitness_history.csv')
        complexity_file = os.path.join(run_dir, 'complexity_history.csv')

    if not os.path.exists(fitness_file):
        print(f"❌ Error: Fitness history file not found: {fitness_file}")
        return

    # Load data
    fitness_df = pd.read_csv(fitness_file)

    print("\n" + "="*70)
    print("📊 PLOTTING TRAINING STATISTICS")
    print("="*70)
    print(f"📁 Run directory: {run_dir}")
    print(f"📈 Generations: {len(fitness_df)}")
    print("="*70 + "\n")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    target_fitness = 270
    config_file = os.path.join(run_dir, 'config.ini')
    if os.path.exists(config_file):
        cfg = configparser.ConfigParser()
        cfg.read(config_file)
        try:
            target_fitness = cfg.getfloat('NEAT', 'fitness_threshold')
        except Exception:
            pass

    # ==========================================
    # Plot 1: Fitness over generations
    # ==========================================
    ax1 = plt.subplot(2, 2, 1)

    generations = fitness_df['generation']
    best_fitness = fitness_df['best_fitness']
    avg_fitness = fitness_df['average_fitness']
    worst_fitness = fitness_df['worst_fitness']
    y_min = min(0, worst_fitness.min())

    # Plot lines
    ax1.plot(generations, best_fitness, 'g-', linewidth=2.5, label='Best', marker='o', markersize=4)
    ax1.plot(generations, avg_fitness, 'b-', linewidth=2, label='Average', marker='s', markersize=3)

    # Fill between for range (use avg instead of worst to avoid extreme negative values)
    ax1.fill_between(generations, avg_fitness, best_fitness, alpha=0.2, color='green')

    # Target line
    ax1.axhline(y=target_fitness, color='gold', linestyle='--', linewidth=2,
                label=f'Target ({target_fitness})', alpha=0.8)

    ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fitness', fontsize=12, fontweight='bold')
    ax1.set_title('Fitness Evolution', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    # Use avg_fitness min for better autoscaling (avoids extreme outliers)
    ax1.set_ylim(bottom=min(0, avg_fitness.min() * 1.1))

    # ==========================================
    # Plot 2: Fitness with standard deviation
    # ==========================================
    ax2 = plt.subplot(2, 2, 2)

    std_fitness = fitness_df['std_fitness']

    ax2.plot(generations, avg_fitness, 'b-', linewidth=2.5, label='Average', marker='o', markersize=4)
    ax2.fill_between(generations,
                     avg_fitness - std_fitness,
                     avg_fitness + std_fitness,
                     alpha=0.3, color='blue', label='±1 Std Dev')

    ax2.axhline(y=target_fitness, color='gold', linestyle='--', linewidth=2,
                label=f'Target ({target_fitness})', alpha=0.8)

    ax2.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fitness', fontsize=12, fontweight='bold')
    ax2.set_title('Average Fitness ± Standard Deviation', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)
    # Autoscale based on avg ± std (not extreme outliers)
    y_min_plot2 = (avg_fitness - std_fitness).min()
    ax2.set_ylim(bottom=min(0, y_min_plot2 * 1.1))

    # ==========================================
    # Plot 3: Species diversity
    # ==========================================
    ax3 = plt.subplot(2, 2, 3)

    num_species = fitness_df['num_species']

    ax3.plot(generations, num_species, 'purple', linewidth=2.5, marker='o', markersize=4)
    ax3.fill_between(generations, 0, num_species, alpha=0.3, color='purple')

    ax3.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Species', fontsize=12, fontweight='bold')
    ax3.set_title('Species Diversity', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0)

    # ==========================================
    # Plot 4: Network complexity
    # ==========================================
    ax4 = plt.subplot(2, 2, 4)

    if os.path.exists(complexity_file):
        complexity_df = pd.read_csv(complexity_file)

        avg_nodes = complexity_df['avg_nodes']
        avg_connections = complexity_df['avg_connections']

        ax4_twin = ax4.twinx()

        line1 = ax4.plot(generations, avg_nodes, 'darkgreen', linewidth=2.5,
                        label='Avg Nodes', marker='o', markersize=4)
        line2 = ax4_twin.plot(generations, avg_connections, 'darkred', linewidth=2.5,
                             label='Avg Connections', marker='s', markersize=4)

        ax4.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Average Nodes', fontsize=12, fontweight='bold', color='darkgreen')
        ax4_twin.set_ylabel('Average Connections', fontsize=12, fontweight='bold', color='darkred')

        ax4.tick_params(axis='y', labelcolor='darkgreen')
        ax4_twin.tick_params(axis='y', labelcolor='darkred')

        ax4.set_title('Network Complexity Evolution', fontsize=14, fontweight='bold', pad=15)

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left', framealpha=0.9)

        ax4.grid(True, alpha=0.3, linestyle='--')
    else:
        ax4.text(0.5, 0.5, 'Complexity data not available',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Network Complexity Evolution', fontsize=14, fontweight='bold', pad=15)
        ax4.set_xlim(left=0)
        ax4.set_ylim(bottom=0)
    ax4.set_xlim(left=0)
    ax4.set_ylim(bottom=0)
    if 'ax4_twin' in locals():
        ax4_twin.set_ylim(bottom=0)

    # Add overall title with statistics
    final_best = best_fitness.iloc[-1]
    final_avg = avg_fitness.iloc[-1]
    final_species = num_species.iloc[-1]

    fig.suptitle(f'NEAT LunarLander Training Results\n'
                f'Final Best: {final_best:.1f} | '
                f'Final Avg: {final_avg:.1f} | '
                f'Species: {final_species}',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure to visualizations directory
    viz_dir = os.path.join(run_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    output_file = os.path.join(viz_dir, 'training_plots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comprehensive plot to: {output_file}\n")

    # ==========================================
    # Create simple fitness plot
    # ==========================================
    fig2, ax = plt.subplots(figsize=(12, 6))

    ax.plot(generations, best_fitness, 'g-', linewidth=3, label='Best Fitness', marker='o', markersize=5)
    ax.plot(generations, avg_fitness, 'b-', linewidth=2.5, label='Average Fitness', marker='s', markersize=4)
    ax.fill_between(generations, avg_fitness, best_fitness, alpha=0.2, color='green', label='Best-Avg Range')

    ax.axhline(y=target_fitness, color='gold', linestyle='--', linewidth=2.5,
               label=f'Target ({target_fitness})', alpha=0.8)

    ax.set_xlabel('Generation', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fitness', fontsize=14, fontweight='bold')
    ax.set_title('NEAT LunarLander: Fitness Over Generations', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    # Better autoscaling - use average fitness minimum
    ax.set_ylim(bottom=min(0, avg_fitness.min() * 1.1))

    plt.tight_layout()

    # Save simple figure to visualizations directory
    simple_output = os.path.join(viz_dir, 'fitness_plot.png')
    plt.savefig(simple_output, dpi=300, bbox_inches='tight')
    print(f"✅ Saved simple fitness plot to: {simple_output}\n")

    print("="*70)
    print("📊 STATISTICS SUMMARY")
    print("="*70)
    print(f"  Starting best fitness: {best_fitness.iloc[0]:.1f}")
    print(f"  Final best fitness:    {final_best:.1f}")
    print(f"  Improvement:           {final_best - best_fitness.iloc[0]:.1f}")
    print(f"  Target reached:        {'✅ YES' if final_best >= target_fitness else '❌ NO'}")
    print(f"  Final species count:   {final_species}")
    print("="*70 + "\n")

    if show:
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\n❌ Usage: python plot_fitness.py <run_directory>")
        print("\nExample:")
        print("  python plot_fitness.py runs/run_20241126_120000")
        print("\nOptions:")
        print("  --show, -s    Display plots after saving\n")
        sys.exit(1)

    run_dir = sys.argv[1]

    if not os.path.exists(run_dir):
        print(f"\n❌ Error: Directory '{run_dir}' not found!\n")
        sys.exit(1)

    show = '--show' in sys.argv or '-s' in sys.argv

    plot_fitness_history(run_dir, show=show)
