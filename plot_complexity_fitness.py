#!/usr/bin/env python3
"""
Complexity vs Fitness Visualization
===================================
Shows how network complexity (nodes & connections) relates to fitness over time.
Perfect for understanding how NEAT grows networks as performance improves.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import configparser


def plot_complexity_vs_fitness(run_dir, show=False):
    """
    Create visualizations showing the relationship between complexity and fitness.

    Args:
        run_dir: Directory containing the training run
        show: Whether to display the plots
    """

    # Load data files
    fitness_file = os.path.join(run_dir, 'data', 'fitness.csv')
    complexity_file = os.path.join(run_dir, 'data', 'complexity.csv')

    # Fallback to old structure
    if not os.path.exists(fitness_file):
        fitness_file = os.path.join(run_dir, 'fitness_history.csv')
        complexity_file = os.path.join(run_dir, 'complexity_history.csv')

    if not os.path.exists(fitness_file) or not os.path.exists(complexity_file):
        print(f"Error: Required data files not found in {run_dir}")
        return

    fitness_df = pd.read_csv(fitness_file)
    complexity_df = pd.read_csv(complexity_file)

    # Get target fitness from config
    target_fitness = 250
    config_file = os.path.join(run_dir, 'config.ini')
    if os.path.exists(config_file):
        cfg = configparser.ConfigParser()
        cfg.read(config_file)
        try:
            target_fitness = cfg.getfloat('NEAT', 'fitness_threshold')
        except Exception:
            pass

    print("\n" + "="*70)
    print("COMPLEXITY vs FITNESS VISUALIZATION")
    print("="*70)
    print(f"Run directory: {run_dir}")
    print(f"Generations:   {len(fitness_df)}")
    print(f"Target:        {target_fitness}")
    print("="*70 + "\n")

    generations = fitness_df['generation']
    best_fitness = fitness_df['best_fitness']
    avg_fitness = fitness_df['average_fitness']
    avg_nodes = complexity_df['avg_nodes']
    avg_connections = complexity_df['avg_connections']
    max_nodes = complexity_df['max_nodes']
    max_connections = complexity_df['max_connections']

    # ==========================================
    # Figure 1: Three synchronized plots
    # ==========================================
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Fitness over time
    ax1.plot(generations, best_fitness, 'g-', linewidth=2.5, label='Best Fitness', marker='o', markersize=4)
    ax1.plot(generations, avg_fitness, 'b--', linewidth=2, label='Average Fitness', alpha=0.7)
    ax1.axhline(y=target_fitness, color='gold', linestyle='--', linewidth=2,
                label=f'Target ({target_fitness:.0f})', alpha=0.8)
    ax1.axhline(y=200, color='orange', linestyle=':', linewidth=1.5,
                label='Solved (200)', alpha=0.6)
    ax1.fill_between(generations, avg_fitness, best_fitness, alpha=0.2, color='green')

    ax1.set_ylabel('Fitness', fontsize=12, fontweight='bold')
    ax1.set_title('Fitness Evolution', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=min(0, avg_fitness.min() * 1.1))

    # Plot 2: Node count over time
    ax2.plot(generations, avg_nodes, 'darkgreen', linewidth=2.5, label='Average Nodes', marker='s', markersize=4)
    ax2.plot(generations, max_nodes, 'lightgreen', linewidth=1.5, label='Max Nodes',
             linestyle='--', marker='o', markersize=3, alpha=0.7)
    ax2.fill_between(generations, avg_nodes, max_nodes, alpha=0.2, color='green')

    ax2.set_ylabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax2.set_title('Network Size (Nodes)', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0)

    # Plot 3: Connection count over time
    ax3.plot(generations, avg_connections, 'darkred', linewidth=2.5, label='Average Connections',
             marker='s', markersize=4)
    ax3.plot(generations, max_connections, 'lightcoral', linewidth=1.5, label='Max Connections',
             linestyle='--', marker='o', markersize=3, alpha=0.7)
    ax3.fill_between(generations, avg_connections, max_connections, alpha=0.2, color='red')

    ax3.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Connections', fontsize=12, fontweight='bold')
    ax3.set_title('Network Connectivity', fontsize=13, fontweight='bold', pad=10)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim(bottom=0)

    # Overall title
    fig.suptitle('How Network Complexity Grows During Training',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    viz_dir = os.path.join(run_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    output_file = os.path.join(viz_dir, 'complexity_vs_fitness.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    # ==========================================
    # Figure 2: Scatter plot - Complexity vs Fitness
    # ==========================================
    fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(14, 6))

    # Filter data: only include fitness >= 150
    import numpy as np
    fitness_mask = best_fitness >= 150
    filtered_nodes = avg_nodes[fitness_mask]
    filtered_connections = avg_connections[fitness_mask]
    filtered_fitness = best_fitness[fitness_mask]
    filtered_generations = generations[fitness_mask]

    # Color by generation (early = blue, late = red)
    if len(filtered_generations) > 0:
        colors = plt.cm.viridis(filtered_generations / generations.max())
    else:
        print("Warning: No data points with fitness >= 150")
        colors = 'blue'

    # Nodes vs Fitness with trend line
    scatter1 = ax4.scatter(filtered_nodes, filtered_fitness, c=filtered_generations, cmap='viridis',
                          s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Add linear regression trend line for nodes
    if len(filtered_nodes) > 1:
        z_nodes = np.polyfit(filtered_nodes, filtered_fitness, 1)
        p_nodes = np.poly1d(z_nodes)
        x_nodes_line = np.linspace(filtered_nodes.min(), filtered_nodes.max(), 100)
        ax4.plot(x_nodes_line, p_nodes(x_nodes_line), "r--", linewidth=2, alpha=0.8, label='Trend')

        # Calculate R²
        y_pred_nodes = p_nodes(filtered_nodes)
        ss_res_nodes = np.sum((filtered_fitness - y_pred_nodes) ** 2)
        ss_tot_nodes = np.sum((filtered_fitness - np.mean(filtered_fitness)) ** 2)
        r2_nodes = 1 - (ss_res_nodes / ss_tot_nodes) if ss_tot_nodes > 0 else 0

        # Display R² on plot
        ax4.text(0.05, 0.95, f'R² = {r2_nodes:.3f}', transform=ax4.transAxes,
                fontsize=12, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax4.axhline(y=target_fitness, color='gold', linestyle='--', linewidth=2, alpha=0.7)
    ax4.axhline(y=200, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    ax4.set_xlabel('Average Number of Nodes', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
    ax4.set_title('Fitness vs Network Size (Fitness >= 250)', fontsize=13, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(loc='lower right', fontsize=10)
    cbar1 = plt.colorbar(scatter1, ax=ax4)
    cbar1.set_label('Generation', fontsize=10, fontweight='bold')

    # Connections vs Fitness with trend line
    scatter2 = ax5.scatter(filtered_connections, filtered_fitness, c=filtered_generations, cmap='viridis',
                          s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Add linear regression trend line for connections
    if len(filtered_connections) > 1:
        z_conn = np.polyfit(filtered_connections, filtered_fitness, 1)
        p_conn = np.poly1d(z_conn)
        x_conn_line = np.linspace(filtered_connections.min(), filtered_connections.max(), 100)
        ax5.plot(x_conn_line, p_conn(x_conn_line), "r--", linewidth=2, alpha=0.8, label='Trend')

        # Calculate R²
        y_pred_conn = p_conn(filtered_connections)
        ss_res_conn = np.sum((filtered_fitness - y_pred_conn) ** 2)
        ss_tot_conn = np.sum((filtered_fitness - np.mean(filtered_fitness)) ** 2)
        r2_conn = 1 - (ss_res_conn / ss_tot_conn) if ss_tot_conn > 0 else 0

        # Display R² on plot
        ax5.text(0.05, 0.95, f'R² = {r2_conn:.3f}', transform=ax5.transAxes,
                fontsize=12, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax5.axhline(y=target_fitness, color='gold', linestyle='--', linewidth=2, alpha=0.7)
    ax5.axhline(y=200, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    ax5.set_xlabel('Average Number of Connections', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
    ax5.set_title('Fitness vs Network Connectivity (Fitness >= 150)', fontsize=13, fontweight='bold', pad=10)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.legend(loc='lower right', fontsize=10)
    cbar2 = plt.colorbar(scatter2, ax=ax5)
    cbar2.set_label('Generation', fontsize=10, fontweight='bold')

    fig2.suptitle('Does More Complexity Mean Better Performance?',
                  fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    output_file2 = os.path.join(viz_dir, 'scatter_complexity_fitness.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file2}")

    # ==========================================
    # Figure 3: Combined dual-axis plot
    # ==========================================
    fig3, ax6 = plt.subplots(figsize=(14, 7))
    ax7 = ax6.twinx()

    # Fitness on left axis
    line1 = ax6.plot(generations, best_fitness, 'g-', linewidth=3,
                     label='Best Fitness', marker='o', markersize=5)
    ax6.axhline(y=target_fitness, color='gold', linestyle='--', linewidth=2, alpha=0.7)

    # Complexity on right axis
    line2 = ax7.plot(generations, avg_nodes, 'darkblue', linewidth=2.5,
                     label='Avg Nodes', marker='s', markersize=4, linestyle='--')
    line3 = ax7.plot(generations, avg_connections, 'darkred', linewidth=2.5,
                     label='Avg Connections', marker='^', markersize=4, linestyle='--')

    ax6.set_xlabel('Generation', fontsize=13, fontweight='bold')
    ax6.set_ylabel('Fitness', fontsize=13, fontweight='bold', color='green')
    ax7.set_ylabel('Network Complexity (Nodes/Connections)', fontsize=13, fontweight='bold', color='blue')

    ax6.tick_params(axis='y', labelcolor='green')
    ax7.tick_params(axis='y', labelcolor='blue')

    ax6.set_title('Fitness and Complexity Growth Over Time', fontsize=15, fontweight='bold', pad=15)

    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.9)

    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.set_xlim(left=0)
    ax6.set_ylim(bottom=min(0, best_fitness.min() * 1.1))
    ax7.set_ylim(bottom=0)

    plt.tight_layout()

    # Save
    output_file3 = os.path.join(viz_dir, 'combined_complexity_fitness.png')
    plt.savefig(output_file3, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file3}")

    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nFitness:")
    print(f"  Starting:  {best_fitness.iloc[0]:7.1f}")
    print(f"  Final:     {best_fitness.iloc[-1]:7.1f}")
    print(f"  Change:    {best_fitness.iloc[-1] - best_fitness.iloc[0]:+7.1f}")

    print(f"\nNetwork Nodes:")
    print(f"  Starting:  {avg_nodes.iloc[0]:7.1f}")
    print(f"  Final:     {avg_nodes.iloc[-1]:7.1f}")
    print(f"  Change:    {avg_nodes.iloc[-1] - avg_nodes.iloc[0]:+7.1f}")

    print(f"\nNetwork Connections:")
    print(f"  Starting:  {avg_connections.iloc[0]:7.1f}")
    print(f"  Final:     {avg_connections.iloc[-1]:7.1f}")
    print(f"  Change:    {avg_connections.iloc[-1] - avg_connections.iloc[0]:+7.1f}")

    # Calculate correlation (all data)
    import numpy as np
    corr_nodes = np.corrcoef(avg_nodes, best_fitness)[0, 1]
    corr_connections = np.corrcoef(avg_connections, best_fitness)[0, 1]

    print(f"\nCorrelations with Best Fitness (All Data):")
    print(f"  Nodes:        {corr_nodes:+.3f}")
    print(f"  Connections:  {corr_connections:+.3f}")

    # Calculate R² for filtered data (fitness >= 150)
    if len(filtered_nodes) > 1:
        print(f"\nR² Values (Fitness >= 150):")
        print(f"  Nodes:        {r2_nodes:.3f}")
        print(f"  Connections:  {r2_conn:.3f}")
        print(f"  Data points:  {len(filtered_nodes)} of {len(best_fitness)} total")
    else:
        print(f"\nR² Values: Not enough data points with fitness >= 150")

    print(f"\n  (R² closer to 1 means stronger linear relationship)")
    print(f"  (Correlation closer to +1 means stronger positive relationship)")
    print("="*70 + "\n")

    if show:
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\nUsage: python plot_complexity_fitness.py <run_directory>")
        print("\nExample:")
        print("  python plot_complexity_fitness.py runs/run_20241127_080353")
        print("\nOptions:")
        print("  --show, -s    Display plots after saving\n")
        sys.exit(1)

    run_dir = sys.argv[1]

    if not os.path.exists(run_dir):
        print(f"\nError: Directory '{run_dir}' not found!\n")
        sys.exit(1)

    show = '--show' in sys.argv or '-s' in sys.argv

    plot_complexity_vs_fitness(run_dir, show=show)
