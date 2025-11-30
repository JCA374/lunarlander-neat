#!/usr/bin/env python3
"""
Enhanced Network Visualization Script (V2)
===========================================
Creates network visualizations with FIXED NODE POSITIONS across generations.

Key improvements over v1:
- Consistent node placement (inputs left, outputs right, hidden in middle)
- Generation info overlay on each image
- Support for both old and new directory structures
- Deterministic layouts for easy comparison
"""

import neat
import pickle
import graphviz
import os
import sys
from glob import glob
from utils.file_manager import get_run_paths, find_genomes, is_old_structure
from utils.network_helpers import (
    get_fixed_node_positions,
    identify_node_types,
    format_node_label
)


def visualize_network_fixed(genome, config, filename=None, view=False,
                            show_disabled=True, generation=None, fitness=None):
    """
    Visualize a NEAT genome as a neural network with fixed node positions.

    Args:
        genome: NEAT genome to visualize
        config: NEAT config object
        filename: Output filename (without extension)
        view: Whether to open the image after creating it
        show_disabled: Whether to show disabled connections
        generation: Optional generation number for overlay
        fitness: Optional fitness score for overlay
    """

    # Get fixed positions for all nodes
    positions = get_fixed_node_positions(genome, config)

    # Identify node types
    input_nodes, output_nodes, hidden_nodes = identify_node_types(genome, config)

    # Create a directed graph with manual positioning
    dot = graphviz.Digraph(format='png', engine='neato')  # neato allows manual positioning

    # Graph attributes
    dot.attr(overlap='false', splines='true', sep='+0.5')

    # Node attributes
    dot.attr('node', shape='circle', style='filled', fixedsize='true',
             width='0.8', fontname='Arial', fontsize='11', fontcolor='black')

    # Add input nodes with fixed positions
    for node_id in sorted(input_nodes, reverse=True):
        x, y = positions[node_id]
        label = format_node_label(node_id, None, input_nodes, output_nodes)
        dot.node(str(node_id), label=label, fillcolor='lightblue',
                pos=f'{x},{y}!', pin='true')

    # Add output nodes with fixed positions
    for node_id in sorted(output_nodes):
        x, y = positions[node_id]
        label = format_node_label(node_id, None, input_nodes, output_nodes)
        dot.node(str(node_id), label=label, fillcolor='lightcoral',
                pos=f'{x},{y}!', pin='true')

    # Add hidden nodes with calculated positions
    for node_id in sorted(hidden_nodes):
        x, y = positions[node_id]
        node = genome.nodes[node_id]
        label = format_node_label(node_id, node, input_nodes, output_nodes)
        dot.node(str(node_id), label=label, fillcolor='lightgreen',
                pos=f'{x},{y}!', pin='true')

    # Add connections
    for cg in genome.connections.values():
        if not show_disabled and not cg.enabled:
            continue

        input_node, output_node = cg.key
        weight = cg.weight

        # Determine color based on weight (green=positive, red=negative)
        if weight > 0:
            intensity = min(abs(weight) / 5.0, 1.0)
            green_value = int(255 * (1 - 0.7 * intensity))
            color = f'#00{green_value:02x}00'
        else:
            intensity = min(abs(weight) / 5.0, 1.0)
            red_value = int(255 * (1 - 0.7 * intensity))
            color = f'#{red_value:02x}0000'

        # Line thickness based on weight magnitude
        penwidth = str(0.5 + 3.0 * min(abs(weight) / 5.0, 1.0))

        # Style for disabled connections
        style = 'dashed' if not cg.enabled else 'solid'

        # Add edge with label showing weight
        dot.edge(str(input_node), str(output_node),
                label=f'{weight:.2f}',
                color=color,
                penwidth=penwidth,
                style=style,
                fontsize='8',
                fontcolor='black')

    # Add generation and fitness info as a label
    if generation is not None or fitness is not None:
        info_parts = []
        if generation is not None:
            info_parts.append(f'Generation {generation}')
        if fitness is not None:
            info_parts.append(f'Fitness: {fitness:.1f}')

        num_nodes = len(genome.nodes)
        num_connections = len([c for c in genome.connections.values() if c.enabled])
        info_parts.append(f'{num_nodes} nodes, {num_connections} connections')

        label_text = '\\n'.join(info_parts)
        dot.attr(label=label_text, fontsize='12', fontname='Arial Bold',
                labelloc='t', labeljust='c')

    # Render the graph
    if filename:
        dot.render(filename, view=view, cleanup=True)
        return filename + '.png'
    else:
        return dot


def visualize_all_generations_fixed(run_dir, view_last=False):
    """
    Visualize all saved genomes from a training run with fixed positions.

    Args:
        run_dir: Directory containing the training run
        view_last: Whether to open the last generation's network
    """

    # Get paths (supports both old and new structures)
    paths = get_run_paths(run_dir)

    # Create networks directory
    networks_dir = paths['networks_dir']
    os.makedirs(networks_dir, exist_ok=True)

    # Load config
    config_file = paths['config']
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Find all genome files
    genomes = find_genomes(run_dir)

    if not genomes:
        print(f"❌ No genome files found in {run_dir}")
        return

    print("\n" + "="*70)
    print("🎨 VISUALIZING NEURAL NETWORKS (Fixed Positions)")
    print("="*70)
    print(f"📁 Run directory: {run_dir}")
    print(f"🧬 Found {len(genomes)} generations")
    print(f"📌 Using fixed node positions for consistency")
    print("="*70 + "\n")

    # Visualize each generation
    for i, (gen_num, genome_file) in enumerate(genomes):
        # Load genome
        with open(genome_file, 'rb') as f:
            genome = pickle.load(f)

        # Create visualization
        output_file = os.path.join(networks_dir, f'net_g{gen_num:03d}')

        # Only view the last one if requested
        view = view_last and (i == len(genomes) - 1)

        visualize_network_fixed(
            genome, config,
            filename=output_file,
            view=view,
            generation=gen_num,
            fitness=genome.fitness if hasattr(genome, 'fitness') and genome.fitness else None
        )

        # Print progress
        num_nodes = len(genome.nodes)
        num_connections = len([c for c in genome.connections.values() if c.enabled])
        fitness = genome.fitness if hasattr(genome, 'fitness') and genome.fitness else 0.0

        print(f"  ✓ Gen {gen_num:3d}: {num_nodes:2d} nodes, "
              f"{num_connections:2d} connections, fitness={fitness:6.1f}")

    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"📊 Saved {len(genomes)} network diagrams to:")
    print(f"   {networks_dir}/")
    print("\n💡 Key features:")
    print("   - Fixed node positions (easy to compare across generations)")
    print("   - Generation and fitness info on each image")
    print("   - Color-coded connection weights")
    print("\n📈 Next steps:")
    print(f"   - Create evolution GIF: python visualize_evolution.py {run_dir}")
    print(f"   - Generate comparisons: python visualize_comparison.py {run_dir}")
    print(f"   - Plot training curves: python plot_fitness.py {run_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\n❌ Usage: python visualize_network_v2.py <run_directory>")
        print("\nExample:")
        print("  python visualize_network_v2.py runs/run_20241126_120000\n")
        print("\nOptions:")
        print("  --view, -v    Open the last network visualization\n")
        sys.exit(1)

    run_dir = sys.argv[1]

    if not os.path.exists(run_dir):
        print(f"\n❌ Error: Directory '{run_dir}' not found!\n")
        sys.exit(1)

    # Check if we should view the last network
    view_last = '--view' in sys.argv or '-v' in sys.argv

    visualize_all_generations_fixed(run_dir, view_last=view_last)
