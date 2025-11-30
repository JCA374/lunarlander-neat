"""
Network visualization helpers for fixed-position layouts.

Provides functions to calculate consistent node positions across
generations, enabling easy visual comparison of network evolution.
"""

import math
from typing import Dict, Set, Tuple, List


def identify_node_types(genome, config) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    Identify which nodes are inputs, outputs, and hidden.

    Args:
        genome: NEAT genome
        config: NEAT config object

    Returns:
        Tuple of (input_nodes, output_nodes, hidden_nodes) sets
    """
    input_nodes = set(range(-config.genome_config.num_inputs, 0))
    output_nodes = set(range(config.genome_config.num_outputs))
    hidden_nodes = set()

    for node_id in genome.nodes.keys():
        if node_id not in input_nodes and node_id not in output_nodes:
            hidden_nodes.add(node_id)

    return input_nodes, output_nodes, hidden_nodes


def get_fixed_node_positions(genome, config) -> Dict[int, Tuple[float, float]]:
    """
    Calculate fixed positions for all nodes in the network.

    This ensures consistent layouts across generations by:
    - Placing inputs on the left (x=0)
    - Placing outputs on the right (x=3)
    - Placing hidden nodes in the middle based on connectivity

    Args:
        genome: NEAT genome
        config: NEAT config object

    Returns:
        Dictionary mapping node_id to (x, y) position
    """
    positions = {}

    input_nodes, output_nodes, hidden_nodes = identify_node_types(genome, config)

    # Fixed X positions
    INPUT_X = 0.0
    OUTPUT_X = 3.0
    HIDDEN_X_MIN = 1.0
    HIDDEN_X_MAX = 2.0

    # Input nodes - evenly spaced on left side
    num_inputs = len(input_nodes)
    if num_inputs > 0:
        input_spacing = 1.0 if num_inputs == 1 else 8.0 / (num_inputs - 1)
        for i, node_id in enumerate(sorted(input_nodes, reverse=True)):
            y = i * input_spacing
            positions[node_id] = (INPUT_X, y)

    # Output nodes - evenly spaced on right side
    num_outputs = len(output_nodes)
    if num_outputs > 0:
        # Center outputs vertically relative to inputs
        output_start_y = (8.0 - (num_outputs - 1) * 2.0) / 2.0 if num_outputs > 1 else 4.0
        output_spacing = 2.0 if num_outputs > 1 else 0
        for i, node_id in enumerate(sorted(output_nodes)):
            y = output_start_y + i * output_spacing
            positions[node_id] = (OUTPUT_X, y)

    # Hidden nodes - position based on connectivity
    if hidden_nodes:
        hidden_positions = calculate_hidden_node_positions(
            genome, hidden_nodes, input_nodes, output_nodes, positions
        )
        positions.update(hidden_positions)

    return positions


def calculate_hidden_node_positions(
    genome, hidden_nodes: Set[int], input_nodes: Set[int],
    output_nodes: Set[int], io_positions: Dict[int, Tuple[float, float]]
) -> Dict[int, Tuple[float, float]]:
    """
    Calculate positions for hidden nodes based on their connectivity.

    Strategy:
    - X position: Based on average "layer depth" from inputs
    - Y position: Based on average Y of connected input/output nodes

    Args:
        genome: NEAT genome
        hidden_nodes: Set of hidden node IDs
        input_nodes: Set of input node IDs
        output_nodes: Set of output node IDs
        io_positions: Positions of input and output nodes

    Returns:
        Dictionary mapping hidden node_id to (x, y) position
    """
    positions = {}

    # Build connectivity graph
    incoming = {node_id: [] for node_id in hidden_nodes}
    outgoing = {node_id: [] for node_id in hidden_nodes}

    for conn in genome.connections.values():
        if not conn.enabled:
            continue

        in_node, out_node = conn.key

        if in_node in hidden_nodes:
            outgoing[in_node].append(out_node)
        if out_node in hidden_nodes:
            incoming[out_node].append(in_node)

    # Calculate positions for each hidden node
    for node_id in sorted(hidden_nodes):  # Sort for deterministic positioning
        # Calculate X position based on connectivity depth
        x_pos = calculate_hidden_x_position(
            node_id, incoming, outgoing, input_nodes, output_nodes, hidden_nodes
        )

        # Calculate Y position based on connected nodes
        y_pos = calculate_hidden_y_position(
            node_id, incoming, outgoing, io_positions, positions
        )

        positions[node_id] = (x_pos, y_pos)

    # Adjust positions to avoid overlaps
    positions = adjust_for_overlaps(positions)

    return positions


def calculate_hidden_x_position(
    node_id: int, incoming: Dict, outgoing: Dict,
    input_nodes: Set[int], output_nodes: Set[int],
    hidden_nodes: Set[int]
) -> float:
    """
    Calculate X position for a hidden node based on its connectivity layer.

    Nodes closer to inputs get lower X values, nodes closer to outputs get higher X values.

    Args:
        node_id: Hidden node ID
        incoming: Dictionary of incoming connections
        outgoing: Dictionary of outgoing connections
        input_nodes: Set of input node IDs
        output_nodes: Set of output node IDs
        hidden_nodes: Set of all hidden node IDs

    Returns:
        X position (between 1.0 and 2.0)
    """
    # Count connections to inputs vs outputs
    direct_from_input = sum(1 for n in incoming[node_id] if n in input_nodes)
    direct_to_output = sum(1 for n in outgoing[node_id] if n in output_nodes)

    total_direct = direct_from_input + direct_to_output

    if total_direct == 0:
        # No direct connections, use node ID for deterministic positioning
        return 1.5 + (abs(hash(node_id)) % 100) / 200.0  # 1.5 ± 0.25

    # Calculate position based on ratio of input vs output connections
    # More input connections -> closer to left (lower X)
    # More output connections -> closer to right (higher X)
    ratio = direct_to_output / total_direct

    # Map ratio [0, 1] to X range [1.0, 2.0]
    x_pos = 1.0 + ratio * 1.0

    # Add small offset based on node ID for separation
    offset = (abs(hash(node_id)) % 20) / 100.0  # ± 0.1
    x_pos += offset

    # Clamp to range
    return max(1.0, min(2.0, x_pos))


def calculate_hidden_y_position(
    node_id: int, incoming: Dict, outgoing: Dict,
    io_positions: Dict[int, Tuple[float, float]],
    calculated_positions: Dict[int, Tuple[float, float]]
) -> float:
    """
    Calculate Y position for a hidden node based on connected nodes.

    Takes average Y position of all connected nodes (inputs, outputs, other hidden).

    Args:
        node_id: Hidden node ID
        incoming: Dictionary of incoming connections
        outgoing: Dictionary of outgoing connections
        io_positions: Positions of input/output nodes
        calculated_positions: Already-calculated hidden node positions

    Returns:
        Y position (between 0.0 and 8.0)
    """
    y_values = []

    # Collect Y positions from incoming nodes
    for in_node in incoming[node_id]:
        if in_node in io_positions:
            y_values.append(io_positions[in_node][1])
        elif in_node in calculated_positions:
            y_values.append(calculated_positions[in_node][1])

    # Collect Y positions from outgoing nodes
    for out_node in outgoing[node_id]:
        if out_node in io_positions:
            y_values.append(io_positions[out_node][1])
        elif out_node in calculated_positions:
            y_values.append(calculated_positions[out_node][1])

    if y_values:
        return sum(y_values) / len(y_values)
    else:
        # No connections or all to uncalculated hidden nodes
        # Use node ID for deterministic positioning
        return 4.0 + (abs(hash(node_id)) % 40) / 10.0 - 2.0  # 4.0 ± 2.0


def adjust_for_overlaps(positions: Dict[int, Tuple[float, float]], min_distance: float = 0.5) -> Dict[int, Tuple[float, float]]:
    """
    Adjust node positions to avoid overlaps.

    Args:
        positions: Current node positions
        min_distance: Minimum distance between nodes

    Returns:
        Adjusted positions dictionary
    """
    adjusted = positions.copy()
    node_ids = list(adjusted.keys())

    # Simple collision detection and resolution
    max_iterations = 10
    for _ in range(max_iterations):
        moved = False

        for i, node1 in enumerate(node_ids):
            for node2 in node_ids[i+1:]:
                x1, y1 = adjusted[node1]
                x2, y2 = adjusted[node2]

                # Calculate distance
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                # If too close, push apart
                if dist < min_distance and dist > 0:
                    # Calculate push vector
                    push_x = (x2 - x1) / dist * (min_distance - dist) / 2
                    push_y = (y2 - y1) / dist * (min_distance - dist) / 2

                    # Apply push (only adjust Y, keep X stable)
                    adjusted[node1] = (x1, y1 - push_y)
                    adjusted[node2] = (x2, y2 + push_y)
                    moved = True

        if not moved:
            break

    return adjusted


def format_node_label(node_id: int, node, input_nodes: Set[int], output_nodes: Set[int]) -> str:
    """
    Format a human-readable label for a node.

    Args:
        node_id: Node ID
        node: Node object (or None for input nodes)
        input_nodes: Set of input node IDs
        output_nodes: Set of output node IDs

    Returns:
        Formatted label string
    """
    if node_id in input_nodes:
        # Input node labels
        input_labels = [
            'X Pos', 'Y Pos', 'X Vel', 'Y Vel',
            'Angle', 'Ang Vel', 'Left Leg', 'Right Leg'
        ]
        index = list(sorted(input_nodes, reverse=True)).index(node_id)
        return input_labels[index] if index < len(input_labels) else f'In{index}'

    elif node_id in output_nodes:
        # Output node labels
        output_labels = ['No\nOp', 'Left\nEngine', 'Main\nEngine', 'Right\nEngine']
        index = list(sorted(output_nodes)).index(node_id)
        return output_labels[index] if index < len(output_labels) else f'Out{index}'

    else:
        # Hidden node labels
        if node:
            activation = node.activation
            return f'{node_id}\n{activation[:4]}'
        else:
            return str(node_id)
