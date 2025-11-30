"""
Training analysis helpers for detecting key moments and evolution phases.

Provides functions to automatically identify important events during
NEAT evolution, such as breakthroughs, bloat, pruning, and stagnation.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


def detect_key_moments(fitness_df: pd.DataFrame, complexity_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Detect key moments in training evolution.

    Args:
        fitness_df: DataFrame with fitness history
        complexity_df: Optional DataFrame with complexity history

    Returns:
        Dictionary of detected key moments
    """
    moments = {}

    # First positive fitness
    first_positive = detect_first_positive_fitness(fitness_df)
    if first_positive:
        moments['first_positive'] = first_positive

    # Breakthroughs (large fitness jumps)
    breakthroughs = detect_breakthrough(fitness_df, threshold=50)
    if breakthroughs:
        moments['breakthroughs'] = breakthroughs

    # Stagnation periods
    stagnation = detect_stagnation(fitness_df, window=10)
    if stagnation:
        moments['stagnation_periods'] = stagnation

    if complexity_df is not None:
        # Complexification spikes
        complexification = detect_complexification(complexity_df, threshold=2)
        if complexification:
            moments['complexification'] = complexification

        # Pruning phases
        pruning = detect_pruning(complexity_df, threshold=2)
        if pruning:
            moments['pruning'] = pruning

        # Complexity peak
        peak = detect_complexity_peak(complexity_df)
        if peak:
            moments['complexity_peak'] = peak

    return moments


def detect_first_positive_fitness(fitness_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Detect the first generation with positive fitness.

    Args:
        fitness_df: DataFrame with fitness history

    Returns:
        Dictionary with generation, fitness, and message
    """
    positive = fitness_df[fitness_df['best_fitness'] > 0]

    if len(positive) > 0:
        first = positive.iloc[0]
        return {
            'generation': int(first['generation']),
            'fitness': float(first['best_fitness']),
            'message': 'First positive fitness achieved!'
        }

    return None


def detect_breakthrough(fitness_df: pd.DataFrame, threshold: float = 50) -> List[Dict[str, Any]]:
    """
    Detect breakthrough moments (large jumps in fitness).

    Args:
        fitness_df: DataFrame with fitness history
        threshold: Minimum fitness improvement to count as breakthrough

    Returns:
        List of breakthrough dictionaries
    """
    breakthroughs = []

    for i in range(1, len(fitness_df)):
        prev_fitness = fitness_df.iloc[i-1]['best_fitness']
        curr_fitness = fitness_df.iloc[i]['best_fitness']
        improvement = curr_fitness - prev_fitness

        if improvement >= threshold:
            breakthroughs.append({
                'generation': int(fitness_df.iloc[i]['generation']),
                'fitness_jump': float(improvement),
                'from': float(prev_fitness),
                'to': float(curr_fitness),
                'message': f'Major breakthrough - fitness jumped {improvement:.1f} points!'
            })

    return breakthroughs


def detect_stagnation(fitness_df: pd.DataFrame, window: int = 10, improvement_threshold: float = 5.0) -> List[Dict[str, Any]]:
    """
    Detect periods of stagnation (no significant improvement).

    Args:
        fitness_df: DataFrame with fitness history
        window: Number of generations to check
        improvement_threshold: Minimum improvement to not count as stagnation

    Returns:
        List of stagnation period dictionaries
    """
    stagnation_periods = []
    current_stagnation = None

    for i in range(window, len(fitness_df)):
        window_start = i - window
        start_fitness = fitness_df.iloc[window_start]['best_fitness']
        end_fitness = fitness_df.iloc[i]['best_fitness']
        improvement = end_fitness - start_fitness

        if improvement < improvement_threshold:
            # Stagnating
            if current_stagnation is None:
                current_stagnation = {
                    'start_gen': int(fitness_df.iloc[window_start]['generation']),
                    'end_gen': int(fitness_df.iloc[i]['generation']),
                    'start_fitness': float(start_fitness),
                    'end_fitness': float(end_fitness)
                }
            else:
                # Extend current stagnation
                current_stagnation['end_gen'] = int(fitness_df.iloc[i]['generation'])
                current_stagnation['end_fitness'] = float(end_fitness)
        else:
            # Not stagnating
            if current_stagnation is not None:
                # End of stagnation period
                duration = current_stagnation['end_gen'] - current_stagnation['start_gen']
                current_stagnation['duration'] = duration
                current_stagnation['message'] = f'Stagnated for {duration} generations'
                stagnation_periods.append(current_stagnation)
                current_stagnation = None

    # Handle ongoing stagnation at end
    if current_stagnation is not None:
        duration = current_stagnation['end_gen'] - current_stagnation['start_gen']
        current_stagnation['duration'] = duration
        current_stagnation['message'] = f'Stagnated for {duration} generations'
        stagnation_periods.append(current_stagnation)

    return stagnation_periods


def detect_complexification(complexity_df: pd.DataFrame, threshold: float = 2.0) -> List[Dict[str, Any]]:
    """
    Detect sudden increases in network complexity.

    Args:
        complexity_df: DataFrame with complexity history
        threshold: Minimum node increase to count as complexification

    Returns:
        List of complexification event dictionaries
    """
    events = []

    for i in range(1, len(complexity_df)):
        prev_nodes = complexity_df.iloc[i-1]['avg_nodes']
        curr_nodes = complexity_df.iloc[i]['avg_nodes']
        node_increase = curr_nodes - prev_nodes

        if node_increase >= threshold:
            events.append({
                'generation': int(complexity_df.iloc[i]['generation']),
                'nodes_added': float(node_increase),
                'from_nodes': float(prev_nodes),
                'to_nodes': float(curr_nodes),
                'message': f'Complexification - added {node_increase:.1f} nodes on average'
            })

    return events


def detect_pruning(complexity_df: pd.DataFrame, threshold: float = 2.0, min_duration: int = 5) -> List[Dict[str, Any]]:
    """
    Detect pruning phases (sustained reduction in complexity).

    Args:
        complexity_df: DataFrame with complexity history
        threshold: Minimum node reduction to count as pruning
        min_duration: Minimum number of generations to count as a pruning phase

    Returns:
        List of pruning phase dictionaries
    """
    pruning_phases = []
    current_pruning = None

    for i in range(1, len(complexity_df)):
        prev_nodes = complexity_df.iloc[i-1]['avg_nodes']
        curr_nodes = complexity_df.iloc[i]['avg_nodes']
        node_change = curr_nodes - prev_nodes

        if node_change < -0.5:  # Decreasing
            if current_pruning is None:
                current_pruning = {
                    'start_gen': int(complexity_df.iloc[i-1]['generation']),
                    'end_gen': int(complexity_df.iloc[i]['generation']),
                    'start_nodes': float(prev_nodes),
                    'end_nodes': float(curr_nodes)
                }
            else:
                current_pruning['end_gen'] = int(complexity_df.iloc[i]['generation'])
                current_pruning['end_nodes'] = float(curr_nodes)
        else:
            # Not decreasing
            if current_pruning is not None:
                # Check if significant pruning occurred
                nodes_removed = current_pruning['start_nodes'] - current_pruning['end_nodes']
                duration = current_pruning['end_gen'] - current_pruning['start_gen']

                if nodes_removed >= threshold and duration >= min_duration:
                    current_pruning['nodes_removed'] = float(nodes_removed)
                    current_pruning['duration'] = duration
                    current_pruning['message'] = f'Pruning phase - simplified from {current_pruning["start_nodes"]:.1f} to {current_pruning["end_nodes"]:.1f} nodes over {duration} generations'
                    pruning_phases.append(current_pruning)

                current_pruning = None

    # Handle ongoing pruning at end
    if current_pruning is not None:
        nodes_removed = current_pruning['start_nodes'] - current_pruning['end_nodes']
        duration = current_pruning['end_gen'] - current_pruning['start_gen']

        if nodes_removed >= threshold and duration >= min_duration:
            current_pruning['nodes_removed'] = float(nodes_removed)
            current_pruning['duration'] = duration
            current_pruning['message'] = f'Pruning phase - simplified from {current_pruning["start_nodes"]:.1f} to {current_pruning["end_nodes"]:.1f} nodes over {duration} generations'
            pruning_phases.append(current_pruning)

    return pruning_phases


def detect_complexity_peak(complexity_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Find the generation with maximum network complexity.

    Args:
        complexity_df: DataFrame with complexity history

    Returns:
        Dictionary with peak complexity information
    """
    max_idx = complexity_df['avg_nodes'].idxmax()
    peak_row = complexity_df.iloc[max_idx]

    return {
        'generation': int(peak_row['generation']),
        'nodes': float(peak_row['avg_nodes']),
        'connections': float(peak_row['avg_connections']),
        'message': f'Maximum complexity reached: {peak_row["avg_nodes"]:.1f} nodes, {peak_row["avg_connections"]:.1f} connections'
    }


def identify_evolution_phases(fitness_df: pd.DataFrame, complexity_df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """
    Identify major phases of evolution.

    Phases might include:
    - Exploration (early random behavior)
    - Initial learning (first improvements)
    - Complexification (adding structure)
    - Refinement (optimizing weights)
    - Pruning (simplifying structure)

    Args:
        fitness_df: DataFrame with fitness history
        complexity_df: Optional DataFrame with complexity history

    Returns:
        List of phase dictionaries
    """
    phases = []

    # Phase 1: Exploration (first 10-20% of training)
    exploration_end = min(len(fitness_df) // 5, 20)
    if exploration_end > 0:
        phase1 = fitness_df.iloc[:exploration_end]
        phases.append({
            'name': 'Exploration',
            'start_gen': int(phase1.iloc[0]['generation']),
            'end_gen': int(phase1.iloc[-1]['generation']),
            'fitness_start': float(phase1.iloc[0]['best_fitness']),
            'fitness_end': float(phase1.iloc[-1]['best_fitness']),
            'description': 'Random exploration and initial evolution'
        })

    # Phase 2: Learning (first positive fitness to first breakthrough)
    first_positive_gen = None
    for i, row in fitness_df.iterrows():
        if row['best_fitness'] > 0:
            first_positive_gen = int(row['generation'])
            break

    if first_positive_gen is not None and first_positive_gen < len(fitness_df) - 10:
        # Find first major improvement after positive
        breakthrough_gen = None
        for i in range(first_positive_gen + 1, len(fitness_df)):
            if i > 0:
                improvement = fitness_df.iloc[i]['best_fitness'] - fitness_df.iloc[i-1]['best_fitness']
                if improvement > 50:
                    breakthrough_gen = int(fitness_df.iloc[i]['generation'])
                    break

        if breakthrough_gen:
            learning_phase = fitness_df[(fitness_df['generation'] >= first_positive_gen) &
                                       (fitness_df['generation'] <= breakthrough_gen)]
            if len(learning_phase) > 0:
                phases.append({
                    'name': 'Initial Learning',
                    'start_gen': int(learning_phase.iloc[0]['generation']),
                    'end_gen': int(learning_phase.iloc[-1]['generation']),
                    'fitness_start': float(learning_phase.iloc[0]['best_fitness']),
                    'fitness_end': float(learning_phase.iloc[-1]['best_fitness']),
                    'description': 'First successful behaviors emerging'
                })

    return phases


def calculate_fitness_improvement(fitness_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate various fitness improvement metrics.

    Args:
        fitness_df: DataFrame with fitness history

    Returns:
        Dictionary of improvement metrics
    """
    first_best = fitness_df.iloc[0]['best_fitness']
    last_best = fitness_df.iloc[-1]['best_fitness']
    max_best = fitness_df['best_fitness'].max()

    first_avg = fitness_df.iloc[0]['average_fitness']
    last_avg = fitness_df.iloc[-1]['average_fitness']

    return {
        'total_best_improvement': float(last_best - first_best),
        'total_avg_improvement': float(last_avg - first_avg),
        'max_fitness_achieved': float(max_best),
        'final_best_fitness': float(last_best),
        'final_avg_fitness': float(last_avg),
        'percent_of_target': float(last_best / 270.0 * 100) if 270.0 > 0 else 0.0
    }
