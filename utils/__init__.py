"""
Utility functions for NEAT LunarLander project.

This package provides helper functions for:
- File management and directory organization
- Network visualization with fixed positions
- Training analysis and key moment detection
"""

from .file_manager import (
    create_run_structure,
    get_run_paths,
    find_genomes,
    is_old_structure,
    migrate_to_new_structure
)

from .network_helpers import (
    get_fixed_node_positions,
    identify_node_types,
    format_node_label
)

from .analysis_helpers import (
    detect_key_moments,
    identify_evolution_phases,
    calculate_fitness_improvement,
    detect_breakthrough,
    detect_pruning,
    detect_complexification
)

__all__ = [
    # File management
    'create_run_structure',
    'get_run_paths',
    'find_genomes',
    'is_old_structure',
    'migrate_to_new_structure',

    # Network helpers
    'get_fixed_node_positions',
    'identify_node_types',
    'format_node_label',

    # Analysis helpers
    'detect_key_moments',
    'identify_evolution_phases',
    'calculate_fitness_improvement',
    'detect_breakthrough',
    'detect_pruning',
    'detect_complexification',
]
