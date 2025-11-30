"""
File management utilities for organizing NEAT training runs.

Handles directory structure creation, path management, and migration
from old to new directory structures.
"""

import os
import shutil
from glob import glob
from typing import Dict, List, Optional


def create_run_structure(run_dir: str) -> Dict[str, str]:
    """
    Create the organized directory structure for a training run.

    Args:
        run_dir: Base directory for the run

    Returns:
        Dictionary mapping directory names to their paths
    """
    structure = {
        'base': run_dir,
        'genomes': os.path.join(run_dir, 'genomes'),
        'data': os.path.join(run_dir, 'data'),
        'visualizations': os.path.join(run_dir, 'visualizations'),
        'networks': os.path.join(run_dir, 'visualizations', 'networks'),
        'comparisons': os.path.join(run_dir, 'visualizations', 'comparisons'),
        'diffs': os.path.join(run_dir, 'visualizations', 'diffs'),
        'plots': os.path.join(run_dir, 'visualizations', 'plots'),
        'checkpoints': os.path.join(run_dir, 'checkpoints'),
        'analysis': os.path.join(run_dir, 'analysis'),
    }

    # Create all directories
    for dir_path in structure.values():
        os.makedirs(dir_path, exist_ok=True)

    return structure


def get_run_paths(run_dir: str) -> Dict[str, str]:
    """
    Get paths for a run directory, supporting both old and new structures.

    Args:
        run_dir: Base directory for the run

    Returns:
        Dictionary with paths to important files and directories
    """
    if is_old_structure(run_dir):
        # Old flat structure
        return {
            'base': run_dir,
            'genomes_dir': run_dir,
            'data_dir': run_dir,
            'networks_dir': os.path.join(run_dir, 'networks'),
            'checkpoints_dir': run_dir,
            'fitness_csv': os.path.join(run_dir, 'fitness_history.csv'),
            'complexity_csv': os.path.join(run_dir, 'complexity_history.csv'),
            'species_csv': None,  # Doesn't exist in old structure
            'config': os.path.join(run_dir, 'config.ini'),
            'winner': os.path.join(run_dir, 'winner.pkl'),
            'stats': os.path.join(run_dir, 'stats.pkl'),
        }
    else:
        # New organized structure
        return {
            'base': run_dir,
            'genomes_dir': os.path.join(run_dir, 'genomes'),
            'data_dir': os.path.join(run_dir, 'data'),
            'networks_dir': os.path.join(run_dir, 'visualizations', 'networks'),
            'comparisons_dir': os.path.join(run_dir, 'visualizations', 'comparisons'),
            'diffs_dir': os.path.join(run_dir, 'visualizations', 'diffs'),
            'plots_dir': os.path.join(run_dir, 'visualizations', 'plots'),
            'checkpoints_dir': os.path.join(run_dir, 'checkpoints'),
            'analysis_dir': os.path.join(run_dir, 'analysis'),
            'fitness_csv': os.path.join(run_dir, 'data', 'fitness.csv'),
            'complexity_csv': os.path.join(run_dir, 'data', 'complexity.csv'),
            'species_csv': os.path.join(run_dir, 'data', 'species.csv'),
            'config': os.path.join(run_dir, 'config.ini'),
            'winner': os.path.join(run_dir, 'genomes', 'winner.pkl'),
            'stats': os.path.join(run_dir, 'data', 'stats.pkl'),
        }


def is_old_structure(run_dir: str) -> bool:
    """
    Check if a run directory uses the old flat structure.

    Args:
        run_dir: Base directory for the run

    Returns:
        True if old structure, False if new structure
    """
    # Check for presence of new subdirectories
    new_markers = [
        os.path.join(run_dir, 'genomes'),
        os.path.join(run_dir, 'data'),
        os.path.join(run_dir, 'visualizations'),
    ]

    # If any new subdirectory exists, assume new structure
    for marker in new_markers:
        if os.path.exists(marker):
            return False

    # Otherwise, assume old structure
    return True


def find_genomes(run_dir: str) -> List[tuple]:
    """
    Find all saved genome files in a run directory.

    Args:
        run_dir: Base directory for the run

    Returns:
        List of (generation_number, file_path) tuples, sorted by generation
    """
    paths = get_run_paths(run_dir)
    genomes_dir = paths['genomes_dir']

    # Find genome files
    if is_old_structure(run_dir):
        pattern = os.path.join(genomes_dir, 'best_genome_g*.pkl')
    else:
        pattern = os.path.join(genomes_dir, 'g*.pkl')

    genome_files = glob(pattern)

    # Extract generation numbers
    genomes = []
    for file_path in genome_files:
        filename = os.path.basename(file_path)

        # Extract generation number from filename
        if 'best_genome_g' in filename:
            gen_str = filename.split('best_genome_g')[1].split('.pkl')[0]
        elif filename.startswith('g') and filename.endswith('.pkl'):
            gen_str = filename[1:].split('.pkl')[0]
        else:
            continue

        try:
            gen_num = int(gen_str)
            genomes.append((gen_num, file_path))
        except ValueError:
            continue

    # Sort by generation number
    genomes.sort(key=lambda x: x[0])

    return genomes


def migrate_to_new_structure(run_dir: str, dry_run: bool = False) -> bool:
    """
    Migrate an old-structure run directory to the new organized structure.

    Args:
        run_dir: Base directory for the run
        dry_run: If True, only print what would be done without actually moving files

    Returns:
        True if migration successful, False otherwise
    """
    if not is_old_structure(run_dir):
        print(f"Directory {run_dir} already uses new structure.")
        return True

    print(f"\n{'='*70}")
    print(f"{'DRY RUN: ' if dry_run else ''}Migrating {run_dir} to new structure")
    print(f"{'='*70}\n")

    # Create new structure
    if not dry_run:
        structure = create_run_structure(run_dir)
    else:
        structure = {
            'genomes': os.path.join(run_dir, 'genomes'),
            'data': os.path.join(run_dir, 'data'),
            'networks': os.path.join(run_dir, 'visualizations', 'networks'),
            'checkpoints': os.path.join(run_dir, 'checkpoints'),
        }

    # Define file movements
    movements = []

    # Move genome files
    genome_files = glob(os.path.join(run_dir, 'best_genome_g*.pkl'))
    for old_path in genome_files:
        filename = os.path.basename(old_path)
        # Rename from best_genome_g000.pkl to g000.pkl
        new_filename = filename.replace('best_genome_', '')
        new_path = os.path.join(structure['genomes'], new_filename)
        movements.append((old_path, new_path, 'genome'))

    # Move winner
    winner_old = os.path.join(run_dir, 'winner.pkl')
    if os.path.exists(winner_old):
        winner_new = os.path.join(structure['genomes'], 'winner.pkl')
        movements.append((winner_old, winner_new, 'winner'))

    # Move CSV files
    csv_files = [
        ('fitness_history.csv', 'fitness.csv'),
        ('complexity_history.csv', 'complexity.csv'),
    ]
    for old_name, new_name in csv_files:
        old_path = os.path.join(run_dir, old_name)
        if os.path.exists(old_path):
            new_path = os.path.join(structure['data'], new_name)
            movements.append((old_path, new_path, 'data'))

    # Move stats
    stats_old = os.path.join(run_dir, 'stats.pkl')
    if os.path.exists(stats_old):
        stats_new = os.path.join(structure['data'], 'stats.pkl')
        movements.append((stats_old, stats_new, 'stats'))

    # Move network visualizations
    networks_old_dir = os.path.join(run_dir, 'networks')
    if os.path.exists(networks_old_dir):
        network_files = glob(os.path.join(networks_old_dir, '*.png'))
        for old_path in network_files:
            filename = os.path.basename(old_path)
            new_path = os.path.join(structure['networks'], filename)
            movements.append((old_path, new_path, 'network'))

    # Move plots
    plot_files = [
        'training_plots.png',
        'fitness_plot.png',
        'neat_evolution.gif',
        'network_comparison.png',
    ]
    for filename in plot_files:
        old_path = os.path.join(run_dir, filename)
        if os.path.exists(old_path):
            # Determine destination
            if filename.endswith('.gif'):
                new_path = os.path.join(run_dir, 'visualizations', filename)
            else:
                new_path = os.path.join(run_dir, 'visualizations', 'plots', filename)
            movements.append((old_path, new_path, 'visualization'))

    # Move checkpoints
    checkpoint_files = glob(os.path.join(run_dir, 'checkpoint-*'))
    for old_path in checkpoint_files:
        filename = os.path.basename(old_path)
        new_path = os.path.join(structure['checkpoints'], filename)
        movements.append((old_path, new_path, 'checkpoint'))

    # Print and execute movements
    for old_path, new_path, file_type in movements:
        print(f"  {file_type:12s}: {os.path.basename(old_path):30s} -> {os.path.relpath(new_path, run_dir)}")

        if not dry_run:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(new_path), exist_ok=True)

            # Move file
            try:
                shutil.move(old_path, new_path)
            except Exception as e:
                print(f"    ERROR: {e}")
                return False

    # Remove old networks directory if empty
    if not dry_run and os.path.exists(networks_old_dir):
        try:
            if not os.listdir(networks_old_dir):
                os.rmdir(networks_old_dir)
                print(f"\n  Removed empty directory: networks/")
        except Exception:
            pass

    print(f"\n{'='*70}")
    print(f"{'DRY RUN COMPLETE' if dry_run else 'Migration complete!'}")
    print(f"{'='*70}\n")

    return True


def get_latest_run(runs_parent_dir: str = 'runs') -> Optional[str]:
    """
    Get the path to the most recent run directory.

    Args:
        runs_parent_dir: Parent directory containing run subdirectories

    Returns:
        Path to latest run directory, or None if no runs found
    """
    run_pattern = os.path.join(runs_parent_dir, 'run_*')
    run_dirs = sorted(glob(run_pattern), reverse=True)

    if run_dirs:
        return run_dirs[0]
    return None
