#!/usr/bin/env python3
"""
Continue NEAT training from a checkpoint.
Usage: python continue_training.py <run_dir> [additional_generations]
"""

import neat
import gymnasium as gym
import pickle
import os
import sys
from datetime import datetime
import csv
import numpy as np
import time
import json
import glob

# Import our utility functions
from utils.file_manager import create_run_structure


class FitnessLogger(neat.reporting.BaseReporter):
    """
    Enhanced custom reporter that logs detailed statistics for each generation.
    Modified to support continuation from a checkpoint.
    """

    def __init__(self, run_dirs, save_best_every=1, max_generations=5000, start_generation=0):
        self.run_dirs = run_dirs
        self.generation = start_generation
        self.save_best_every = max(1, int(save_best_every))
        self.max_generations = max_generations
        self.start_time = time.time()
        self.gen_start_time = time.time()

        # Previous generation stats for comparison
        self.prev_best_fitness = None
        self.prev_avg_nodes = None

        # CSV file paths in new structure
        self.fitness_file = os.path.join(run_dirs['data'], 'fitness.csv')
        self.complexity_file = os.path.join(run_dirs['data'], 'complexity.csv')
        self.species_file = os.path.join(run_dirs['data'], 'species.csv')
        self.key_moments_file = os.path.join(run_dirs['analysis'], 'key_moments.json')

        # Load existing key moments if available
        if os.path.exists(self.key_moments_file):
            with open(self.key_moments_file, 'r') as f:
                self.key_moments = json.load(f)
        else:
            self.key_moments = {
                'first_positive': None,
                'breakthroughs': [],
                'complexification_events': [],
                'pruning_events': []
            }

        # Load previous best fitness for comparison if available
        if os.path.exists(self.fitness_file):
            with open(self.fitness_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Skip header
                    last_line = lines[-1].strip().split(',')
                    self.prev_best_fitness = float(last_line[1])

            # Read last complexity for comparison
            if os.path.exists(self.complexity_file):
                with open(self.complexity_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split(',')
                        self.prev_avg_nodes = float(last_line[1])

        # Note: We append to existing CSVs, don't create new headers

    def post_evaluate(self, config, population, species_set, best_genome):
        """Called after each generation's evaluation with enhanced analysis."""

        gen_time = time.time() - self.gen_start_time

        # Collect fitness statistics
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]

        if fitnesses:
            best_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            worst_fitness = min(fitnesses)
            std_fitness = (sum((f - avg_fitness)**2 for f in fitnesses) / len(fitnesses))**0.5
        else:
            best_fitness = avg_fitness = worst_fitness = std_fitness = 0.0

        num_species = len(species_set.species)

        # Collect complexity statistics
        nodes = [len(g.nodes) for g in population.values()]
        connections = [len(g.connections) for g in population.values()]

        avg_nodes = sum(nodes) / len(nodes) if nodes else 0
        avg_connections = sum(connections) / len(connections) if connections else 0
        max_nodes = max(nodes) if nodes else 0
        max_connections = max(connections) if connections else 0

        # Write to CSV files (append mode)
        with open(self.fitness_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.generation, best_fitness, avg_fitness,
                           worst_fitness, std_fitness, num_species])

        with open(self.complexity_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.generation, avg_nodes, avg_connections,
                           max_nodes, max_connections])

        # Write species information
        with open(self.species_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for species_id, species in species_set.species.items():
                species_fitnesses = [m.fitness for m in species.members.values() if m.fitness is not None]
                species_best = max(species_fitnesses) if species_fitnesses else 0.0
                species_avg = sum(species_fitnesses) / len(species_fitnesses) if species_fitnesses else 0.0

                writer.writerow([
                    self.generation,
                    species_id,
                    len(species.members),
                    self.generation - species.created,  # Age
                    species_best,
                    species_avg,
                    species.last_improved if hasattr(species, 'last_improved') else 0
                ])

        # Detect key moments
        self._detect_key_moments(best_fitness, avg_nodes)

        # Save the best genome for this generation (throttled to reduce disk use)
        saved_best = False
        if self.generation % self.save_best_every == 0:
            genome_file = os.path.join(self.run_dirs['genomes'], f'g{self.generation:03d}.pkl')
            with open(genome_file, 'wb') as f:
                pickle.dump(best_genome, f)
            saved_best = True

        # Print BEGINNER-FRIENDLY educational output
        print(f"\n{'='*70}")
        print(f"GENERATION {self.generation} / {self.max_generations}")
        print(f"{'='*70}")

        # Fitness with clear context
        fitness_str = f"  Best Fitness:    {best_fitness:7.1f}"
        if self.prev_best_fitness is not None:
            improvement = best_fitness - self.prev_best_fitness
            if improvement > 50:
                fitness_str += f"  (+{improvement:.1f} HUGE LEAP!)"
            elif improvement > 10:
                fitness_str += f"  (+{improvement:.1f} improving)"
            elif improvement < -10:
                fitness_str += f"  ({improvement:.1f} worse)"
            else:
                fitness_str += f"  (stable)"
        print(fitness_str)

        print(f"  Average Fitness: {avg_fitness:7.1f}")
        print(f"  Worst Fitness:   {worst_fitness:7.1f}")
        print()

        # Species info with CLEAR explanations
        print(f"  Species:         {num_species}")
        if num_species == 1:
            print("    -> Only 1 species (all networks similar)")
        elif num_species <= 3:
            print("    -> Low diversity (population converging)")
        elif num_species <= 5:
            print("    -> Moderate diversity (healthy exploration)")
        else:
            print("    -> High diversity (trying many strategies)")
        print()

        # Network complexity with SIMPLE explanation
        print(f"  Network Size:    {avg_nodes:.1f} nodes, {avg_connections:.1f} connections")
        print(f"    -> Largest:    {max_nodes} nodes, {max_connections} connections")

        if self.prev_avg_nodes is not None:
            node_change = avg_nodes - self.prev_avg_nodes
            if node_change > 1.0:
                print(f"    -> Growing (+{node_change:.1f} nodes - adding complexity)")
            elif node_change < -1.0:
                print(f"    -> Shrinking ({node_change:.1f} nodes - simplifying)")
            else:
                print(f"    -> Stable size (fine-tuning weights)")
        print()

        # VISUAL progress bar
        percent_of_target = (best_fitness / config.fitness_threshold * 100) if config.fitness_threshold > 0 else 0
        bar_length = 40
        filled = int(bar_length * min(percent_of_target, 100) / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"  Progress: [{bar}] {percent_of_target:.1f}%")
        print(f"  Target:   {config.fitness_threshold:.0f} (LunarLander solved at 200+)")

        if best_fitness >= config.fitness_threshold:
            print(f"\n  *** SOLVED! Target reached! ***")
        elif best_fitness >= 200:
            print(f"  (Already good enough to land! Going for higher score...)")
        elif best_fitness >= 0:
            print(f"  (Learning to land - getting positive scores!)")
        else:
            print(f"  (Still crashing - learning basics...)")
        print()

        # Time info
        elapsed = time.time() - self.start_time
        if self.generation > 0:
            avg_gen_time = elapsed / (self.generation + 1)
            remaining_gens = self.max_generations - (self.generation + 1)
            est_remaining = avg_gen_time * remaining_gens
            hours = int(est_remaining // 3600)
            minutes = int((est_remaining % 3600) // 60)
            print(f"  Time/gen: {gen_time:.1f}s | Est. remaining: {hours}h {minutes}m")

        # Save info
        if saved_best:
            print(f"  Saved: g{self.generation:03d}.pkl")

        print(f"{'='*70}\n")

        # Update previous values
        self.prev_best_fitness = best_fitness
        self.prev_avg_nodes = avg_nodes
        self.gen_start_time = time.time()

        self.generation += 1

    def _detect_key_moments(self, best_fitness, avg_nodes):
        """Detect and record key moments during training."""

        # First positive fitness
        if self.key_moments['first_positive'] is None and best_fitness > 0:
            self.key_moments['first_positive'] = {
                'generation': self.generation,
                'fitness': float(best_fitness)
            }

        # Breakthroughs
        if self.prev_best_fitness is not None:
            improvement = best_fitness - self.prev_best_fitness
            if improvement > 50:
                self.key_moments['breakthroughs'].append({
                    'generation': self.generation,
                    'improvement': float(improvement),
                    'from': float(self.prev_best_fitness),
                    'to': float(best_fitness)
                })

        # Complexification
        if self.prev_avg_nodes is not None:
            node_change = avg_nodes - self.prev_avg_nodes
            if node_change > 2.0:
                self.key_moments['complexification_events'].append({
                    'generation': self.generation,
                    'nodes_added': float(node_change),
                    'from': float(self.prev_avg_nodes),
                    'to': float(avg_nodes)
                })
            elif node_change < -2.0:
                self.key_moments['pruning_events'].append({
                    'generation': self.generation,
                    'nodes_removed': float(-node_change),
                    'from': float(self.prev_avg_nodes),
                    'to': float(avg_nodes)
                })

    def save_key_moments(self):
        """Save detected key moments to JSON file."""
        with open(self.key_moments_file, 'w') as f:
            json.dump(self.key_moments, f, indent=2)


EVAL_EPISODES = 8


def eval_genome(genome, config, seeds=None):
    """
    Evaluate a single genome on LunarLander-v2 by averaging multiple episodes.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make('LunarLander-v2')

    num_episodes = EVAL_EPISODES
    total_fitness = 0.0
    seeds = list(seeds) if seeds is not None else [None] * num_episodes
    if len(seeds) < num_episodes:
        seeds.extend([None] * (num_episodes - len(seeds)))

    for ep_idx in range(num_episodes):
        seed = seeds[ep_idx]
        if seed is not None:
            obs, _ = env.reset(seed=int(seed))
            env.action_space.seed(int(seed))
        else:
            obs, _ = env.reset()

        fitness = 0.0
        done = False

        while not done:
            output = net.activate(obs)
            action = int(np.argmax(output))
            obs, reward, terminated, truncated, _ = env.step(action)
            fitness += reward
            done = terminated or truncated

        total_fitness += fitness

    env.close()
    return total_fitness / num_episodes


def eval_genomes(genomes, config):
    """Evaluate all genomes in the population."""
    generation = getattr(eval_genomes, "generation", 0)
    rng = np.random.default_rng(1234 + generation)
    seeds = rng.integers(0, 2**31 - 1, size=EVAL_EPISODES)

    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, seeds=seeds)

    eval_genomes.generation = generation + 1


def find_latest_checkpoint(run_dir):
    """Find the latest checkpoint file in the run directory."""
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint-*'))
    if not checkpoints:
        return None

    # Extract generation numbers and find the max
    checkpoint_nums = []
    for cp in checkpoints:
        basename = os.path.basename(cp)
        if basename.startswith('checkpoint-'):
            try:
                num = int(basename.split('-')[1])
                checkpoint_nums.append((num, cp))
            except:
                continue

    if not checkpoint_nums:
        return None

    checkpoint_nums.sort(key=lambda x: x[0], reverse=True)
    return checkpoint_nums[0]  # Returns (generation_num, filepath)


def continue_training(run_dir, additional_generations=1000, save_best_every=5, config_file=None):
    """
    Continue training from the latest checkpoint in the specified run directory.

    Args:
        run_dir: Path to existing run directory
        additional_generations: Number of additional generations to train
        save_best_every: Save best genome every N generations
        config_file: Optional path to config file (if None, tries run_dir/config.ini, then neat-config.ini)
    """

    if not os.path.exists(run_dir):
        print(f"❌ Error: Run directory '{run_dir}' not found!")
        sys.exit(1)

    # Find latest checkpoint
    checkpoint_info = find_latest_checkpoint(run_dir)
    if checkpoint_info is None:
        print(f"❌ Error: No checkpoints found in '{run_dir}/checkpoints/'")
        sys.exit(1)

    checkpoint_gen, checkpoint_path = checkpoint_info
    print(f"\n✓ Found checkpoint at generation {checkpoint_gen}")
    print(f"  Path: {checkpoint_path}")

    # Load config - try multiple locations
    if config_file is not None:
        config_path = config_file
        print(f"✓ Using custom config: {config_path}")
    else:
        # Try run_dir/config.ini first
        config_path = os.path.join(run_dir, 'config.ini')
        if not os.path.exists(config_path):
            # Fall back to neat-config.ini in current directory
            config_path = 'neat-config.ini'
            print(f"⚠ Config not found in run directory, using: {config_path}")
        else:
            print(f"✓ Using original config: {config_path}")

    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found at '{config_path}'")
        print(f"   Please specify a config file or ensure neat-config.ini exists")
        sys.exit(1)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Get directory structure (already exists)
    run_dirs = {
        'root': run_dir,
        'data': os.path.join(run_dir, 'data'),
        'genomes': os.path.join(run_dir, 'genomes'),
        'checkpoints': os.path.join(run_dir, 'checkpoints'),
        'analysis': os.path.join(run_dir, 'analysis')
    }

    # Calculate total generations
    total_generations = checkpoint_gen + additional_generations

    print("\n" + "="*70)
    print("CONTINUING NEAT LUNARLANDER TRAINING")
    print("="*70)
    print(f"\nResuming from:")
    print(f"  Run directory:     {run_dir}")
    print(f"  Checkpoint:        Generation {checkpoint_gen}")
    print(f"  Additional gens:   {additional_generations}")
    print(f"  New target:        Generation {total_generations}")
    print(f"  Population:        {config.pop_size} networks per generation")
    print(f"  Target fitness:    {config.fitness_threshold:.0f}")
    print("="*70 + "\n")

    # Restore population from checkpoint
    population = neat.Checkpointer.restore_checkpoint(checkpoint_path)

    # Set up eval_genomes generation counter to match checkpoint
    eval_genomes.generation = checkpoint_gen + 1

    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))

    # Add fitness logger (starting from checkpoint generation)
    fitness_logger = FitnessLogger(
        run_dirs,
        save_best_every=save_best_every,
        max_generations=total_generations,
        start_generation=checkpoint_gen + 1
    )
    population.add_reporter(fitness_logger)

    # Add statistics reporter
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Add checkpointer (continues saving)
    checkpointer = neat.Checkpointer(
        generation_interval=10,
        filename_prefix=os.path.join(run_dirs['checkpoints'], 'checkpoint-')
    )
    population.add_reporter(checkpointer)

    # Continue training
    print("\n🧬 Resuming evolution...\n")
    start_time = time.time()
    winner = population.run(eval_genomes, additional_generations)
    total_time = time.time() - start_time

    # Save key moments
    fitness_logger.save_key_moments()

    # Save final results
    print("\n" + "="*70)
    print("CONTINUED TRAINING COMPLETE!")
    print("="*70)
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"\nSession time: {hours}h {minutes}m {seconds}s")
    print(f"\nFinal Results:")
    print(f"  Best fitness:        {winner.fitness:.1f}")
    print(f"  Network complexity:  {len(winner.nodes)} nodes, {len(winner.connections)} connections")
    print(f"  Final generation:    {checkpoint_gen + additional_generations}")

    # Check if target reached
    target_fitness = config.fitness_threshold
    if winner.fitness >= target_fitness:
        print(f"\n  *** SUCCESS! Target of {target_fitness:.0f} reached! ***")
    elif winner.fitness >= 200:
        print(f"\n  Great! Fitness of {winner.fitness:.1f} is above 200 (successful landing)")
    else:
        percent = (winner.fitness / target_fitness * 100) if target_fitness > 0 else 0
        print(f"\n  Reached {percent:.1f}% of target ({target_fitness:.0f})")

    # Update winner file
    winner_file = os.path.join(run_dirs['genomes'], 'winner.pkl')
    with open(winner_file, 'wb') as f:
        pickle.dump(winner, f)
    print(f"\nUpdated files:")
    print(f"  - Winner network: genomes/winner.pkl")

    # Save updated statistics
    stats_file = os.path.join(run_dirs['data'], 'stats.pkl')
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)
    print(f"  - Training stats: data/stats.pkl")

    print("\n" + "="*70)
    print("What to do next:")
    print("="*70)
    print(f"\n1. Watch the best network play:")
    print(f"   python demo_best.py {run_dir}")
    print(f"\n2. Generate visualizations:")
    print(f"   python visualize_all.py {run_dir}")
    print("="*70 + "\n")

    return winner, run_dir


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python continue_training.py <run_dir> [additional_generations] [config_file]")
        print("\nExamples:")
        print("  python continue_training.py runs/run_20251130_163831")
        print("  python continue_training.py runs/run_20251130_163831 500")
        print("  python continue_training.py runs/run_20251130_163831 1000 neat-config.ini")
        print("  python continue_training.py runs/run_20251130_163831 1000 custom-config.ini")
        print("\nConfig priority:")
        print("  1. Specified config file (3rd argument)")
        print("  2. run_dir/config.ini (if exists)")
        print("  3. neat-config.ini (fallback)")
        sys.exit(1)

    run_dir = sys.argv[1]
    additional_gens = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    config_file = sys.argv[3] if len(sys.argv) > 3 else None

    winner, run_dir = continue_training(run_dir, additional_generations=additional_gens, config_file=config_file)
