#!/usr/bin/env python3
"""
NEAT LunarLander Training Script
================================
Train a NEAT (NeuroEvolution of Augmenting Topologies) agent to solve
the LunarLander-v2 environment with an emphasis on pedagogy and visualization.
"""

import neat
import gymnasium as gym
import pickle
import os
from datetime import datetime
import csv
import numpy as np
import time
import json

# Import our utility functions
from utils.file_manager import create_run_structure


class FitnessLogger(neat.reporting.BaseReporter):
    """
    Enhanced custom reporter that logs detailed statistics for each generation.

    Features:
    - Tracks fitness, complexity, and species information
    - Detects key moments (breakthroughs, first positive, etc.)
    - Provides contextual console output
    - Saves to organized directory structure
    """

    def __init__(self, run_dirs, save_best_every=1, max_generations=5000):
        self.run_dirs = run_dirs
        self.generation = 0
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

        # Track key moments
        self.key_moments = {
            'first_positive': None,
            'breakthroughs': [],
            'complexification_events': [],
            'pruning_events': []
        }

        # Initialize CSV files with headers
        with open(self.fitness_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['generation', 'best_fitness', 'average_fitness',
                           'worst_fitness', 'std_fitness', 'num_species'])

        with open(self.complexity_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['generation', 'avg_nodes', 'avg_connections',
                           'max_nodes', 'max_connections'])

        with open(self.species_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['generation', 'species_id', 'size', 'age',
                           'best_fitness', 'avg_fitness', 'stagnation'])

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

        # Write to CSV files
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

    Args:
        genome: A NEAT genome to evaluate
        config: NEAT configuration object
        seeds: Optional list/array of seeds (one per episode) for determinism

    Returns:
        float: Average fitness score across episodes
    """
    # Create a neural network from the genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Create the LunarLander environment
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

        # Run one episode
        while not done:
            # Network outputs 4 values (one per discrete action)
            output = net.activate(obs)
            action = int(np.argmax(output))

            # Take action in environment
            obs, reward, terminated, truncated, _ = env.step(action)

            fitness += reward
            done = terminated or truncated

        total_fitness += fitness

    env.close()
    return total_fitness / num_episodes


def eval_genomes(genomes, config):
    """
    Evaluate all genomes in the population.
    This is called by NEAT for each generation.
    """
    generation = getattr(eval_genomes, "generation", 0)
    rng = np.random.default_rng(1234 + generation)
    seeds = rng.integers(0, 2**31 - 1, size=EVAL_EPISODES)

    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, seeds=seeds)

    eval_genomes.generation = generation + 1


def run_neat(config_file, max_generations=5000, save_best_every=10):
    """
    Main training function with enhanced directory structure and logging.

    Args:
        config_file: Path to NEAT configuration file
        max_generations: Maximum number of generations to run
        save_best_every: Save best genome snapshot every N generations
    """

    # Load NEAT configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    target_fitness = config.fitness_threshold

    # Create run directory with timestamp and new structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join('runs', f'run_{timestamp}')

    # Create organized directory structure
    run_dirs = create_run_structure(run_dir)

    print("\n" + "="*70)
    print("NEAT LUNARLANDER TRAINING - Beginner-Friendly Mode")
    print("="*70)
    print(f"\nWhat is this?")
    print(f"  This program teaches neural networks to land a lunar lander")
    print(f"  using NEAT - a genetic algorithm that evolves both the")
    print(f"  network structure AND the connection weights!")
    print(f"\nTraining Setup:")
    print(f"  Run directory:  {run_dir}")
    print(f"  Population:     {config.pop_size} networks per generation")
    print(f"  Target score:   {target_fitness:.0f} (200+ = successful landing)")
    print(f"  Max iterations: {max_generations} generations")
    print(f"\nWhat to watch for:")
    print(f"  - Networks start simple (just inputs -> outputs)")
    print(f"  - They gradually add nodes and connections")
    print(f"  - Fitness improves as they learn to land")
    print(f"  - Training saves snapshots every {save_best_every} generations")
    print("="*70 + "\n")

    # Create population
    population = neat.Population(config)

    # Add reporters for monitoring
    population.add_reporter(neat.StdOutReporter(True))

    # Add our enhanced fitness logger
    fitness_logger = FitnessLogger(run_dirs, save_best_every=save_best_every, max_generations=max_generations)
    population.add_reporter(fitness_logger)

    # Add statistics reporter
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Add checkpointer (saves every 10 generations to checkpoints subdirectory)
    checkpointer = neat.Checkpointer(
        generation_interval=10,
        filename_prefix=os.path.join(run_dirs['checkpoints'], 'checkpoint-')
    )
    population.add_reporter(checkpointer)

    # Run NEAT
    print("\n🧬 Starting evolution...\n")
    start_time = time.time()
    winner = population.run(eval_genomes, max_generations)
    total_time = time.time() - start_time

    # Save key moments
    fitness_logger.save_key_moments()

    # Save final results
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"\nTotal time: {hours}h {minutes}m {seconds}s")
    print(f"\nFinal Results:")
    print(f"  Best fitness:        {winner.fitness:.1f}")
    print(f"  Network complexity:  {len(winner.nodes)} nodes, {len(winner.connections)} connections")

    # Check if target reached with context
    if winner.fitness >= target_fitness:
        print(f"\n  *** SUCCESS! Target of {target_fitness:.0f} reached! ***")
        print(f"  The network learned to land the lunar lander successfully!")
    elif winner.fitness >= 200:
        print(f"\n  Great! Fitness of {winner.fitness:.1f} is above 200 (successful landing)")
        print(f"  Target was {target_fitness:.0f}, which is just a higher score goal")
    else:
        percent = (winner.fitness / target_fitness * 100) if target_fitness > 0 else 0
        print(f"\n  Reached {percent:.1f}% of target ({target_fitness:.0f})")
        print(f"  Consider training longer or adjusting config parameters")

    # Save winner
    winner_file = os.path.join(run_dirs['genomes'], 'winner.pkl')
    with open(winner_file, 'wb') as f:
        pickle.dump(winner, f)
    print(f"\nSaved files:")
    print(f"  - Winner network: genomes/winner.pkl")

    # Save statistics
    stats_file = os.path.join(run_dirs['data'], 'stats.pkl')
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)
    print(f"  - Training stats: data/stats.pkl")

    # Save config for later use
    config_copy = os.path.join(run_dir, 'config.ini')
    import shutil
    shutil.copy(config_file, config_copy)
    print(f"  - Config copy: config.ini")

    print("\n" + "="*70)
    print("What to do next:")
    print("="*70)
    print(f"\n1. Watch the best network play:")
    print(f"   python demo_best.py {run_dir}")
    print(f"\n2. Generate ALL visualizations at once:")
    print(f"   python visualize_all.py {run_dir}")
    print(f"\n3. Or create individual visualizations:")
    print(f"   python visualize_network_v2.py {run_dir}       # Network diagrams")
    print(f"   python plot_fitness.py {run_dir}               # Fitness graphs")
    print(f"   python plot_complexity_fitness.py {run_dir}    # Complexity analysis")
    print(f"   python visualize_evolution.py {run_dir}        # Evolution animation")
    print("="*70 + "\n")

    return winner, run_dir


if __name__ == '__main__':
    # Run with default settings
    config_path = 'neat-config.ini'

    if not os.path.exists(config_path):
        print(f"❌ Error: Config file '{config_path}' not found!")
        exit(1)

    winner, run_dir = run_neat(config_path, max_generations=5000, save_best_every=5)
