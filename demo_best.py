#!/usr/bin/env python3
"""
Demo Script - Watch the Best Agent Play
========================================
Loads the best trained genome and watches it play LunarLander.
Great for seeing the final result in action!
"""

import neat
import gymnasium as gym
import pickle
import os
import sys
import time
import numpy as np
from utils.file_manager import get_run_paths


def run_winner(genome, config, render=True, episodes=5, delay=0.01):
    """
    Run the winning genome on LunarLander and optionally render it.

    Args:
        genome: The NEAT genome to evaluate
        config: NEAT configuration
        render: Whether to render the environment
        episodes: Number of episodes to run
        delay: Delay between frames (for visualization)
    """

    # Create neural network from genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Create environment with rendering if requested
    if render:
        env = gym.make('LunarLander-v2', render_mode='human')
    else:
        env = gym.make('LunarLander-v2')

    print("\n" + "="*70)
    print("🎮 DEMO: WATCHING THE BEST AGENT")
    print("="*70)
    print(f"🧠 Network has {len(genome.nodes)} nodes and {len(genome.connections)} connections")
    print(f"🎯 Running {episodes} episodes...")
    print("="*70 + "\n")

    episode_rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        print(f"Episode {episode + 1}/{episodes}:")

        while not done:
            # Get action from network
            output = net.activate(obs)
            action = int(np.argmax(output))

            # Take step
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            # Add delay for better visualization
            if render and delay > 0:
                time.sleep(delay)

        episode_rewards.append(total_reward)
        status = "✅ SOLVED" if total_reward >= 200 else "  "
        print(f"  {status} Reward: {total_reward:6.1f} | Steps: {steps:3d}")

    env.close()

    # Print statistics
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    max_reward = max(episode_rewards)
    min_reward = min(episode_rewards)

    print("\n" + "="*70)
    print("📊 PERFORMANCE SUMMARY")
    print("="*70)
    print(f"  Average reward: {avg_reward:6.1f}")
    print(f"  Best reward:    {max_reward:6.1f}")
    print(f"  Worst reward:   {min_reward:6.1f}")
    print(f"  Success rate:   {sum(1 for r in episode_rewards if r >= 200)}/{episodes}")
    print("="*70 + "\n")

    return episode_rewards


def analyze_network(genome, config):
    """
    Analyze and print information about the network structure.

    Args:
        genome: The NEAT genome to analyze
        config: NEAT configuration
    """

    print("\n" + "="*70)
    print("🔍 NETWORK ANALYSIS")
    print("="*70)

    # Count nodes by type
    input_nodes = set(range(-config.genome_config.num_inputs, 0))
    output_nodes = set(range(config.genome_config.num_outputs))
    hidden_nodes = set()

    for node_id in genome.nodes.keys():
        if node_id not in input_nodes and node_id not in output_nodes:
            hidden_nodes.add(node_id)

    print(f"  Input nodes:   {len(input_nodes)}")
    print(f"  Hidden nodes:  {len(hidden_nodes)}")
    print(f"  Output nodes:  {len(output_nodes)}")
    print(f"  Total nodes:   {len(genome.nodes)}")
    print()

    # Analyze connections
    enabled_connections = [c for c in genome.connections.values() if c.enabled]
    disabled_connections = [c for c in genome.connections.values() if not c.enabled]

    print(f"  Enabled connections:  {len(enabled_connections)}")
    print(f"  Disabled connections: {len(disabled_connections)}")
    print(f"  Total connections:    {len(genome.connections)}")
    print()

    # Weight statistics
    if enabled_connections:
        weights = [c.weight for c in enabled_connections]
        avg_weight = sum(weights) / len(weights)
        max_weight = max(weights)
        min_weight = min(weights)

        print(f"  Weight statistics:")
        print(f"    Average: {avg_weight:7.3f}")
        print(f"    Max:     {max_weight:7.3f}")
        print(f"    Min:     {min_weight:7.3f}")
        print()

    # Activation functions used
    activations = {}
    for node_id, node in genome.nodes.items():
        if node_id not in input_nodes:  # Input nodes don't have activation
            act = node.activation
            activations[act] = activations.get(act, 0) + 1

    if activations:
        print(f"  Activation functions:")
        for act, count in sorted(activations.items()):
            print(f"    {act:8s}: {count} nodes")

    print("="*70 + "\n")


def demo_interactive(genome, config):
    """
    Interactive demo where user can trigger episodes.
    """

    print("\n" + "="*70)
    print("🎮 INTERACTIVE DEMO MODE")
    print("="*70)
    print("Commands:")
    print("  <Enter>  - Run one episode")
    print("  5        - Run 5 episodes")
    print("  10       - Run 10 episodes")
    print("  q        - Quit")
    print("="*70 + "\n")

    while True:
        try:
            user_input = input("➡️  Command: ").strip().lower()

            if user_input == 'q' or user_input == 'quit':
                print("👋 Goodbye!\n")
                break
            elif user_input == '' or user_input == '1':
                run_winner(genome, config, render=True, episodes=1)
            elif user_input.isdigit():
                episodes = int(user_input)
                if episodes > 0 and episodes <= 100:
                    run_winner(genome, config, render=True, episodes=episodes)
                else:
                    print("❌ Please enter a number between 1 and 100")
            else:
                print("❌ Invalid command")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\n❌ Usage: python demo_best.py <run_directory> [options]")
        print("\nExample:")
        print("  python demo_best.py runs/run_20241126_120000")
        print("\nOptions:")
        print("  --episodes <n>    Number of episodes to run (default: 5)")
        print("  --no-render       Don't render the environment")
        print("  --analyze         Show network analysis only")
        print("  --interactive     Interactive mode")
        print("  --fast            Faster animation (less delay)")
        print("  --slow            Slower animation (more delay)")
        print()
        sys.exit(1)

    run_dir = sys.argv[1]

    if not os.path.exists(run_dir):
        print(f"\n❌ Error: Directory '{run_dir}' not found!\n")
        sys.exit(1)

    # Get paths (supports both old and new structures)
    paths = get_run_paths(run_dir)

    # Load config
    config_file = paths['config']
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Load winner
    winner_file = paths['winner']
    if not os.path.exists(winner_file):
        print(f"❌ Winner file not found: {winner_file}")
        print("💡 Training might not have completed. Try loading a specific generation:")
        print(f"   python demo_best.py {run_dir} --generation <n>")
        sys.exit(1)

    with open(winner_file, 'rb') as f:
        winner = pickle.load(f)

    # Parse options
    episodes = 5
    render = True
    delay = 0.01

    if '--episodes' in sys.argv:
        try:
            idx = sys.argv.index('--episodes')
            episodes = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("❌ Invalid episodes value")
            sys.exit(1)

    if '--no-render' in sys.argv:
        render = False

    if '--fast' in sys.argv:
        delay = 0.001
    elif '--slow' in sys.argv:
        delay = 0.05

    # Run analysis
    analyze_network(winner, config)

    # Check mode
    if '--analyze' in sys.argv:
        print("Analysis complete. Use without --analyze to watch the agent play.\n")
    elif '--interactive' in sys.argv:
        demo_interactive(winner, config)
    else:
        run_winner(winner, config, render=render, episodes=episodes, delay=delay)
