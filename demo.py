#!/usr/bin/env python3
"""Watch a genome fly.

    python demo.py runs/<run>                     # the winner (or latest champion while training)
    python demo.py runs/<run>/genomes/winner.pkl --episodes 10 --seed 7
    python demo.py runs/<run> --generation 50     # the champion of generation 50
"""

import argparse

from neat_lander import runs
from neat_lander.env import make_env, make_net, run_episode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('target', nargs='?', default=None, help='run dir or genome .pkl')
    ap.add_argument('--generation', type=int, default=None)
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0, help='first episode seed')
    args = ap.parse_args()
    target = args.target or runs.latest_run()
    genome, config, run_dir = runs.load_genome_arg(target)
    if args.generation is not None:
        genome = runs.load_champions(run_dir)[args.generation]

    net, env = make_net(genome, config), make_env('human')
    for i in range(args.episodes):
        ep = run_episode(net, args.seed + i, env=env)
        print(f"episode {i + 1}: {ep['outcome']:8s} reward {ep['reward']:7.1f}  ({ep['steps']} steps)")
    env.close()


if __name__ == '__main__':
    main()
