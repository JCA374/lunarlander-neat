#!/usr/bin/env python3
"""Training plots for a run -> <run>/plots/

    python plots.py runs/<run>

  training.png   fitness, species, network size and validation in one figure
  species.png    the classic NEAT picture: species sizes stacked over the generations
"""

import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from neat_lander import runs


def col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def species_matrix(species_rows):
    """(generations, species ids, sizes[species, generation]) for a stacked-area plot."""
    gens = sorted({r['generation'] for r in species_rows})
    sids = sorted({r['species'] for r in species_rows})
    gi = {g: i for i, g in enumerate(gens)}
    si = {s: i for i, s in enumerate(sids)}
    sizes = np.zeros((len(sids), len(gens)))
    for r in species_rows:
        sizes[si[r['species']], gi[r['generation']]] = r['size']
    return np.array(gens), sids, sizes


def plot_species(ax, species_rows):
    gens, sids, sizes = species_matrix(species_rows)
    cmap = plt.get_cmap('tab20')
    ax.stackplot(gens, sizes, colors=[cmap(i % 20) for i in range(len(sids))], linewidth=0)
    ax.set_xlim(gens[0], gens[-1])
    ax.set_ylim(0, sizes.sum(axis=0).max())
    ax.set_ylabel('genomes')
    ax.set_title(f'Species ({len(sids)} appeared over the run)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('run_dir', nargs='?', default=None)
    args = ap.parse_args()
    run_dir = args.run_dir or runs.latest_run()
    config = runs.load_config(run_dir)
    target = config.fitness_threshold

    fit = runs.read_csv(run_dir, 'fitness')
    cpx = runs.read_csv(run_dir, 'complexity')
    val = runs.read_csv(run_dir, 'validation')
    spc = runs.read_csv(run_dir, 'species')
    out = runs.path(run_dir, 'plots')
    os.makedirs(out, exist_ok=True)
    g = col(fit, 'generation')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    ax = axes[0, 0]
    ax.plot(g, col(fit, 'best'), color='#2f6fd6', lw=1, alpha=.6, label='best (training episodes)')
    ax.plot(g, col(fit, 'mean'), color='#8b97b8', lw=1, label='population mean')
    ax.plot(col(val, 'generation'), col(val, 'reward'), color='#1d8f55', lw=2,
            label='champion on 100 unseen episodes')
    ax.axhline(target, color='#b86e00', ls='--', lw=1, label=f'solved ({target:.0f})')
    ax.set_ylim(max(-400, col(fit, 'mean').min()), max(320, col(fit, 'best').max() + 10))
    ax.set_ylabel('average reward')
    ax.set_title('Fitness')
    ax.legend(loc='lower right', fontsize=9, frameon=False)

    plot_species(axes[0, 1], spc)

    ax = axes[1, 0]
    ax.plot(col(cpx, 'generation'), col(cpx, 'mean_conns'), color='#8b97b8', label='connections (mean)')
    ax.plot(col(cpx, 'generation'), col(cpx, 'best_conns'), color='#2f6fd6', lw=1, label='connections (champion)')
    ax2 = ax.twinx()
    ax2.plot(col(cpx, 'generation'), col(cpx, 'mean_hidden'), color='#e0a030', label='hidden nodes (mean)')
    ax2.plot(col(cpx, 'generation'), col(cpx, 'best_hidden'), color='#b86e00', lw=1, label='hidden nodes (champion)')
    ax.set_ylabel('enabled connections')
    ax2.set_ylabel('hidden nodes')
    ax.set_xlabel('generation')
    ax.set_title('Network size')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9, frameon=False)

    ax = axes[1, 1]
    ax.plot(col(val, 'generation'), 100 * col(val, 'landed'), color='#1d8f55', lw=2,
            label='champion: landings on 100 unseen episodes')
    ax.plot(g, 100 * col(fit, 'landers'), color='#8b97b8', label='population: share landing ≥ half its episodes')
    ax.set_ylim(0, 100)
    ax.set_ylabel('%')
    ax.set_xlabel('generation')
    ax.set_title('Landing')
    ax.legend(loc='lower right', fontsize=9, frameon=False)

    for a in axes.flat:
        a.grid(alpha=.25)
        a.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'NEAT on LunarLander: {os.path.basename(os.path.normpath(run_dir))}', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'training.png'), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    plot_species(ax, spc)
    ax.set_xlabel('generation')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'species.png'), dpi=130)
    plt.close(fig)
    print(f'training.png, species.png -> {out}')


if __name__ == '__main__':
    main()
