#!/usr/bin/env python3
"""Draw the champion network of every N-th generation, plus an animated GIF of its growth.

    python networks.py runs/<run>            # every 10th generation -> <run>/networks/
    python networks.py runs/<run> --every 5
"""

import argparse
import os

from neat_lander import runs
from neat_lander.draw import network_image
from neat_lander.recorder import network_size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('run_dir', nargs='?', default=None)
    ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--ms', type=int, default=350, help='GIF frame duration')
    args = ap.parse_args()
    run_dir = args.run_dir or runs.latest_run()

    config = runs.load_config(run_dir)
    champs = runs.load_champions(run_dir)
    val = {r['generation']: r for r in runs.read_csv(run_dir, 'validation')}
    out = runs.path(run_dir, 'networks')
    os.makedirs(out, exist_ok=True)

    gens = [g for g in sorted(champs) if g % args.every == 0]
    if max(champs) not in gens:
        gens.append(max(champs))
    frames = []
    for g in gens:
        hidden, conns = network_size(champs[g], config)
        v = val.get(g, {})
        sub = (f'{hidden} hidden nodes, {conns} connections  |  training fitness {champs[g].fitness:.0f}'
               + (f', {v["reward"]:.0f} on 100 unseen episodes ({100 * v["landed"]:.0f}% land)' if v else ''))
        img = network_image(champs[g], config, title=f'Champion of generation {g}', subtitle=sub)
        img.save(os.path.join(out, f'net_g{g:04d}.png'))
        frames.append(img)
    frames[0].save(os.path.join(out, 'evolution.gif'), save_all=True, append_images=frames[1:] + [frames[-1]] * 4,
                   duration=args.ms, loop=0)
    print(f'{len(frames)} networks + evolution.gif -> {out}')


if __name__ == '__main__':
    main()
