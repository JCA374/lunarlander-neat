#!/usr/bin/env python3
"""Failure analysis of one genome's flights -> results/failure_analysis/<run>/report.html

    python experiments/analyze_flights.py runs/<run>     # after record_flights.py

One figure per question: what failure looks like, where it happens, what drifts before it, what
predicts it (AUC and partial correlation beyond the network's own inputs), what causes it, and how
it starts. Findings are leads, not results: test them by changing the setup and retraining.
"""

import argparse
import gzip
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neat_lander import runs  # noqa: E402
from neat_lander.env import make_env  # noqa: E402

SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
BLUES = LinearSegmentedColormap.from_list('blues', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5',
                                                    '#1c5cab', '#0d366b'])
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'text.color': INK, 'axes.titlecolor': INK, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2, 'font.family': 'DejaVu Sans',
})

K = 30          # "fails soon" = within K steps
WINDOW = 150    # run-up window drawn before failure
CURRENT = ['x', 'y', 'vx', 'vy', 'angle', 'spin', 'L leg', 'R leg']      # what the network sees
CANDIDATES = ['speed', '|angle|', '|spin|', 'tilt growing', 'distance', 'main duty', 'side duty',
              'switches', 'one leg down', 'margin', 'low & fast', 'low & off-centre', 'time']
LABELS = {'x': 'x position', 'y': 'altitude', 'vx': 'horizontal velocity', 'vy': 'vertical velocity',
          'angle': 'angle', 'spin': 'angular velocity', 'L leg': 'left leg contact',
          'R leg': 'right leg contact', 'speed': 'speed |v|', '|angle|': '|angle|', '|spin|': '|angular velocity|',
          'tilt growing': 'angle × angular velocity (tilt growing)', 'distance': 'distance to pad',
          'main duty': 'main engine share, last 20 steps', 'side duty': 'side engine share, last 20 steps',
          'switches': 'action changes, last 20 steps', 'one leg down': 'exactly one leg down',
          'margin': 'best minus 2nd-best output', 'low & fast': 'falling speed below altitude 0.3',
          'low & off-centre': '|x| below altitude 0.3', 'time': 'steps flown'}


def load(path):
    with gzip.open(path, 'rt') as fh:
        header = json.loads(fh.readline())
        return header, [json.loads(line) for line in fh]


def rolling(a, n=20):
    c = np.cumsum(np.concatenate([[0.0], a]))
    idx = np.arange(1, len(a) + 1)
    lo = np.maximum(0, idx - n)
    return (c[idx] - c[lo]) / (idx - lo)


def signals(f):
    """{name: array over the flight's steps}."""
    s = np.array(f['states'])
    a = np.array(f['actions'])
    o = np.array(f['outputs'])
    srt = np.sort(o, axis=1)
    low = s[:, 1] < 0.3
    d = {k: s[:, i] for i, k in enumerate(CURRENT)}
    d.update({'speed': np.hypot(s[:, 2], s[:, 3]), '|angle|': np.abs(s[:, 4]), '|spin|': np.abs(s[:, 5]),
              'tilt growing': s[:, 4] * s[:, 5], 'distance': np.hypot(s[:, 0], s[:, 1]),
              'main duty': rolling((a == 2).astype(float)), 'side duty': rolling(np.isin(a, (1, 3)).astype(float)),
              'switches': rolling(np.concatenate([[0.0], (np.diff(a) != 0).astype(float)])),
              'one leg down': ((s[:, 6] + s[:, 7]) == 1).astype(float), 'margin': srt[:, -1] - srt[:, -2],
              'low & fast': np.where(low, -s[:, 3], 0.0), 'low & off-centre': np.where(low, np.abs(s[:, 0]), 0.0),
              'time': np.arange(len(a), dtype=float)})
    return d


def shaping(f):
    """The game's own score of a state (its potential): the reference line for the AUC chart."""
    s = np.array(f['states'])
    return -(-100 * np.hypot(s[:, 0], s[:, 1]) - 100 * np.hypot(s[:, 2], s[:, 3]) - 100 * np.abs(s[:, 4])
             + 10 * s[:, 6] + 10 * s[:, 7])


def auc(scores, labels):
    order = np.argsort(scores, kind='mergesort')
    s = scores[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = labels.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def residualize(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else 0.0


def save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, name), dpi=130, bbox_inches='tight')
    plt.close(fig)
    return name


def replay_frames(seed, actions, steps):
    """Frames at the given step indices of a recorded flight (same seed + actions = same flight)."""
    env = make_env('rgb_array')
    env.reset(seed=seed)
    want, out = set(steps), {}
    for t, a in enumerate(actions):
        if t in want:
            out[t] = env.render()
        env.step(a)
    if len(actions) in want or max(steps) >= len(actions):
        out[len(actions)] = env.render()
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('run_dir')
    args = ap.parse_args()
    name = os.path.basename(os.path.normpath(args.run_dir))
    out_dir = os.path.join(runs.ROOT, 'results', 'failure_analysis', name)
    header, flights = load(os.path.join(out_dir, 'flights.jsonl.gz'))
    fails = [f for f in flights if f['outcome'] != 'landed']
    lands = [f for f in flights if f['outcome'] == 'landed']
    sig = [signals(f) for f in flights]
    keys = CURRENT + CANDIDATES
    figures, summary = [], {'episodes': len(flights), 'landed': len(lands),
                            'crashed': sum(f['outcome'] == 'crashed' for f in flights),
                            'timeout': sum(f['outcome'] == 'timeout' for f in flights)}

    # 1. What does failure look like? -----------------------------------------------------------
    show = sorted(fails, key=lambda f: (f['cause'], f['reward']))[:12] + lands[:4]
    fig, axes = plt.subplots(4, 4, figsize=(13, 8.5))
    for ax, f in zip(axes.flat, show):
        fr = replay_frames(f['seed'], f['actions'], [len(f['actions'])])[len(f['actions'])]
        ax.imshow(fr)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        col = SERIES[7] if f['outcome'] == 'crashed' else (SERIES[3] if f['outcome'] == 'timeout' else SERIES[2])
        ax.set_title(f"{f['cause']} · {f['reward']:+.0f}", fontsize=8, color=col, fontweight='normal')
    for ax in axes.flat[len(show):]:
        ax.axis('off')
    fig.suptitle('Final frame of each failure (last row: four landings for comparison)', fontsize=11)
    fig.tight_layout()
    figures.append(('What does failure look like?', save(fig, out_dir, '1_gallery.png'),
                    f'{len(fails)} of {len(flights)} flights failed. Titles give the cause and the episode reward.'))

    # 2. Where does it happen? ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    rng = [[-1, 1], [-0.1, 1.5]]
    norm = np.concatenate([np.array(f['states'])[:, :2] for f in lands])
    late = np.concatenate([np.array(f['states'])[-K:, :2] for f in fails])
    for ax, pts, title in [(axes[0], norm, 'landing flights: all steps'),
                           (axes[1], late, f'failed flights: last {K} steps')]:
        h, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=[40, 32], range=rng, density=True)
        ax.imshow(h.T, origin='lower', extent=[*rng[0], *rng[1]], cmap=BLUES, aspect='auto')
        ax.axvspan(-0.2, 0.2, ymin=0, ymax=0.06, color=SERIES[3], alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel('x (pad between ±0.2)')
        ax.set_ylabel('altitude')
        ax.grid(False)
    ax = axes[2]
    causes = sorted({f['cause'] for f in fails})
    for i, c in enumerate(causes):
        p = np.array([f['final'][:2] for f in fails if f['cause'] == c])
        ax.scatter(p[:, 0], p[:, 1], s=28, color=SERIES[[7, 3, 1][i % 3]], label=c)
    ax.set_xlim(*rng[0])
    ax.set_ylim(*rng[1])
    ax.axvspan(-0.2, 0.2, ymin=0, ymax=0.06, color=SERIES[3], alpha=0.4)
    ax.set_title('where each failed flight ended')
    ax.set_xlabel('x')
    ax.legend(frameon=False)
    fig.tight_layout()
    on_pad = np.mean([abs(f['final'][0]) < 0.2 for f in fails])
    summary['failures_ending_over_pad'] = float(on_pad)
    figures.append(('Where does it happen?', save(fig, out_dir, '2_where.png'),
                    f'Darker = more time spent there. {100 * on_pad:.0f}% of failures end above the pad (|x| < 0.2).'))

    # 3. What drifts before failure? ------------------------------------------------------------
    drift_keys = ['vy', 'angle', 'spin', 'speed', 'side duty', 'main duty', 'x', 'switches']
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2))
    normal = {k: np.concatenate([s[k] for s, f in zip(sig, flights) if f['outcome'] == 'landed']) for k in keys}
    departure = {}
    for k in keys:
        M = np.full((len(fails), WINDOW), np.nan)
        for i, f in enumerate(fails):
            v = sig[flights.index(f)][k][-WINDOW:]
            M[i, WINDOW - len(v):] = v
        med = np.nanmedian(M, axis=0)
        p90 = np.percentile(np.abs(normal[k]), 90)
        out = np.where(np.abs(med) > p90)[0]
        departure[k] = int(WINDOW - out[0]) if len(out) else None
        if k in drift_keys:
            ax = axes.flat[drift_keys.index(k)]
            xs = np.arange(-WINDOW, 0)
            col = SERIES[0] if k in CURRENT else SERIES[1]
            ax.fill_between(xs, np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0), color=col, alpha=0.18, lw=0)
            ax.plot(xs, med, color=col)
            ax.axhline(np.median(normal[k]), color=INK_2, ls='--', lw=1)
            ax.set_title(LABELS[k], fontsize=10)
            ax.set_xlabel('steps before failure')
    fig.tight_layout()
    figures.append(('What drifts before failure?', save(fig, out_dir, '3_drift.png'),
                    'Median and interquartile band over the last 150 steps of failed flights. Dashed: median over '
                    'landing flights. Blue = an input the network sees, orange = a derived candidate.'))

    # 4. What predicts failure? -----------------------------------------------------------------
    pos, vals = [], {k: [] for k in keys}
    ref = []
    for s, f in zip(sig, flights):
        n = len(f['actions'])
        if f['outcome'] == 'landed':
            m = np.arange(n) < n - K              # landing flights, minus their final touchdown steps
            lab = np.zeros(n, bool)
        else:
            m = np.ones(n, bool)
            lab = np.arange(n) >= n - K
        pos.append(lab[m])
        for k in keys:
            vals[k].append(s[k][m])
        ref.append(shaping(f)[m])
    labels = np.concatenate(pos)
    stats = {}
    ttf, cand_fail = [], {k: [] for k in keys}
    for f in fails:
        s = sig[flights.index(f)]
        n = len(f['actions'])
        idx = np.arange(max(0, n - WINDOW), n)
        ttf.append(n - idx)
        for k in keys:
            cand_fail[k].append(s[k][idx])
    ttf = np.concatenate(ttf).astype(float)
    X_cur = np.column_stack([np.concatenate(cand_fail[k]) for k in CURRENT])
    y_res = residualize(ttf, X_cur)
    for k in keys:
        v = np.concatenate(vals[k])
        a = auc(v, labels)
        stats[k] = {'auc': max(a, 1 - a), 'direction': 'higher' if a >= 0.5 else 'lower'}
        if k in CANDIDATES:
            c = np.concatenate(cand_fail[k])
            stats[k]['partial_corr'] = -corr(residualize(c, X_cur), y_res)   # + = higher value -> fails sooner
        else:
            stats[k]['partial_corr'] = None
    ref_auc = auc(np.concatenate(ref), labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    order = sorted(keys, key=lambda k: stats[k]['auc'])
    ax1.barh([LABELS[k] for k in order], [stats[k]['auc'] for k in order],
             color=[SERIES[0] if k in CURRENT else SERIES[1] for k in order])
    ax1.axvline(ref_auc, color=INK, ls='--', lw=1)
    ax1.text(ref_auc, len(order) - 0.4, f" game's shaping score {ref_auc:.2f}", fontsize=8, va='bottom')
    ax1.set_xlim(0.5, 1)
    ax1.set_title(f'Warning power: AUC for "fails within {K} steps"')
    for i, k in enumerate(order):
        ax1.text(stats[k]['auc'] + 0.005, i, f"{stats[k]['auc']:.2f}", va='center', fontsize=7, color=INK_2)
    cand = sorted(CANDIDATES, key=lambda k: abs(stats[k]['partial_corr']))
    pc = [stats[k]['partial_corr'] for k in cand]
    ax2.barh([LABELS[k] for k in cand], pc, color=[SERIES[1] if v > 0 else '#b4b2ac' for v in pc])
    ax2.axvline(0, color=INK_2, lw=1)
    for i, v in enumerate(pc):
        ax2.text(v + (0.005 if v >= 0 else -0.005), i, f'{v:+.2f}', va='center', ha='left' if v >= 0 else 'right',
                 fontsize=7, color=INK_2)
    ax2.set_title('Added information beyond the 8 inputs')
    ax2.set_xlabel('partial correlation with "fails sooner"\n(after regressing out the 8 inputs the network sees)')
    fig.tight_layout()
    summary['shaping_auc'] = ref_auc
    summary['signals'] = stats
    figures.append(('What predicts failure?', save(fig, out_dir, '4_predict.png'),
                    'Left: AUC of each signal alone for failing within 30 steps (positives: the last 30 steps of failed '
                    'flights; negatives: landing flights before touchdown). Right: what each derived signal adds about '
                    'time-to-failure once the network\'s own inputs are accounted for.'))

    # 5. What causes it? ------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    cnt = {c: sum(f['cause'] == c for f in fails) for c in causes}
    ax1.barh(list(cnt), list(cnt.values()), color=SERIES[7])
    for i, v in enumerate(cnt.values()):
        ax1.text(v + 0.2, i, str(v), va='center', fontsize=9)
    ax1.set_title(f'Why the {len(fails)} failures ended')
    push_f = [np.hypot(*f['states'][0][2:4]) for f in fails]
    push_l = [np.hypot(*f['states'][0][2:4]) for f in lands]
    bins = np.linspace(0, max(push_f + push_l), 20)
    ax2.hist(push_l, bins=bins, density=True, color=SERIES[2], alpha=0.6, label='landed')
    ax2.hist(push_f, bins=bins, density=True, color=SERIES[7], alpha=0.6, label='failed')
    ax2.set_title('Size of the random starting push')
    ax2.set_xlabel('initial speed')
    ax2.legend(frameon=False)
    fig.tight_layout()
    summary['causes'] = cnt
    summary['start_push_median'] = {'failed': float(np.median(push_f)), 'landed': float(np.median(push_l))}
    figures.append(('What causes it?', save(fig, out_dir, '5_causes.png'),
                    'Left: how failures end. Right: whether failures start with a bigger random push than landings.'))

    # 6. How does it start? ---------------------------------------------------------------------
    thr = {k: np.percentile(np.abs(normal[k]), 75) for k in ('speed', '|angle|', '|spin|')}
    picks = sorted(fails, key=lambda f: f['reward'])[:3]
    fig, axes = plt.subplots(len(picks), 4, figsize=(14, 2.6 * len(picks)))
    onsets = []
    for r, f in enumerate(picks):
        s = sig[flights.index(f)]
        ok = (s['speed'] <= thr['speed']) & (s['|angle|'] <= thr['|angle|']) & (s['|spin|'] <= thr['|spin|'])
        n = len(f['actions'])
        healthy = int(np.where(ok)[0][-1]) if ok.any() else 0
        onsets.append(n - healthy)
        steps = [healthy, min(n - 1, healthy + 10), min(n - 1, healthy + 30), n]
        frames = replay_frames(f['seed'], f['actions'], steps)
        for c, t in enumerate(steps):
            ax = axes[r, c]
            ax.imshow(frames[t])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            lab = ['last healthy state', '+10 steps', '+30 steps', f"end: {f['cause']}"][c]
            ax.set_title(f'{lab} (step {t})', fontsize=8, fontweight='normal', color=INK_2)
    fig.tight_layout()
    summary['onset_to_failure_steps'] = onsets
    figures.append(('How does it start?', save(fig, out_dir, '6_onset.png'),
                    'The three worst failures. "Healthy" = speed, |angle| and |angular velocity| all inside the '
                    '75th percentile of landing flights.'))

    # summary + report --------------------------------------------------------------------------
    for k in keys:
        stats[k]['normal_median'] = float(np.median(normal[k]))
        stats[k]['median_at_failure'] = float(np.median([sig[flights.index(f)][k][-1] for f in fails]))
        stats[k]['leaves_normal_range_steps_before'] = departure[k]
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)

    def row(k):
        s = stats[k]
        tag = ' <span class=cur>input</span>' if k in CURRENT else ''
        pc = '' if s['partial_corr'] is None else f"{s['partial_corr']:+.2f}"
        dep = '' if s['leaves_normal_range_steps_before'] is None else s['leaves_normal_range_steps_before']
        return (f"<tr><td>{LABELS[k]}{tag}</td><td>{s['normal_median']:.2f}</td><td>{s['median_at_failure']:.2f}</td>"
                f"<td>{s['auc']:.2f} ({s['direction']})</td><td>{pc}</td><td>{dep}</td></tr>")
    rows = ''.join(row(k) for k in sorted(keys, key=lambda k: -stats[k]['auc']))
    secs = ''.join(f'<section><h2>{t}</h2><p>{d}</p><img src="{img}" alt="{t}"></section>' for t, img, d in figures)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Flight analysis: {name} winner</title>
<style>body{{font-family:system-ui,sans-serif;background:{SURFACE};color:{INK};max-width:1200px;margin:24px auto;padding:0 16px}}
h1{{margin-bottom:4px}} .sub{{color:{INK_2}}} img{{max-width:100%;border:1px solid {GRID};border-radius:6px}}
table{{border-collapse:collapse;font-size:14px;margin:12px 0}} td,th{{padding:4px 10px;border-bottom:1px solid {GRID};text-align:right}}
td:first-child,th:first-child{{text-align:left}} .cur{{font-size:11px;color:#fff;background:{SERIES[0]};border-radius:4px;padding:1px 5px}}
section{{margin:28px 0}}</style></head><body>
<h1>Flight analysis: {name} winner (genome {header['genome']})</h1>
<p class="sub">{len(flights)} flights on analysis seeds {header['seed0']:,}+ (never used in training, validation or test):
{summary['landed']} landed, {summary['crashed']} crashed, {summary['timeout']} timed out. "Fails soon" = within {K} steps.</p>
<table><tr><th>signal</th><th>median, landing flights</th><th>median at failure</th><th>AUC (warning)</th>
<th>added info (partial corr.)</th><th>leaves normal range (steps before)</th></tr>{rows}</table>
<p class="sub">AUC of the game's own shaping score of the state: {ref_auc:.2f}. "Leaves normal range" = steps before failure
where the median first exceeds the 90th percentile of |value| on landing flights. Findings are leads: test them by retraining.</p>
{secs}</body></html>"""
    with open(os.path.join(out_dir, 'report.html'), 'w') as fh:
        fh.write(html)
    print(f'Wrote {out_dir}/report.html')


if __name__ == '__main__':
    main()
