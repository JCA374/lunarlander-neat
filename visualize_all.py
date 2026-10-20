#!/usr/bin/env python3
"""Make every picture for a run: plots, networks and an evaluation.

    python visualize_all.py runs/<run>
"""

import subprocess
import sys

from neat_lander import runs

run_dir = sys.argv[1] if len(sys.argv) > 1 else runs.latest_run()
for script in ('plots.py', 'networks.py', 'evaluate.py'):
    print(f'\n== {script}')
    subprocess.run([sys.executable, script, run_dir], check=True)
