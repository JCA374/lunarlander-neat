# Repository Guidelines

## Project Structure & Module Organization
- Root scripts: `train.py` (NEAT training), `demo_best.py` (playback), `visualize_network.py`, `visualize_evolution.py`, `plot_fitness.py`.
- Config and deps: `neat-config.ini`, `requirements.txt`, `setup.sh`.
- Docs: `README.md`, `QUICKSTART.md`, `PROJECT_SUMMARY.txt`, `UNDERSTANDING_OUTPUT.md`, `NEXT_CHALLENGE_ANALYSIS.md`, `COMMANDS.txt`.
- Outputs: `runs/run_YYYYMMDD_HHMM/` holding per-generation genomes, stats, plots, and GIFs.

## Build, Test, and Development Commands
- Install deps (Box2D included): `pip install -r requirements.txt` (or `./setup.sh` to create venv and install).
- Train LunarLander agent: `python train.py` (default 5000 generations, creates `runs/run_*`, saves best every 10 generations).
- Visualize networks: `python visualize_network.py runs/run_*`.
- Plot fitness: `python plot_fitness.py runs/run_*`.
- Create evolution GIF: `python visualize_evolution.py runs/run_*`.
- Demo best genome: `python demo_best.py runs/run_* [--episodes N --no-render]`.

## Coding Style & Naming Conventions
- Language: Python 3. Indent with 4 spaces, prefer clear inline comments only where logic is non-obvious.
- Avoid Unicode in code; stick to ASCII identifiers.
- Use descriptive variable names; functions should be small and purpose-driven.
- Configuration lives in `neat-config.ini`; keep training hyperparameters there, not hard-coded.

## Testing Guidelines
- No automated test suite is present. Validate changes by running a short smoke train (e.g., 2–3 generations) and running `demo_best.py` against a saved run.
- For plotting/visualization changes, regenerate assets on a recent `runs/run_*` directory to confirm output renders without errors.
- If adding tests, place them under `tests/` (create if needed) and use `pytest`; name files `test_*.py`.

## Commit & Pull Request Guidelines
- Commit messages: use clear, imperative summaries (e.g., “Add LunarLander labels to visualizer”). Group related changes per commit when possible.
- PRs should include: what changed, why, how to reproduce (commands), and any screenshots/artifacts for visuals (plots, GIFs).
- Note training cost: avoid committing large run artifacts unless required; reference the run directory path instead.

## Security & Configuration Tips
- Gymnasium Box2D is required: ensure `gymnasium[box2d]` is installed.
- Avoid checking in virtualenvs or large binary outputs from `runs/`; keep them git-ignored if a VCS is used.
- When changing config defaults, update corresponding docs (`README.md`, `QUICKSTART.md`) to keep instructions accurate.
