# Geometry Survivor — Reinforcement Learning

A reproducible reinforcement-learning research project for **Geometry Survivor**, the geometric survival game featured on [Andrea Bianco's portfolio](https://andrea-bianco.com).

> **Current status:** the Python simulator, Gymnasium wrapper, PPO training script and baseline evaluation scripts are implemented. **No trained model or validated performance results have been published yet.** Cross-language parity with the live TypeScript simulator remains a required verification step.

## Research question

Can a reinforcement-learning agent learn to survive longer than random movement in the *same* arena that human visitors play?

The project uses **Proximal Policy Optimization (PPO)** and evaluates on deterministic, held-out game seeds. The live game's physics is the canonical specification: [geometry-core.ts](https://github.com/andrea-bianco00/andrea-portfolio/blob/main/app/lib/geometry-core.ts).

## Project structure

- `environment/simulator.py` — Python port of the seeded game physics.
- `environment/observations.py` — 53-dimensional normalized observations.
- `environment/geometry_env.py` — Gymnasium environment with nine discrete actions.
- `training/train_ppo.py` — reproducible PPO training entrypoint.
- `evaluation/evaluate.py` — held-out survival-time evaluation and random baseline.
- `tests/` — deterministic and Gymnasium API tests.
- `docs/architecture.md` — design decisions and remaining verification work.

## Install

Python 3.11 recommended.

```bash
python -m venv .venv
# Activate .venv for your operating system.
python -m pip install -r requirements.txt
python -m pytest -q
```

## Train (experimental)

```bash
python -m training.train_ppo --steps 100000 --seed 42 --output models/ppo_geometry
```

This command produces a **candidate model**, not a validated result. Training for longer or tuning hyperparameters may be necessary.

## Evaluate

```bash
python -m evaluation.evaluate --seeds 100 --seed-start 1000000 --output results/random.csv
python -m evaluation.evaluate --model models/ppo_geometry.zip --seeds 100 --seed-start 1000000 --output results/ppo.csv
```

Both runs use the same held-out game seeds. Random actions use a separately seeded, reproducible RNG. The evaluator prints mean, median and 90th percentile survival time and saves per-episode CSV data. A 120-second episode limit applies to both agents.

## Important limitations

- The Python port has **not yet been independently certified against the original TypeScript engine**. Cross-language state-by-state parity tests must be added before reporting trustworthy comparative results.
- A random baseline alone is not sufficient: a simple heuristic baseline and multi-seed PPO experiments are planned.
- The initial reward is elapsed survival time; no claim is made that it produces a strong policy without experiments.
- No neural model currently runs on the portfolio. A future **WATCH AI** mode will use the same existing canvas, load a verified trained policy on demand, and never submit AI scores to the human leaderboard.
- The code is research infrastructure; metrics and training curves will be published only when measured.

## Reproducibility

Game physics runs at 120 Hz. One policy action is held for six physics ticks (20 decisions per second). Observations include player state, up to eight predicted closest-approach threats, and elapsed difficulty. See [architecture](docs/architecture.md).

## License

MIT (see LICENSE).

## Run a real PPO pilot in GitHub Actions (no local installation)

1. Open [Actions → PPO pilot experiment](https://github.com/andrea-bianco00/geometry-survivor-rl/actions/workflows/train-ppo.yml).
2. Select **Run workflow**, choose the default **30,000 timesteps**, **50 evaluation seeds**, and **training seed 42**, then confirm.
3. Wait for the job to finish. The run summary contains measured mean, median and p90 survival for **random**, **non-learning heuristic**, and **trained PPO**, evaluated on identical held-out game seeds.
4. Download the run's `ppo-pilot-<run-id>` artifact for the trained model (`.zip`), per-episode CSV files and Markdown summary. Artifacts are retained for 30 days; they are not automatically committed to the repository.

**Scientific caution:** 30,000 steps is a *smoke-test/pilot*, not evidence of a good policy. If PPO underperforms, report that honestly and improve training. The job reruns the simulator parity suite before training. This workflow is manually triggered to avoid consuming GitHub Actions minutes on every commit.

## Experimental protocol after the 30k pilot

The first 30k-step run used evaluation seeds 1000000–1000049. It is a *pilot*, and the results were inspected before selecting the next training budget. **Do not treat those same seeds as an untouched test set for model selection.**

For the next diagnostic run, use **300000 training steps, seed 42, 100 evaluation seeds, evaluation seed start 2000000**. Compare the 30k and 300k results cautiously because the evaluation seed sets differ; rerun both checkpoints on one common untouched seed set for a strictly paired comparison. Reserve an additional independent range (e.g. 4000000+) for final reporting after all hyperparameters are frozen.

The heuristic is a non-learning baseline, not the deployed RL agent. A single training seed does not establish stability: eventually run at least three training seeds and report variability. Do not claim statistical significance from a single run.
