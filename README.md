# Geometry Survivor — Autonomous AI Agent

**Can an agent survive by predicting danger instead of just reacting?**

Geometry Survivor is a geometric obstacle-dodging game featured in [my live portfolio](https://andrea-bianco.com). This repository documents the Python simulation, reinforcement-learning experiments, and the final **Model Predictive Control (MPC)** agent running in the browser.

## How it works

The deployed AI is **not a trained neural network**. Every 0.05 seconds it observes the player and up to eight visible threats, simulates nine possible actions over a 1.2-second horizon, scores collision risks and wall proximity, and briefly executes the best action before planning again. Its observation vector has 53 features. It cannot see future spawns or the random seed.

We also experimented with Proximal Policy Optimization (PPO), a reinforcement-learning algorithm, but it underperformed simpler controllers in our trials. MPC was selected based on measured results, not because it is a more fashionable technique.

## Results

A frozen-controller comparison used the same **200 predeclared seeds (9300000–9300199)**, with episodes capped at 120 seconds:

| Controller | Mean survival | Median | Reached 120 s |
|---|---:|---:|---:|
| Heuristic | 47.51 s | 47.49 s | 0 / 200 |
| **MPC** | **107.57 s** | **120.00 s** | **102 / 200** |

The paired mean difference was **60.06 seconds** (bootstrap 95% interval **56.36–63.62 s**). These are *capped* survival times: 120-second episodes are right-censored. The numbers come from the Python simulator, not a browser performance benchmark. Earlier PPO and random-agent trials used different evaluation protocols and should not be compared as one matched experiment.

## Reproduce

Python 3.11 recommended. Create and activate a virtual environment, then run:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m evaluation.validate_mpc
```

The validation writes episode CSVs and a report to `results/mpc_validation/`. Experimental PPO training:

```bash
python -m training.train_ppo --steps 100000 --seed 42 --output models/ppo_geometry
```

## Explore the code

- `environment/` — seeded game simulation, observations and Gymnasium wrapper
- `training/` — PPO training
- `evaluation/mpc.py` — observation-only MPC policy
- `evaluation/heuristic.py` — comparison baseline
- `evaluation/validate_mpc.py` — paired validation
- `tests/` — automated checks
- [Live TypeScript MPC](https://github.com/andrea-bianco00/andrea-portfolio/blob/main/app/lib/geometry-mpc.ts) and [game component](https://github.com/andrea-bianco00/andrea-portfolio/blob/main/app/components/GeometrySurvivor.tsx)

The physics runs at **120 Hz**, with AI decisions at **20 Hz**. The live AI demo never submits scores to the human leaderboard.

**Takeaway:** the goal is to build, compare, and deploy an autonomous decision-making system with reproducible evidence—not to claim that every intelligent controller is machine learning.

MIT License.
