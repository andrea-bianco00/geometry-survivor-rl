# Geometry Survivor RL — architecture

## Goal
Train a reinforcement-learning policy to survive the same Geometry Survivor arena used by the public portfolio game.

## Source of truth
The game implementation is maintained in [andrea-portfolio](https://github.com/andrea-bianco00/andrea-portfolio/blob/main/app/lib/geometry-core.ts).
A Python environment must reproduce the game physics and obstacle generator, including deterministic seed, collision logic, time step (1/120 s), speed cap, drag, spawning and difficulty ramp. Automated cross-language parity tests are required before trusting training results.

## Observation proposal (v1)
53 float features: player normalized position and velocity (4); up to eight threats, each with relative position, relative velocity, collision radius and presence mask (8×6); elapsed difficulty (1). Threats are ranked by predicted closest approach over a two-second horizon. This ranking is feature engineering, not a hand-coded policy.

## Action proposal (v1)
Nine discrete actions: idle, four cardinal directions, four diagonals. Hold each action for six simulation ticks (20 policy decisions per second).

## Experiments
1. Establish random and simple heuristic baselines on held-out seeds.
2. Train PPO with Gymnasium + Stable-Baselines3; evaluate multiple training seeds.
3. Report mean, median, p90 and distribution of survival time on held-out game seeds.
4. Compare identical episode limits and seed sets, and report variability.
5. Export policy for local browser inference only after successful parity and evaluation.

## Web integration
WATCH AI will replace player input in the *existing* game canvas, never create a second homepage game. Display live survival, episodes and agent actions; an optional vision overlay will display only genuine model observations. Do not submit AI scores to the human leaderboard.

## Status
Research scaffold only. No trained agent, benchmark result or model artifact has been produced yet.
