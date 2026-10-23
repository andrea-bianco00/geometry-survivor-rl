"""Inspect a trained PPO policy without changing the game or reward.

Report action distribution, idle frequency, episode lengths, and player movement
on fixed seeds. Diagnostic results are measured, not assumed.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from environment.geometry_env import GeometrySurvivorEnv


def diagnose(model, seeds, max_seconds=120.0):
    env = GeometrySurvivorEnv(max_seconds=max_seconds)
    episodes = []
    actions = Counter()
    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))
        counts = Counter()
        positions = []
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            action = int(action)
            counts[action] += 1
            actions[action] += 1
            observation, _, terminated, truncated, info = env.step(action)
            positions.append((env.game.x, env.game.y))
            done = terminated or truncated
        travel = sum(
            np.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(positions, positions[1:])
        )
        episodes.append({
            "seed": int(seed),
            "survival_seconds": round(float(info["survival_seconds"]), 6),
            "action_decisions": sum(counts.values()),
            "idle_fraction": counts[0] / max(1, sum(counts.values())),
            "distance_travelled_px": round(float(travel), 3),
            "distinct_actions": len(counts),
            **{f"action_{i}": counts[i] for i in range(env.action_space.n)},
        })
    env.close()
    return episodes, dict(sorted(actions.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=3000000)
    parser.add_argument("--output", type=Path, default=Path("results/ppo_diagnostics.csv"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    model = PPO.load(str(args.model), device="cpu")
    rows, action_counts = diagnose(model, range(args.seed_start, args.seed_start + args.seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "episodes": len(rows),
        "mean_survival_seconds": float(np.mean([r["survival_seconds"] for r in rows])),
        "mean_idle_fraction": float(np.mean([r["idle_fraction"] for r in rows])),
        "mean_distance_travelled_px": float(np.mean([r["distance_travelled_px"] for r in rows])),
        "total_action_counts": action_counts,
        "action_indices": "0 idle; 1 up; 2 down; 3 left; 4 right; 5 up-left; 6 up-right; 7 down-left; 8 down-right",
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
