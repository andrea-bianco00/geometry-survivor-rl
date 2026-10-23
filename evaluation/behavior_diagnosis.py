"""Behavioral diagnosis of saved PPO checkpoints on paired game seeds.

Uses actual simulator states only for post-hoc measurement. The policy receives
only its normal observation. No retraining and no modifications to gameplay.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from environment.geometry_env import GeometrySurvivorEnv
from environment.simulator import PLAYER_R, W, H
from training.danger_features import DangerFeatures
from evaluation.heuristic import ClosestApproachBaseline


def evaluate_agent(policy, seeds):
    env = GeometrySurvivorEnv()
    if policy is not None and hasattr(policy, "observation_space"):
        size = policy.observation_space.shape[0]
        if size == 57:
            env = DangerFeatures(env)
        elif size != 53:
            raise ValueError(f"Unexpected model observation dimension: {size}")
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        previous_action = None
        actions = Counter()
        switches = 0
        wall_decisions = 0
        danger_decisions = 0
        unsafe_decisions = 0
        min_clearances = []
        first_three = Counter()
        steps = 0
        while True:
            g = env.unwrapped.game
            action = int(policy.predict(obs, deterministic=True)[0])
            actions[action] += 1
            if g.t < 3:
                first_three[action] += 1
            if previous_action is not None and previous_action != action:
                switches += 1
            previous_action = action
            wall = min(g.x - PLAYER_R, W - PLAYER_R - g.x,
                       g.y - PLAYER_R, H - PLAYER_R - g.y)
            if wall < 30:
                wall_decisions += 1
            clearances = [
                math.hypot(e.x - g.x, e.y - g.y) - (PLAYER_R + e.r - 2)
                for e in g.enemies
            ]
            clearance = min(clearances, default=float("inf"))
            if math.isfinite(clearance):
                min_clearances.append(clearance)
            if clearance < 70:
                danger_decisions += 1
                if action == 0:
                    unsafe_decisions += 1
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        g = env.unwrapped.game
        collision_wall_distance = min(g.x - PLAYER_R, W - PLAYER_R - g.x,
                                      g.y - PLAYER_R, H - PLAYER_R - g.y)
        rows.append({
            "seed": int(seed),
            "survival_seconds": round(float(info["survival_seconds"]), 6),
            "collision_x": round(g.x, 3),
            "collision_y": round(g.y, 3),
            "collision_wall_distance_px": round(collision_wall_distance, 3),
            "death_near_wall_30px": int(terminated and collision_wall_distance < 30),
            "action_decisions": steps,
            "action_switch_fraction": switches / max(1, steps - 1),
            "idle_fraction": actions[0] / steps,
            "wall_fraction": wall_decisions / steps,
            "danger_fraction": danger_decisions / steps,
            "idle_when_danger_fraction": unsafe_decisions / max(1, danger_decisions),
            "mean_nearest_clearance_px": float(np.mean(min_clearances)) if min_clearances else float("nan"),
            "first_3s_idle_fraction": first_three[0] / max(1, sum(first_three.values())),
            **{f"action_{i}_fraction": actions[i] / steps for i in range(9)},
        })
    env.close()
    return rows


def summarize(rows):
    fields = ("survival_seconds", "death_near_wall_30px", "action_switch_fraction",
              "idle_fraction", "wall_fraction", "danger_fraction",
              "idle_when_danger_fraction", "mean_nearest_clearance_px",
              "first_3s_idle_fraction")
    return {key: round(float(np.nanmean([r[key] for r in rows])), 4) for key in fields}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--enriched", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=3700000)
    parser.add_argument("--output", type=Path, default=Path("results/behavior"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    policies = {
        "original_ppo": PPO.load(str(args.original), device="cpu"),
        "enriched_ppo": PPO.load(str(args.enriched), device="cpu"),
        "heuristic": ClosestApproachBaseline(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, policy in policies.items():
        rows = evaluate_agent(policy, seeds)
        with (args.output / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        summaries[name] = summarize(rows)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    labels = {
        "survival_seconds": "Mean survival (s)",
        "death_near_wall_30px": "Fraction dying near wall (<30px)",
        "action_switch_fraction": "Fraction changing action",
        "idle_fraction": "Fraction idle",
        "wall_fraction": "Fraction decisions near wall",
        "danger_fraction": "Fraction decisions with obstacle <70px",
        "idle_when_danger_fraction": "Idle fraction when obstacle <70px",
        "mean_nearest_clearance_px": "Mean instantaneous nearest clearance (px)",
        "first_3s_idle_fraction": "Fraction idle in first 3s",
    }
    lines = [
        "# Behavioral diagnosis of trained PPO checkpoints", "",
        f"Saved checkpoints; {args.seeds} paired game seeds {args.seed_start}–{args.seed_start+args.seeds-1}.",
        "Policies are evaluated deterministically; no training is performed.",
        "Game state is used only to measure outcomes, never passed to the policy.",
        "The heuristic is a non-learning comparator.", "",
        "| Metric | Original PPO | Enriched PPO | Heuristic |",
        "|---|---:|---:|---:|",
    ]
    for key, label in labels.items():
        lines.append("| " + label + " | " + " | ".join(
            f"{summaries[name][key]:.3f}" for name in policies) + " |")
    lines += ["", "These are descriptive measurements, not causal explanations.",
              "A higher action-switch rate does not by itself prove oscillation is harmful.",
              "Near-wall deaths and instantaneous danger rates are correlated with survival duration and enemy density.",
              "The checkpoints were selected after prior experiments; this is diagnostic, not untouched final testing."]
    report = "\n".join(lines) + "\n"
    (args.output / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
