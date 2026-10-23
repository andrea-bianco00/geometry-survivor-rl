"""Evaluate random or PPO agents on the SAME fixed, held-out game seeds."""
import argparse
import csv
from pathlib import Path

import numpy as np
from environment.geometry_env import GeometrySurvivorEnv


def evaluate(policy, seeds, max_seconds=120):
    env = GeometrySurvivorEnv(max_seconds=max_seconds)
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        rng = np.random.default_rng(int(seed) ^ 0xA51C)
        done = False
        while not done:
            if policy is None:
                action = int(rng.integers(env.action_space.n))
            else:
                action, _ = policy.predict(obs, deterministic=True)
                action = int(action)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        rows.append({"seed": int(seed), "seconds": float(info["survival_seconds"]),
                     "terminated": bool(terminated), "truncated": bool(truncated)})
    env.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.csv"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    policy = None
    if args.model is not None:
        from stable_baselines3 import PPO
        policy = PPO.load(str(args.model), device="cpu")
    rows = evaluate(policy, range(args.seed_start, args.seed_start + args.seeds))
    times = np.array([row["seconds"] for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"n={len(times)} mean={times.mean():.3f}s median={np.median(times):.3f}s "
          f"p90={np.percentile(times,90):.3f}s | saved {args.output}")


if __name__ == "__main__":
    main()
