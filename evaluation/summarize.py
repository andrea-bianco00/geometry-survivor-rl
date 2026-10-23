"""Generate a factual markdown report from actual per-episode evaluation CSVs."""
import argparse
import csv
from pathlib import Path
import numpy as np


def load(path):
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty evaluation file: {path}")
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate evaluation seeds: {path}")
    return seeds, np.asarray([float(row["seconds"]) for row in rows], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    for name in ("random", "heuristic", "ppo"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = {name: load(getattr(args, name)) for name in ("random", "heuristic", "ppo")}
    seeds = data["random"][0]
    if any(item[0] != seeds for item in data.values()):
        raise ValueError("All agents must use exactly the same ordered evaluation seeds")
    lines = [
        "# Geometry Survivor — PPO pilot evaluation",
        "",
        f"Evaluation episodes per agent: **{len(seeds)}**. Held-out game seeds: "
        f"**{min(seeds)}–{max(seeds)}**.",
        "",
        "| Agent | Mean survival (s) | Median (s) | P90 (s) |",
        "|---|---:|---:|---:|",
    ]
    for name, (_, seconds) in data.items():
        lines.append(
            f"| {name.title()} | {seconds.mean():.3f} | "
            f"{np.median(seconds):.3f} | {np.percentile(seconds, 90):.3f} |"
        )
    random_mean = data["random"][1].mean()
    heuristic_mean = data["heuristic"][1].mean()
    ppo_mean = data["ppo"][1].mean()
    lines.extend([
        "",
        f"PPO / random mean survival ratio: **{ppo_mean / random_mean:.3f}×**.",
        f"PPO / heuristic mean survival ratio: **{ppo_mean / heuristic_mean:.3f}×**.",
        "",
        "**Interpretation:** This is a pilot evaluation, not a claim of generalization "
        "across training seeds. Larger experiments and confidence intervals are needed.",
        "All episode times are capped by the same evaluation time limit.",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
