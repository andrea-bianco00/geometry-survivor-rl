"""Paired descriptive comparison; no fabricated p-values or significance claims."""
import argparse
import csv
from pathlib import Path
import numpy as np


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    seeds = [int(r["seed"]) for r in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds in {path}")
    return seeds, np.array([float(r["seconds"]) for r in rows])


def main():
    parser = argparse.ArgumentParser()
    for name in ("control", "treatment", "heuristic", "random"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experiment", choices=("reward", "observations"), default="reward")
    args = parser.parse_args()
    results = {name: read(getattr(args, name)) for name in ("control", "treatment", "heuristic", "random")}
    seeds = results["control"][0]
    if any(seeds != v[0] for v in results.values()):
        raise ValueError("All agents must be evaluated on exactly the same ordered seeds")
    control = results["control"][1]
    treatment = results["treatment"][1]
    differences = treatment - control
    title = ("Controlled observation-feature ablation" if args.experiment == "observations"
             else "Controlled reward-shaping ablation")
    shared = ("Identical game physics, action space, training reward and survival-time scoring."
              if args.experiment == "observations" else
              "Identical game physics, action space, observation representation and survival-time scoring.")
    difference_label = ("danger features minus original observations" if args.experiment == "observations"
                        else "potential minus survival")
    lines = [
        "# " + title,
        "",
        f"**{len(seeds)} paired evaluation seeds**, {min(seeds)}–{max(seeds)}. "
        shared,
        "",
        "| Agent | Mean survival (s) | Median (s) | P90 (s) |",
        "|---|---:|---:|---:|",
    ]
    for name, (_, seconds) in results.items():
        lines.append(f"| {name} | {seconds.mean():.3f} | {np.median(seconds):.3f} | {np.percentile(seconds,90):.3f} |")
    lines.extend([
        "",
        f"**Paired mean difference** ({difference_label}): {differences.mean():+.3f} seconds.",
        f"**Fraction of seeds improved**: {np.mean(differences > 0):.1%}; "
        f"tied: {np.mean(differences == 0):.1%}.",
        "",
        "**Limitations:** one training seed per run, selected diagnostic evaluation seeds, "
        "no confidence interval or significance claim. If the shaped model performs worse, "
        "that negative result must be retained. A separate untouched final test range "
        "and multiple training seeds are required before making generalization claims.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
