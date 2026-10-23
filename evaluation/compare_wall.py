"""Report paired effects separately for each independent training seed."""
import argparse
import csv
from pathlib import Path
import numpy as np


def load(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [int(r["seed"]) for r in rows], np.array([float(r["seconds"]) for r in rows])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=Path("results/wall"))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    a = p.parse_args()
    lines = ["# PPO wall-potential experiment", "",
             "Identical simulator, observations, actions, potential safety reward, evaluation seeds and steps.",
             "Only the added wall potential differs. All models trained from scratch.", "",
             "| Training seed | Control mean (s) | Wall treatment mean (s) | Paired difference (s) | Improved game seeds |",
             "|---|---:|---:|---:|---:|"]
    effects = []
    for seed in a.seeds:
        cs, c = load(a.dir / f"control_{seed}.csv")
        ts, t = load(a.dir / f"wall_{seed}.csv")
        if cs != ts:
            raise ValueError(f"Unpaired game seeds for training seed {seed}")
        d = t - c
        effects.append(float(d.mean()))
        lines.append(f"| {seed} | {c.mean():.3f} | {t.mean():.3f} | {d.mean():+.3f} | {np.mean(d>0):.1%} |")
    lines += ["", f"Mean of per-training-seed effects: {np.mean(effects):+.3f} seconds.",
              f"Number of training seeds improved: {sum(e > 0 for e in effects)}/{len(effects)}.",
              "", "Exploratory comparison only: training seeds and evaluation range were selected after prior diagnostics.",
              "These results are not an untouched final test and do not establish generalization.",
              "No policy was replaced by the heuristic; performance is real measured survival time."]
    report = "\n".join(lines) + "\n"
    (a.dir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
