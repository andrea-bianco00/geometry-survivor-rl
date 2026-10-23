"""Audit of the frozen MPC controller, using new seeds and no tuning."""
import csv
import math
import statistics
import time
from pathlib import Path
import numpy as np
from environment.geometry_env import GeometrySurvivorEnv
from evaluation.mpc import MPCPolicy
from evaluation.heuristic import ClosestApproachBaseline

SEEDS = tuple(range(9300000, 9300200))
CAP = 120
AGENTS = {"heuristic": ClosestApproachBaseline, "mpc": MPCPolicy}

def run(agent_factory, seeds):
    env = GeometrySurvivorEnv(max_seconds=CAP)
    rows = []
    policy = agent_factory()
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        durations = []
        terminated = truncated = False
        while not (terminated or truncated):
            before = time.perf_counter_ns()
            action, _ = policy.predict(obs, deterministic=True)
            durations.append((time.perf_counter_ns() - before) / 1e6)
            obs, _, terminated, truncated, info = env.step(int(action))
        rows.append(dict(seed=seed, seconds=float(info["survival_seconds"]),
                         terminated=terminated, truncated=truncated,
                         decisions=len(durations),
                         mean_decision_ms=statistics.mean(durations),
                         max_decision_ms=max(durations)))
    env.close()
    return rows

def summary(rows):
    values = [r["seconds"] for r in rows]
    return (statistics.mean(values), statistics.median(values),
            sum(r["truncated"] for r in rows), len(rows))

def main():
    output = Path("results/mpc_validation")
    output.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, factory in AGENTS.items():
        rows = run(factory, SEEDS)
        data[name] = rows
        with (output / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    differences = [m["seconds"] - h["seconds"]
                   for m,h in zip(data["mpc"],data["heuristic"])]
    assert all(m["seed"] == h["seed"] for m,h in zip(data["mpc"],data["heuristic"]))
    # Paired seed bootstrap for uncertainty; fixed RNG solely for resampling.
    rng = np.random.default_rng(20261023)
    samples = [statistics.mean(rng.choice(differences, size=len(differences), replace=True))
               for _ in range(5000)]
    lo, hi = np.quantile(samples,[0.025,0.975])
    report = [
        "# Frozen MPC independent-seed validation",
        "",
        "200 predeclared seeds 9300000–9300199; 120-second cap; identical game rules.",
        "No tuning or selection from these results. The controller remains unchanged.",
        "This is an independent *seed set*, not a guarantee of independence from every design choice.",
        "",
        "| Agent | Mean (s) | Median (s) | Censored at 120 s |",
        "|---|---:|---:|---:|"
    ]
    for name, rows in data.items():
        mean, median, censored, n = summary(rows)
        report.append(f"| {name} | {mean:.3f} | {median:.3f} | {censored}/{n} |")
    report += ["", f"Paired mean difference (MPC - heuristic): {statistics.mean(differences):.3f} s",
               f"Paired bootstrap 95% interval: [{lo:.3f}, {hi:.3f}] s (descriptive; capped times)",
               f"MPC mean decision wall time: {statistics.mean(r['mean_decision_ms'] for r in data['mpc']):.3f} ms",
               f"MPC worst measured decision wall time: {max(r['max_decision_ms'] for r in data['mpc']):.3f} ms",
               "",
               "Timing is GitHub Actions CPU timing, not browser timing. 120 s results are right-censored.",
               "No random or future seed information is passed to either controller.",
               "This benchmark does not prove parity with the browser runtime; that requires separate integration tests."]
    report_text = "\n".join(report) + "\n"
    (output / "report.md").write_text(report_text)
    print(report_text)
    import os
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(report_text)

if __name__ == "__main__":
    main()
