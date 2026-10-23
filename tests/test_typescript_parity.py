"""Compare live TypeScript engine against Python on identical seeds/actions.

Requires GEOMETRY_TS_CORE pointing to the actual portfolio geometry-core.ts.
CI checks out the public portfolio source. No hand-maintained JS simulator.
"""
import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from environment.simulator import ACTIONS, STEP, initial, update


@pytest.mark.parametrize("seed,mode", [
    (1, 0), (42, 1), (271828, 2), (4294967295, 3), (20261023, 4),
])
def test_reference_parity(seed, mode):
    source = os.getenv("GEOMETRY_TS_CORE")
    if not source:
        pytest.skip("Set GEOMETRY_TS_CORE to run cross-language parity")
    script = Path(__file__).with_name("ts_reference.mjs")
    result = subprocess.run(
        ["node", "--experimental-strip-types", str(script)],
        capture_output=True, text=True, check=True, timeout=60,
        env=os.environ.copy(),
    )
    test = next(item for item in json.loads(result.stdout) if item["seed"] == seed)
    assert test["mode"] == mode
    g = initial(seed)
    expected = {entry["tick"]: entry["state"] for entry in test["states"]}
    for tick in range(1, max(expected) + 1):
        ax, ay = ACTIONS[((tick - 1) // 6 + mode * 3) % len(ACTIONS)]
        update(g, STEP, ax, ay)
        if tick not in expected:
            continue
        state = expected[tick]
        for key in ("seed", "score", "dead", "nearCount"):
            actual = g.near_count if key == "nearCount" else getattr(g, key)
            assert actual == state[key], (seed, tick, key, actual, state[key])
        for key in ("x", "y", "vx", "vy", "t", "spawn"):
            assert math.isclose(getattr(g, key), state[key], abs_tol=1e-7, rel_tol=1e-9), (seed,tick,key)
        assert len(g.enemies) == len(state["enemies"]), (seed, tick, "enemy count")
        for actual, reference in zip(g.enemies, state["enemies"]):
            assert actual.kind == reference["kind"]
            for key in ("x", "y", "vx", "vy", "r", "angle"):
                assert math.isclose(getattr(actual, key), reference[key], abs_tol=1e-7, rel_tol=1e-9), (seed,tick,key)
        if g.dead:
            break
