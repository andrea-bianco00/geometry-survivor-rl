import numpy as np
from evaluation.evaluate import evaluate
from evaluation.heuristic import ClosestApproachBaseline


def test_random_baseline_is_reproducible():
    a = evaluate(None, [40001, 40002], max_seconds=3)
    b = evaluate(None, [40001, 40002], max_seconds=3)
    assert a == b


def test_heuristic_runs_with_valid_actions():
    policy = ClosestApproachBaseline()
    action, state = policy.predict(np.zeros(53, dtype=np.float32))
    assert 0 <= action <= 8
    assert state is None
    rows = evaluate(policy, [40001, 40002], max_seconds=3)
    assert len(rows) == 2
    assert all(0 < row["seconds"] <= 3.01 for row in rows)
