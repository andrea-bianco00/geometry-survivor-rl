import numpy as np
from environment.geometry_env import GeometrySurvivorEnv
from training.danger_features import DangerFeatures, danger_features


def test_danger_features_are_finite_bounded_and_deterministic():
    env = GeometrySurvivorEnv()
    obs, _ = env.reset(seed=2026)
    a = danger_features(obs)
    b = danger_features(obs)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (4,)
    assert np.isfinite(a).all()
    assert np.all(np.abs(a) <= 1)
    env.close()


def test_observation_wrapper_does_not_change_game():
    base = GeometrySurvivorEnv(max_seconds=2)
    enriched = DangerFeatures(GeometrySurvivorEnv(max_seconds=2))
    a, _ = base.reset(seed=123)
    b, _ = enriched.reset(seed=123)
    np.testing.assert_array_equal(a, b[:53])
    assert b.shape == (57,)
    for action in [0, 5, 1, 3, 8, 2, 7, 4] * 20:
        a, ra, da, ta, ia = base.step(action)
        b, rb, db, tb, ib = enriched.step(action)
        np.testing.assert_array_equal(a, b[:53])
        assert ra == rb
        assert (da, ta) == (db, tb)
        assert ia["survival_seconds"] == ib["survival_seconds"]
        if da or ta:
            break
    base.close()
    enriched.close()
