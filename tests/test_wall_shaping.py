import math
import numpy as np
from environment.geometry_env import GeometrySurvivorEnv
from training.reward_shaping import PotentialSafetyReward
from training.wall_shaping import PotentialWallReward, wall_potential


def test_wall_potential_is_bounded_and_center_safer():
    center = np.zeros(53, dtype=np.float32)
    near_wall = center.copy()
    near_wall[0] = -0.97
    assert -1 <= wall_potential(center) <= 0
    assert -1 <= wall_potential(near_wall) <= 0
    assert wall_potential(center) > wall_potential(near_wall)


def test_wall_shaping_preserves_physics_observations_and_raw_reward():
    base = PotentialSafetyReward(GeometrySurvivorEnv(max_seconds=2))
    shaped = PotentialWallReward(PotentialSafetyReward(GeometrySurvivorEnv(max_seconds=2)), scale=0.2)
    a, _ = base.reset(seed=2026)
    b, _ = shaped.reset(seed=2026)
    np.testing.assert_array_equal(a, b)
    for action in [1, 5, 2, 7, 0, 3] * 50:
        a, ra, da, ta, ia = base.step(action)
        b, rb, db, tb, ib = shaped.step(action)
        np.testing.assert_array_equal(a, b)
        assert (da, ta) == (db, tb)
        assert ia["survival_seconds"] == ib["survival_seconds"]
        assert math.isclose(rb-ra, ib["wall_potential_shaping"], abs_tol=1e-9)
        if da or ta:
            break
    base.close()
    shaped.close()
