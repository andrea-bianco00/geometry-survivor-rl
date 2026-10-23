import math
import numpy as np
from environment.geometry_env import GeometrySurvivorEnv
from training.reward_shaping import PotentialSafetyReward, safety_potential


def test_empty_arena_has_zero_risk():
    obs = np.zeros(53, dtype=np.float32)
    assert safety_potential(obs) == 0


def test_approaching_obstacle_has_negative_potential():
    obs = np.zeros(53, dtype=np.float32)
    obs[4:10] = (0.02, 0.0, -0.3, 0.0, 0.5, 1)
    assert -1 <= safety_potential(obs) < 0


def test_shaping_does_not_change_physics_or_episode_end():
    original = GeometrySurvivorEnv(max_seconds=2)
    shaped = PotentialSafetyReward(GeometrySurvivorEnv(max_seconds=2))
    a, _ = original.reset(seed=2026)
    b, _ = shaped.reset(seed=2026)
    np.testing.assert_array_equal(a, b)
    for i in range(100):
        action = (i // 3) % 9
        a, raw, done_a, trunc_a, info_a = original.step(action)
        b, reward, done_b, trunc_b, info_b = shaped.step(action)
        np.testing.assert_array_equal(a, b)
        assert (done_a, trunc_a) == (done_b, trunc_b)
        assert math.isclose(info_a["survival_seconds"], info_b["survival_seconds"])
        assert math.isclose(raw, info_b["raw_survival_reward"])
        assert math.isfinite(reward)
        if done_a or trunc_a:
            break
    original.close()
    shaped.close()
