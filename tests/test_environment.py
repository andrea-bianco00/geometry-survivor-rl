import numpy as np
import pytest

from environment.simulator import ACTIONS, STEP, initial, update
from environment.observations import OBSERVATION_SIZE, observe

gym = pytest.importorskip("gymnasium")
from environment.geometry_env import GeometrySurvivorEnv


def test_initial_state_and_observation():
    game = initial(42)
    assert game.x == 360 and game.y == 202.5
    obs = observe(game)
    assert obs.shape == (OBSERVATION_SIZE,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert obs[4:-1].sum() == 0


def test_same_seed_and_actions_reproduce_exactly():
    a, b = initial(42), initial(42)
    for step in range(3000):
        ax, ay = ACTIONS[(step // 6) % len(ACTIONS)]
        update(a, STEP, ax, ay)
        update(b, STEP, ax, ay)
        assert (a.seed, a.x, a.y, a.vx, a.vy, a.t, a.score, a.spawn, a.dead, a.near_count) == (b.seed, b.x, b.y, b.vx, b.vy, b.t, b.score, b.spawn, b.dead, b.near_count)
        assert a.enemies == b.enemies
        if a.dead:
            break


def test_environment_obeys_gymnasium_api():
    env = GeometrySurvivorEnv(max_seconds=0.15)
    observation, info = env.reset(seed=123)
    assert env.observation_space.contains(observation)
    assert info["game_seed"] == 123
    done = False
    while not done:
        observation, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(observation)
        assert reward >= 0
        done = terminated or truncated
    assert info["survival_seconds"] >= 0.15
    with pytest.raises(RuntimeError):
        env.step(0)


def test_invalid_actions_rejected():
    env = GeometrySurvivorEnv()
    env.reset(seed=3)
    with pytest.raises(ValueError):
        env.step(99)
