"""Contract checks for the frozen observation-only MPC.

These tests protect the public observation boundary. They do not by themselves
establish equivalence with the browser renderer or end-to-end browser gameplay.
"""
import numpy as np
from environment.simulator import Game, Enemy, initial
from environment.observations import observe, OBSERVATION_SIZE
from evaluation.mpc import MPCPolicy

def test_observation_is_finite_and_bounded():
    g=initial(123)
    g.enemies=[Enemy(100, 120, 30, -20, 12, 0, 0)]
    obs=observe(g)
    assert obs.shape==(OBSERVATION_SIZE,)
    assert np.all(np.isfinite(obs))
    assert np.all(obs >= -1) and np.all(obs <= 1)

def test_hidden_spawn_rng_and_score_cannot_change_mpc_decision():
    a=initial(123)
    a.enemies=[Enemy(200, 100, 25, 10, 12, 1, 0)]
    b=initial(987654321)
    b.enemies=[Enemy(200, 100, 25, 10, 12, 1, 0)]
    b.spawn=0.001
    b.score=999999
    b.near_count=42
    oa,ob=observe(a),observe(b)
    np.testing.assert_array_equal(oa,ob)
    assert MPCPolicy().predict(oa)[0]==MPCPolicy().predict(ob)[0]

def test_threat_encoding_is_relative_and_includes_collision_radius():
    g=Game(x=360,y=202.5,vx=20,vy=-10)
    g.enemies=[Enemy(400,202.5,40,0,12,0,0)]
    o=observe(g)
    assert np.isclose(o[4],40/720)
    assert np.isclose(o[6],20/400)
    assert np.isclose(o[7],10/400)
    assert np.isclose(o[8],20/40)
    assert o[9]==1

def test_mpc_action_always_in_discrete_action_space():
    policy=MPCPolicy()
    rng=np.random.default_rng(456)
    for _ in range(50):
        obs=rng.uniform(-1,1,OBSERVATION_SIZE).astype(np.float32)
        assert policy.predict(obs)[0] in range(9)
