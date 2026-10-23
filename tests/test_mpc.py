import numpy as np
from evaluation.mpc import MPCPolicy

def test_mpc_is_deterministic_and_uses_only_observation():
    obs=np.zeros(53,dtype=np.float32)
    a=MPCPolicy()
    assert a.predict(obs)[0] == a.predict(obs.copy())[0]
    assert 0 <= a.predict(obs)[0] < 9

def test_mpc_ignores_unobserved_rng_and_seed():
    obs=np.zeros(53,dtype=np.float32)
    obs[4:10]=[0.1,0,0,0,0.5,1]
    assert MPCPolicy().predict(obs)[0] == MPCPolicy().predict(obs)[0]
