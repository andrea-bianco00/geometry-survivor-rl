"""Cross-language frozen MPC decision parity; no controller tuning or game-state access."""
import json
import subprocess
from pathlib import Path
import numpy as np
from environment.simulator import Game, Enemy
from environment.observations import observe
from evaluation.mpc import MPCPolicy

ROOT=Path(__file__).resolve().parents[1]
JS=r"""
import {mpcAction} from './browser/mpc.ts';
let data='';
for await (const chunk of process.stdin) data+=chunk;
const observations=JSON.parse(data);
process.stdout.write(JSON.stringify(observations.map(o=>mpcAction(o))));
"""

def test_frozen_mpc_python_typescript_decision_parity():
    rng=np.random.default_rng(20261023)
    observations=[]
    # Explicit empty-threat, wall and high-density situations plus random cases.
    for i in range(120):
        g=Game(x=float(rng.uniform(15,705)), y=float(rng.uniform(15,390)),
               vx=float(rng.uniform(-180,180)), vy=float(rng.uniform(-180,180)),
               t=float(rng.uniform(0,75)))
        if i==0:
            g.x,g.y,g.vx,g.vy=360,202.5,0,0
        if i==1: g.x,g.y=15,15
        if i==2: g.x,g.y=705,390
        count=0 if i==0 else (25 if i%5==0 else int(rng.integers(1,14)))
        g.enemies=[Enemy(float(rng.uniform(-30,750)),float(rng.uniform(-30,435)),
                         float(rng.uniform(-160,160)),float(rng.uniform(-160,160)),
                         float(rng.choice([11,12,15])),0,0) for _ in range(count)]
        observations.append(observe(g).tolist())
    expected=[MPCPolicy().predict(np.asarray(o,dtype=np.float32))[0] for o in observations]
    result=subprocess.run(['node','--experimental-strip-types','--input-type=module','-e',JS],
        input=json.dumps(observations),text=True,capture_output=True,cwd=ROOT,timeout=45,check=True)
    actual=json.loads(result.stdout)
    mismatches=[(i,p,t) for i,(p,t) in enumerate(zip(expected,actual)) if p!=t]
    assert len(actual)==len(expected)
    assert not mismatches, f"Python/TypeScript MPC action mismatch: {mismatches[:10]}"
