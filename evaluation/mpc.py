"""Observation-only receding-horizon controller. No game state, RNG, or future spawns."""
from __future__ import annotations
from math import exp, hypot
from environment.simulator import ACTIONS, W, H

class MPCPolicy:
    """Evaluate nine constant-action trajectories; execute only their first action.

    Predicts visible enemies by constant velocity; never calls the game simulator.
    This is a limited MPC baseline, not an oracle or a trained neural policy.
    """
    def __init__(self, horizon=1.2, dt=0.05):
        self.horizon, self.dt = horizon, dt

    def predict(self, obs, deterministic=True):
        px, py = (float(obs[0])+1)*W/2, (float(obs[1])+1)*H/2
        vx, vy = float(obs[2])*255, float(obs[3])*255
        threats = []
        for j in range(8):
            k = 4+6*j
            if float(obs[k+5]) <= 0:
                continue
            threats.append((px+float(obs[k])*W, py+float(obs[k+1])*H,
                            vx+float(obs[k+2])*400, vy+float(obs[k+3])*400,
                            float(obs[k+4])*40))
        best_action, best_score = 0, float("-inf")
        n = round(self.horizon/self.dt)
        for action, (ax, ay) in enumerate(ACTIONS):
            x,y,ux,uy = px,py,vx,vy
            score = 0.
            norm = hypot(ax,ay) or 1.
            for i in range(1,n+1):
                t = i*self.dt
                drag = exp(-7*self.dt)
                ux = (ux+ax/norm*740*self.dt)*drag
                uy = (uy+ay/norm*740*self.dt)*drag
                speed = hypot(ux,uy)
                if speed > 255:
                    ux,uy = ux*255/speed,uy*255/speed
                x = max(15,min(W-15,x+ux*self.dt))
                y = max(15,min(H-15,y+uy*self.dt))
                if x in (15,W-15): ux=0.
                if y in (15,H-15): uy=0.
                wall = min(x-15,W-15-x,y-15,H-15-y)
                score += 0.0005*min(wall,90)*self.dt
                for ex,ey,evx,evy,r in threats:
                    clearance = hypot(x-(ex+evx*t),y-(ey+evy*t))-r
                    if clearance < 0:
                        score -= 1000*self.dt
                    else:
                        score -= 2.5*max(0,45-clearance)**2/2025*self.dt
            if score > best_score:
                best_score, best_action = score,action
        return best_action, None
