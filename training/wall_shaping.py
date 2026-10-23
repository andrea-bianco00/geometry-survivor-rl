"""Bounded wall-distance potential shaping for a controlled training-only ablation.

The potential uses ONLY the existing normalized player position observation.
It does not change physics, policy observations, actions or evaluation scoring.
"""
import math
import gymnasium as gym
from environment.simulator import W, H, PLAYER_R


def wall_potential(obs):
    x = (float(obs[0]) + 1.0) * (W / 2)
    y = (float(obs[1]) + 1.0) * (H / 2)
    # Player center must remain at least PLAYER_R from the arena boundary.
    margin = min(x - PLAYER_R, W - 11.0 - x,
                 y - 11.0, H - 11.0 - y)
    # A smooth bounded negative potential: ~0 at safe distances.
    return -math.exp(-max(0.0, margin) / 45.0)


class PotentialWallReward(gym.Wrapper):
    """Gamma-consistent difference of wall potential; terminal potential zero."""

    def __init__(self, env, scale=0.2, gamma=0.99):
        super().__init__(env)
        if scale < 0 or not 0 < gamma <= 1:
            raise ValueError("Invalid wall shaping parameters")
        self.scale = scale
        self.gamma = gamma
        self.previous_potential = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_potential = wall_potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        potential = 0.0 if terminated else wall_potential(obs)
        delta = self.scale * (self.gamma * potential - self.previous_potential)
        self.previous_potential = potential
        info = dict(info)
        info["wall_potential_shaping"] = delta
        return obs, reward + delta, terminated, truncated, info
