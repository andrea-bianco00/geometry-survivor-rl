"""Potential-based risk shaping for TRAINING ONLY.

Phi is a bounded, deterministic function of the *observable* state.
Reward = elapsed_seconds + gamma * Phi(next_obs) - Phi(obs)
on nonterminal transitions. On collision the terminal potential is zero.
For gamma-discounted returns, this is policy-invariant potential shaping
under the usual MDP assumptions. It is NOT used for evaluation scores.
"""
from __future__ import annotations
import math
import gymnasium as gym
import numpy as np


def safety_potential(obs: np.ndarray) -> float:
    """A bounded potential rewarding states with less imminent collision risk."""
    vx = float(obs[2]) * 255
    vy = float(obs[3]) * 255
    risks = []
    for i in range(8):
        j = 4 + i * 6
        if obs[j + 5] < 0.5:
            continue
        dx = float(obs[j]) * 720
        dy = float(obs[j + 1]) * 405
        dvx = float(obs[j + 2]) * 400
        dvy = float(obs[j + 3]) * 400
        v2 = dvx * dvx + dvy * dvy
        closest_t = max(0.0, min(0.8, -(dx * dvx + dy * dvy) / v2)) if v2 > 1e-8 else 0.0
        clearance = math.hypot(dx + dvx * closest_t, dy + dvy * closest_t) - float(obs[j + 4]) * 40
        # A risk of 1 denotes imminent collision; risk falls off smoothly.
        risk = math.exp(-max(-20.0, clearance) / 38.0) * math.exp(-closest_t / 0.65)
        risks.append(min(2.0, risk))
    danger = max(risks, default=0.0)
    return -min(1.0, danger)


class PotentialSafetyReward(gym.Wrapper):
    """Training-only potential shaping; never modify game physics or test metric."""

    def __init__(self, env: gym.Env, scale: float = 0.5, gamma: float = 0.99):
        super().__init__(env)
        if scale < 0 or not 0 < gamma <= 1:
            raise ValueError("Invalid potential shaping parameters")
        self.scale = scale
        self.gamma = gamma
        self.previous_potential = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_potential = safety_potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Time-limit truncation retains bootstrap potential; death does not.
        next_potential = 0.0 if terminated else safety_potential(obs)
        shaping = self.scale * (self.gamma * next_potential - self.previous_potential)
        self.previous_potential = next_potential
        info = dict(info)
        info["raw_survival_reward"] = reward
        info["potential_shaping"] = shaping
        return obs, reward + shaping, terminated, truncated, info
