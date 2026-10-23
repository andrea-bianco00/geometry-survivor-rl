"""Observation-only engineering: deterministic danger features, no privileged state.

All features are derived exclusively from the same 53 observed numbers already
available to PPO. This does not change physics, actions, reward, or game seeds.
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def danger_features(obs):
    # Player position and velocity are obs[0:4]; eight tracked hazards follow.
    # Hazard relative velocities are already relative to the player.
    nearest = 1.0
    soonest = 1.0
    imminent = 0
    approaching = 0
    for i in range(8):
        j = 4 + 6 * i
        if obs[j + 5] < 0.5:
            continue
        dx = float(obs[j]) * 720.0
        dy = float(obs[j + 1]) * 405.0
        dvx = float(obs[j + 2]) * 400.0
        dvy = float(obs[j + 3]) * 400.0
        radius = float(obs[j + 4]) * 40.0 + 14.0
        speed2 = dvx * dvx + dvy * dvy
        t = max(0.0, min(1.0, -(dx * dvx + dy * dvy) / speed2)) if speed2 > 1e-9 else 0.0
        clearance = math.hypot(dx + dvx * t, dy + dvy * t) - radius
        nearest = min(nearest, max(-1.0, min(1.0, clearance / 150.0)))
        if clearance < 40:
            imminent += 1
            soonest = min(soonest, t)
        if dx * dvx + dy * dvy < 0:
            approaching += 1
    # All output components are within [-1,1].
    return np.array([
        nearest,
        soonest,
        min(1.0, imminent / 4.0),
        min(1.0, approaching / 8.0),
    ], dtype=np.float32)


class DangerFeatures(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        original = env.observation_space
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(original.shape[0] + 4,),
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.concatenate((observation, danger_features(observation))).astype(np.float32)
