"""Gymnasium environment with the same discrete movement controls as the game."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .simulator import ACTIONS, STEP, initial, update
from .observations import OBSERVATION_SIZE, observe


class GeometrySurvivorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, action_repeat: int = 6, max_seconds: float = 120.0):
        super().__init__()
        if action_repeat < 1 or max_seconds <= 0:
            raise ValueError("action_repeat and max_seconds must be positive")
        self.action_repeat = action_repeat
        self.max_seconds = max_seconds
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.game = initial()
        self._episode_steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(0, 2**32)) if seed is None else int(seed)
        self.game = initial(game_seed)
        self._episode_steps = 0
        return observe(self.game), {"game_seed": game_seed}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if self.game.dead or self.game.t >= self.max_seconds:
            raise RuntimeError("Episode finished; call reset()")
        ax, ay = ACTIONS[int(action)]
        elapsed = 0.0
        for _ in range(self.action_repeat):
            if self.game.dead or self.game.t >= self.max_seconds:
                break
            update(self.game, STEP, ax, ay)
            elapsed += STEP
        self._episode_steps += 1
        terminated = self.game.dead
        truncated = not terminated and self.game.t >= self.max_seconds
        # Survival is the objective; a collision ends the episode.
        reward = elapsed
        info = {
            "survival_seconds": self.game.t,
            "near_misses": self.game.near_count,
            "game_seed": self.game.seed,
            "episode_steps": self._episode_steps,
        }
        return observe(self.game), reward, terminated, truncated, info
