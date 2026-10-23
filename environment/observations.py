"""Fixed-length policy observations; do not modify without versioning."""
import math
import numpy as np
from .simulator import Game, PLAYER_R, W, H

OBSERVATION_VERSION = 1
MAX_THREATS = 8
OBSERVATION_SIZE = 53


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


def observe(g: Game) -> np.ndarray:
    values = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    values[:4] = (
        _clip(g.x / (W / 2) - 1),
        _clip(g.y / (H / 2) - 1),
        _clip(g.vx / 255),
        _clip(g.vy / 255),
    )

    def priority(e):
        dx, dy = e.x - g.x, e.y - g.y
        vx, vy = e.vx - g.vx, e.vy - g.vy
        v2 = vx * vx + vy * vy
        closest_t = max(0.0, min(2.0, -(dx * vx + dy * vy) / v2)) if v2 > 1e-6 else 0.0
        clearance = max(0.0, math.hypot(dx + vx * closest_t, dy + vy * closest_t) - (PLAYER_R + e.r - 2))
        return clearance, closest_t, math.hypot(dx, dy)

    for i, e in enumerate(sorted(g.enemies, key=priority)[:MAX_THREATS]):
        k = 4 + i * 6
        values[k:k+6] = (
            _clip((e.x - g.x) / W),
            _clip((e.y - g.y) / H),
            _clip((e.vx - g.vx) / 400),
            _clip((e.vy - g.vy) / 400),
            _clip((PLAYER_R + e.r - 2) / 40),
            1.0,
        )
    values[-1] = min(g.t, 75) / 75
    return values
