"""Deterministic Python port of the Geometry Survivor gameplay core.

Canonical implementation:
https://github.com/andrea-bianco00/andrea-portfolio/blob/main/app/lib/geometry-core.ts

Keep this implementation aligned with the TypeScript core. Visual-only
particles/trails are deliberately omitted: they do not affect game physics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, exp, hypot, pi, sin
from typing import List

W, H, PLAYER_R, STEP = 720, 405, 10, 1 / 120


@dataclass
class Enemy:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    kind: int
    angle: float


@dataclass
class Game:
    seed: int = 271828
    x: float = W / 2
    y: float = H / 2
    vx: float = 0.0
    vy: float = 0.0
    t: float = 0.0
    score: int = 0
    spawn: float = 1.15
    dead: bool = False
    near_count: int = 0
    enemies: List[Enemy] = field(default_factory=list)
    # JavaScript's Set<Enemy> uses object identity, not structural equality.
    near: set[int] = field(default_factory=set)


def initial(seed: int = 271828) -> Game:
    return Game(seed=seed & 0xFFFFFFFF)


def _rand(g: Game) -> float:
    g.seed = (g.seed * 1664525 + 1013904223) & 0xFFFFFFFF
    return g.seed / 4294967296


def _spawn(g: Game) -> None:
    kind = int(_rand(g) * 3)
    edge = int(_rand(g) * 4)
    x = -24 if edge == 0 else W + 24 if edge == 1 else _rand(g) * W
    y = -24 if edge == 2 else H + 24 if edge == 3 else _rand(g) * H
    aim = atan2(g.y - y, g.x - x)
    spread = (_rand(g) - 0.5) * (0.26 if kind == 2 else 1.25)
    a = aim + spread
    speed = (60 if kind == 1 else 112 if kind == 2 else 85) * (
        1 + min(g.t, 75) / 110
    )
    radius = 15 if kind == 1 else 12 if kind == 2 else 11
    g.enemies.append(
        Enemy(x, y, cos(a) * speed, sin(a) * speed, radius, kind, _rand(g) * 6.28)
    )


def update(g: Game, dt: float, ax: float, ay: float) -> None:
    if g.dead:
        return
    g.t += dt
    g.score = int(g.t * 10)
    norm = hypot(ax, ay) or 1.0
    drag = exp(-7 * dt)
    g.vx = (g.vx + ax / norm * 740 * dt) * drag
    g.vy = (g.vy + ay / norm * 740 * dt) * drag
    speed = hypot(g.vx, g.vy)
    if speed > 255:
        g.vx *= 255 / speed
        g.vy *= 255 / speed
    g.x = max(PLAYER_R + 5, min(W - PLAYER_R - 5, g.x + g.vx * dt))
    g.y = max(PLAYER_R + 5, min(H - PLAYER_R - 5, g.y + g.vy * dt))
    if g.x <= PLAYER_R + 5 or g.x >= W - PLAYER_R - 5:
        g.vx = 0.0
    if g.y <= PLAYER_R + 5 or g.y >= H - PLAYER_R - 5:
        g.vy = 0.0

    g.spawn -= dt
    if g.spawn <= 0:
        _spawn(g)
        interval = max(0.24, 1.05 - g.t * 0.010)
        g.spawn += interval * (0.78 + _rand(g) * 0.45)

    for e in g.enemies:
        e.x += e.vx * dt
        e.y += e.vy * dt
        e.angle += dt * (1.1 if e.kind == 1 else 2)
    g.enemies = [
        e for e in g.enemies
        if -70 < e.x < W + 70 and -70 < e.y < H + 70
    ]
    for e in g.enemies:
        distance = hypot(g.x - e.x, g.y - e.y)
        contact = PLAYER_R + e.r - 2
        if contact + 3 < distance < contact + 17 and id(e) not in g.near:
            g.near.add(id(e))
            g.near_count += 1
        if (g.x - e.x) ** 2 + (g.y - e.y) ** 2 < contact ** 2:
            g.dead = True
            # TypeScript generates 76 random draws for death particles.
            # These affect the seed even though particles are visual-only.
            for _ in range(76):
                _rand(g)
            break


# Stable action index contract shared with future browser inference.
ACTIONS = (
    (0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
    (-1, -1), (1, -1), (-1, 1), (1, 1),
)
