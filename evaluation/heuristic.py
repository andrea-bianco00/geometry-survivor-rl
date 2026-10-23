"""Transparent, non-learning baseline: avoid predicted collisions.

This is a comparator only, never presented as the trained AI.
"""
from math import hypot
from environment.simulator import ACTIONS, PLAYER_R, W, H


class ClosestApproachBaseline:
    def predict(self, obs, deterministic=True):
        # Observations contain the player and eight tracked obstacles.
        # Candidate velocity is approximated from the chosen acceleration.
        px = (float(obs[0]) + 1) * W / 2
        py = (float(obs[1]) + 1) * H / 2
        vx = float(obs[2]) * 255
        vy = float(obs[3]) * 255
        best_action, best_score = 0, float("-inf")
        for i, (ax, ay) in enumerate(ACTIONS):
            norm = hypot(ax, ay) or 1.0
            candidate_vx = vx * 0.65 + ax / norm * 75
            candidate_vy = vy * 0.65 + ay / norm * 75
            x = min(W - 15, max(15, px + candidate_vx * 0.55))
            y = min(H - 15, max(15, py + candidate_vy * 0.55))
            # Prefer open space and avoid the walls without rewarding camping.
            clearance = min(x - 15, W - 15 - x, y - 15, H - 15 - y)
            score = 0.012 * clearance
            for j in range(8):
                k = 4 + 6 * j
                if float(obs[k + 5]) == 0:
                    continue
                ex = px + float(obs[k]) * W
                ey = py + float(obs[k + 1]) * H
                evx = vx + float(obs[k + 2]) * 400
                evy = vy + float(obs[k + 3]) * 400
                radius = float(obs[k + 4]) * 40
                for horizon in (0.3, 0.6, 1.0):
                    dx = ex + evx * horizon - (px + candidate_vx * horizon)
                    dy = ey + evy * horizon - (py + candidate_vy * horizon)
                    distance = hypot(dx, dy) - radius
                    score -= max(0, 85 - distance) ** 2 / (1500 * horizon)
            if score > best_score:
                best_action, best_score = i, score
        return best_action, None
