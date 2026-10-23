/**
 * Pure TypeScript port of evaluation/mpc.py.
 * Accepts ONLY the 53-element public observation vector.
 * No simulator access, hidden spawn schedule, or seed access.
 * Do not deploy before cross-language and browser parity tests.
 */
export const MPC_ACTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 0], [0, -1], [0, 1], [-1, 0], [1, 0],
  [-1, -1], [1, -1], [-1, 1], [1, 1],
];
const W = 720, H = 405;
export function mpcAction(obs: ReadonlyArray<number>, horizon = 1.2, dt = 0.05): number {
  if (obs.length !== 53 || !obs.every(Number.isFinite)) throw new Error("Invalid MPC observation");
  const px = (obs[0] + 1) * W / 2, py = (obs[1] + 1) * H / 2;
  const vx = obs[2] * 255, vy = obs[3] * 255;
  const threats: Array<[number, number, number, number, number]> = [];
  for (let j = 0; j < 8; j++) {
    const k = 4 + 6 * j;
    if (obs[k + 5] <= 0) continue;
    threats.push([px + obs[k] * W, py + obs[k + 1] * H,
      vx + obs[k + 2] * 400, vy + obs[k + 3] * 400, obs[k + 4] * 40]);
  }
  let bestAction = 0, bestScore = -Infinity;
  const n = Math.round(horizon / dt);
  for (let action = 0; action < MPC_ACTIONS.length; action++) {
    const [ax, ay] = MPC_ACTIONS[action];
    let x = px, y = py, ux = vx, uy = vy, score = 0;
    const norm = Math.hypot(ax, ay) || 1;
    for (let i = 1; i <= n; i++) {
      const t = i * dt, drag = Math.exp(-7 * dt);
      ux = (ux + ax / norm * 740 * dt) * drag;
      uy = (uy + ay / norm * 740 * dt) * drag;
      const speed = Math.hypot(ux, uy);
      if (speed > 255) { ux = ux * 255 / speed; uy = uy * 255 / speed; }
      x = Math.max(15, Math.min(W - 15, x + ux * dt));
      y = Math.max(15, Math.min(H - 15, y + uy * dt));
      if (x === 15 || x === W - 15) ux = 0;
      if (y === 15 || y === H - 15) uy = 0;
      const wall = Math.min(x - 15, W - 15 - x, y - 15, H - 15 - y);
      score += 0.0005 * Math.min(wall, 90) * dt;
      for (const [ex, ey, evx, evy, radius] of threats) {
        const clearance = Math.hypot(x - (ex + evx * t), y - (ey + evy * t)) - radius;
        if (clearance < 0) score -= 1000 * dt;
        else score -= 2.5 * Math.max(0, 45 - clearance) ** 2 / 2025 * dt;
      }
    }
    if (score > bestScore) { bestScore = score; bestAction = action; }
  }
  return bestAction;
}
