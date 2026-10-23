"""Reproducible PPO training entrypoint. No pretrained weights are included."""
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment.geometry_env import GeometrySurvivorEnv
from training.reward_shaping import PotentialSafetyReward
from training.danger_features import DangerFeatures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/ppo_geometry"))
    parser.add_argument("--reward-mode", choices=("survival", "potential"), default="survival")
    parser.add_argument("--potential-scale", type=float, default=0.5)
    parser.add_argument("--danger-features", action="store_true", help="Append four observation-only danger features")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = GeometrySurvivorEnv()
    if args.reward_mode == 'potential':
        env = PotentialSafetyReward(env, scale=args.potential_scale)
    if args.danger_features:
        env = DangerFeatures(env)
    env = Monitor(env)
    model = PPO(
        "MlpPolicy", env, seed=args.seed, verbose=1,
        tensorboard_log=None, device="cpu",
    )
    model.learn(total_timesteps=args.steps)
    model.save(str(args.output))
    print(f'Training reward mode: {args.reward_mode}; potential scale: {args.potential_scale}')
    print(f"Saved model to {args.output}.zip; NOT yet evaluated.")
    env.close()


if __name__ == "__main__":
    main()
