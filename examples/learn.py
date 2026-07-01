"""Example: Train a hover policy with Stable-Baselines3 PPO.

Usage:
    python -m multi_drone_mujoco.examples.learn
    python -m multi_drone_mujoco.examples.learn --multiagent true
"""

import argparse
import os
from pathlib import Path

import numpy as np


def train_single(total_timesteps: int = 100_000,output_dir: str = "results/rl_hover", warm_start: bool = False, warm_start_model_path: str = None):
    """Train single-drone hover with PPO."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("[ERROR] stable-baselines3 not installed. Run: pip install stable-baselines3")
        return

    from multi_drone_mujoco.envs.hover_aviary import HoverAviary
    from multi_drone_mujoco.envs.fly_through_aviary import FlyThroughAviary
    from multi_drone_mujoco.envs.adaptive_hook_hover import AdaptiveHookHover
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Training Single-Drone Hover (PPO)")
    print(f"  Timesteps: {total_timesteps}")
    print("=" * 60)
    if args.env_type == "adaptive_hook_hover":
        env_class = AdaptiveHookHover
    elif args.env_type == "fly_through":
        env_class = FlyThroughAviary
    else:
        env_class = HoverAviary
    # Create vectorized environment
    env = make_vec_env(
        lambda: env_class(ctrl_freq=48, sim_freq=240),
        n_envs=8,
    )
    if args.env_type == "adaptive_hook_hover":
        eval_env = AdaptiveHookHover(ctrl_freq=48, sim_freq=240)
    elif args.env_type == "fly_through":
        eval_env = FlyThroughAviary(ctrl_freq=48, sim_freq=240)
    else:
        eval_env = HoverAviary(ctrl_freq=48, sim_freq=240)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=5000,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tb"),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    if warm_start and warm_start_model_path:
        model.load(warm_start_model_path)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(os.path.join(output_dir, "final_model"))
    print(f"\n[DONE] Model saved to {output_dir}/final_model.zip")

    env.close()
    eval_env.close()


def train_multi(total_timesteps: int = 200_000, output_dir: str = "results/rl_multi_hover"):
    """Train multi-drone hover with PPO (centralized)."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("[ERROR] stable-baselines3 not installed. Run: pip install stable-baselines3")
        return

    from multi_drone_mujoco.envs.multi_hover_aviary import MultiHoverAviary

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Training Multi-Drone Hover (PPO, 2 drones)")
    print(f"  Timesteps: {total_timesteps}")
    print("=" * 60)

    env = make_vec_env(
        lambda: MultiHoverAviary(
            num_drones=2,
            ctrl_freq=48,
            sim_freq=240,
            target_heights=[1.2, 0.7],
        ),
        n_envs=4,
    )

    eval_env = MultiHoverAviary(
        num_drones=2,
        ctrl_freq=48,
        sim_freq=240,
        target_heights=[1.2, 0.7],
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=5000,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tb"),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(os.path.join(output_dir, "final_model"))
    print(f"\n[DONE] Model saved to {output_dir}/final_model.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiagent", type=str, default="false")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--env_type", type=str, default="hover")
    parser.add_argument("--warm_start",type=str, default="false")
    parser.add_argument("--warm_start_model_path", type=str, default=None)
    args = parser.parse_args()

    if args.multiagent.lower() == "true":
        train_multi(total_timesteps=args.timesteps)
    else:
        train_single(
            total_timesteps=args.timesteps,
            output_dir=f"results/rl_{args.env_type}",
            warm_start=args.warm_start.lower() == "true",
            warm_start_model_path=args.warm_start_model_path
        )