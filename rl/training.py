"""RL training module with PPO and custom MuJoCo environment."""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import typer
import yaml
import numpy as np
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback,BaseCallback
from stable_baselines3.common.monitor import Monitor
from common.support import load_config
from rich.console import Console
from rl.hover_env import env_creator, DroneHoverEnv
import torch
from typing import Callable
logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer()


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def make_env(env_cfg, rank):
    def _init():
        env = env_creator(env_cfg)
        return Monitor(env)
    return _init


def make_vec(env_cfg: dict, n_envs: int):
    if n_envs == 1:
        return make_env(env_cfg, 0)()

    return SubprocVecEnv([make_env(env_cfg, i) for i in range(n_envs)])


# ----------------------------
# CALLBACKS
# ----------------------------
def make_callbacks(cfg, eval_env, model_dir, log_dir):

    train = cfg["rl_training_params"]

    save_freq = max(1, train["save_freq"] // train["num_envs"])
    eval_freq = max(1, train["eval_freq"] // train["num_envs"])

    return [
        CheckpointCallback(
            save_freq=save_freq,
            save_path=str(model_dir),
            name_prefix="model",
        ),

        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "best_model"), 
            log_path=str(log_dir),
            eval_freq=eval_freq,
            n_eval_episodes=train["n_eval_episodes"],
            deterministic=True,
            render=False,
        ),
    ]



@app.command()
def train(config: Optional[str] = typer.Option(None, "--config", "-c")):
    cfg = load_config(config)
    train_cfg = cfg["rl_training_params"]
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_rl")
    root = Path("results") / run_name
    model_dir = root / "models"
    log_dir = root / "logs" 

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ENV
    train_env = make_vec(cfg, train_cfg["num_envs"])
    eval_env = make_env(cfg,1)().env
    net = [int(x) for x in eval_env.net_arch]
    policy_kwargs = dict(
        net_arch=dict(pi=net, vf=net),
        activation_fn=eval_env.activation_fn,
    )
    # MODEL
    if (eval_env.warm_start and eval_env.prev_ppo_path is not None):
        print("Previous PPO is loaded")
       
        model = PPO.load(eval_env.prev_ppo_path, print_system_info=True,env=train_env, tensorboard_log=str(log_dir),learning_rate=linear_schedule(train_cfg["learning_rate"]))
        
    else:
        print("No previous POO. Training from scratch.")
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=linear_schedule(train_cfg["learning_rate"]),
            n_steps=train_cfg["n_steps"],
            batch_size=train_cfg["batch_size"],
            n_epochs=train_cfg["n_epochs"],
            gamma=train_cfg["gamma"],
            gae_lambda=train_cfg["gae_lambda"],
            clip_range=train_cfg["clip_range"],
            ent_coef=train_cfg["ent_coef"],
            policy_kwargs=policy_kwargs,
            target_kl=train_cfg["target_kl"],
            verbose=1,
            tensorboard_log=str(log_dir),  
            device="cuda",
        )


    
   



    callbacks = make_callbacks(cfg, eval_env, model_dir, log_dir)


    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=callbacks,
    )

    model.save(str(model_dir / "final_model"))

    train_env.close()
    eval_env.close()




if __name__ == "__main__":
    app()