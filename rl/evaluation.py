"""Clean RL evaluation script (CLI version, same style as training)."""

from pathlib import Path
from typing import Optional

import typer
import numpy as np
import torch
from stable_baselines3 import PPO
from rich.console import Console

from rl.hover_env import DroneHoverEnv
from common.support import load_config

console = Console()
app = typer.Typer()


# ----------------------------
# MODEL LOADER (BC + PPO unified)
# ----------------------------
def load_policy(model_path: Path, env, verbose=True):

    

    if verbose:
        console.print("[green]Loading PPO model[/green]")

    return PPO.load(str(model_path), env=env)


# ----------------------------
# EVALUATE COMMAND
# ----------------------------
@app.command()
def evaluate(
    model_path: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    num_episodes: Optional[int] = typer.Option(None),
    render: bool = typer.Option(True),
    deterministic: Optional[bool] = typer.Option(None),
    render_delay: Optional[float] = typer.Option(None),
    save_results: bool = typer.Option(False),
    verbose: bool = typer.Option(True),
):

    # --------------------
    # CONFIG
    # --------------------
    cfg = load_config(config)
    rl_eval = cfg["rl_evaluation"]

    if num_episodes is not None:
        rl_eval["num_episodes"] = num_episodes
    if deterministic is not None:
        rl_eval["deterministic_actions"] = deterministic
    if render_delay is not None:
        rl_eval["render_delay"] = render_delay

    rl_eval["render_mode"] = "human" if render else None

    # --------------------
    # ENV
    # --------------------
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    env = DroneHoverEnv(
        cfg,
        render_mode=rl_eval["render_mode"]
    )

    model = load_policy(model_path, env, verbose=verbose)

    # --------------------
    # LOG
    # --------------------
    if verbose:
        console.print(f"Episodes: {rl_eval['num_episodes']}")
        console.print(f"Render: {rl_eval['render_mode']}")
        console.print(f"Deterministic: {rl_eval['deterministic_actions']}")

    # --------------------
    # METRICS
    # --------------------
    rewards, lengths, distances = [], [], []
    success = 0
    threshold = 0.5

    try:
        for ep in range(rl_eval["num_episodes"]):

            obs,_ = env.reset()
            done = False

            ep_reward = 0.0
            ep_len = 0

            while not done:

                action, _ = model.predict(
                    obs,
                    deterministic=rl_eval["deterministic_actions"],
                )

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_reward += reward
                ep_len += 1

                if rl_eval["render_mode"] == "human":
                    env.render()

            final_dist = info.get("distance_to_target", float("inf"))

            rewards.append(ep_reward)
            lengths.append(ep_len)
            distances.append(final_dist)

            if final_dist <= threshold:
                success += 1

            if verbose:
                console.print(f"Ep {ep}: R={ep_reward:.3f}, L={ep_len}")

    finally:
        env.close()

    # --------------------
    # RESULTS
    # --------------------
    console.print("\n=== RESULTS ===")
    console.print(f"Episodes: {len(rewards)}")
    console.print(f"Reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    console.print(f"Length: {np.mean(lengths):.1f}")
    console.print(f"Distance: {np.mean(distances):.4f}")
    console.print(f"Success: {100 * success / len(rewards):.1f}%")

    if save_results:
        out = model_path.parent / f"eval_{model_path.stem}.txt"
        out.write_text(
            f"Reward mean: {np.mean(rewards)}\n"
            f"Length mean: {np.mean(lengths)}\n"
            f"Distance mean: {np.mean(distances)}\n"
        )
        console.print(f"Saved: {out}")


# ----------------------------
# ENTRYPOINT
# ----------------------------
if __name__ == "__main__":
    app()