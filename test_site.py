"""
Simple MuJoCo GUI tester for TentacleAviary.

Usage:
    python test_site.py --env tentacle
    python test_site.py --env hover
    python test_site.py --env multi
"""

import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np


def make_env(env_name: str):
    from multi_drone_mujoco.envs.hover_aviary import HoverAviary
    from multi_drone_mujoco.envs.multi_hover_aviary import MultiHoverAviary
    from multi_drone_mujoco.envs.adaptive_hook_hover import TentacleAviary

    if env_name == "hover":
        return HoverAviary(ctrl_freq=48, sim_freq=240, render_mode=None)
    elif env_name == "multi":
        return MultiHoverAviary(num_drones=2, ctrl_freq=48, sim_freq=240, render_mode=None)
    else:
        return TentacleAviary(ctrl_freq=48, sim_freq=240, render_mode=None)


def run_gui(env_name: str):
    env = make_env(env_name)
    obs, info = env.reset()

    model = env.model
    data = env.data

    print("MuJoCo GUI running. Close the window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # step physics
            action=np.random.uniform(-1,1,6)
            env.step(action)
            # update GUI
            viewer.sync()
            # slow down to real time
            time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="tentacle",
                        help="tentacle | hover | multi")
    args = parser.parse_args()

    run_gui(args.env)
