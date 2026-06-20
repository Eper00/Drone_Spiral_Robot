"""MuJoCo-based reinforcement learning environment for drone + tentacle."""

import numpy as np
from typing import Dict, Any
import os
from collections import deque
from common.support import load_config,get_rpy_from_quat
from common.base_env import BaseEnv


class DroneHoverEnv(BaseEnv):
    def __init__(self, config: Dict[str, Any] = None, render_mode: str = None):

        super().__init__(config, render_mode)

        # Warm start (PPO load)
        self.warm_start = self.rl_env["warm_start"]
        if self.warm_start and os.path.exists(self.rl_env["prev_ppo_path"]):
            self.prev_ppo_path = self.rl_env["prev_ppo_path"]
        else:
            self.prev_ppo_path = None

    # ---------------------------------------------------------
    # Observation stacking
    # ---------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        assert len(self.obs_buffer) == self.num_frames, "Observation buffer not full!"
        return np.concatenate(list(self.obs_buffer), axis=0).astype(np.float32)

    # ---------------------------------------------------------
    # Step
    # ---------------------------------------------------------
    def step(self, action):
       
        self._base_step(action)
        action[-2:0]=0
        obs = self._get_current_raw_obs()
        reward = self.reward_function(action)

        truncated = self._elapsed_steps >= self._max_episode_steps
        
        self.obs_buffer.append(obs)
        rxy = get_rpy_from_quat(self.data.sensor("body_quat").data)
        terminated=False
        
        return (
            self._get_obs(),
            float(reward),
            terminated,          # terminated
            truncated,      # truncated
            self._get_info()
        )

        
        
    def reward_function(self, action):

        # 1) Távolság
        
        dist_z = -np.abs(self.delta_goal[2])
        dist_x_y= -np.sqrt(self.delta_goal[0]**2+self.delta_goal[1]**2)
        

        # 2) Cél elérése
        goal_reward = 0.0
        if np.abs(dist_z) < 0.05:
            goal_reward = 5.0

        # 3) Stabilitás csak a cél közelében
        rxy = get_rpy_from_quat(self.data.sensor("body_quat").data)
        stability_penalty =  -(np.abs(rxy[0])+np.abs(rxy[1]))
        # 4) Összesítés
        if self.drone_pos[2]<0.0 or self.drone_pos[2]>10 or max(np.abs(rxy[0]),np.abs(rxy[1]))>np.pi/2:
            termination_penalty=-10
        else:
            termination_penalty=0
        reward = dist_z + 0.1*dist_x_y + goal_reward + 0.05*stability_penalty+termination_penalty

        return reward


    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._base_reset()
        self.delta_goal=[0,0,1.5]
        self.drone_pos = self.data.xpos[self.drone_body_id].copy()
        self.goal_position = self.drone_pos.copy()+self.delta_goal
        site_id = self.model.site('goal').id
        self.model.site_pos[site_id] = self.goal_position
        self.obs_buffer.clear()
        raw = self._get_current_raw_obs()

        for _ in range(self.num_frames):
            self.obs_buffer.append(raw.copy())

        return self._get_obs(), self._get_info()


# ---------------------------------------------------------
# RLlib creator
# ---------------------------------------------------------
def env_creator(env_config: Dict[str, Any]) -> DroneHoverEnv:
    config = load_config(env_config.get("config_path")) if "config_path" in env_config else env_config
    render_mode = env_config.get("render_mode", None)
    return DroneHoverEnv(config=config, render_mode=render_mode)
