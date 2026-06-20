import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
from common.support import _get_geoms_positions, _action_to_ctrl, load_config
import torch


class BaseEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    def __init__(self, config, render_mode=None):

        super().__init__()

        # -------------------------
        # Load config
        # -------------------------
        cfg = load_config(config) if isinstance(config, str) else config
        self.policy = cfg["policy"]
        self.rl_env = cfg["rl_env"]
        self.rl_eval = cfg["rl_evaluation"]

        xml_file = self.rl_env["xml_file"]
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        # -------------------------
        # Policy settings
        # -------------------------
        self.net_arch = self.policy["net_arch"]
        activation_fn = self.policy["activation_fn"]
        if activation_fn == "relu":
            self.activation_fn = torch.nn.ReLU
        elif activation_fn == "tanh":
            self.activation_fn = torch.nn.Tanh
        else:
            raise ValueError("Unsupported activation function")

        # -------------------------
        # Rendering
        # -------------------------
        self.render_mode = render_mode
        self.render_delay = self.rl_eval["render_delay"]
        self.num_frames = self.rl_env["num_frames"]
        self.obs_buffer = deque(maxlen=self.num_frames)

        # -------------------------
        # Marker names
        # -------------------------
        # 24 tentacle + 3 drone markers
        self.marker_names = [f"marker_{i}" for i in range(1, 28)]
        # -------------------------
        # Tentacle effective radius (only 24 tentacle segments)
        # -------------------------
        self.segment_effective_radius = []
        delta = 0.01
        for i in range(1, 25):
            s1 = f"s{i}_1"
            s3 = f"s{i}_3"

            id1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s1)
            id3 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s3)

            y1 = self.model.site_pos[id1][1]
            y3 = self.model.site_pos[id3][1]
            width = abs(y1 - y3)
            radius = width / 2.0
            self.segment_effective_radius.append(radius + delta)
        # -------------------------
        # Simulation timing
        # -------------------------
        self.simulation_length_seconds = self.rl_env["simulation_length_seconds"]
        self.time_between_steps_seconds = self.rl_env["time_between_steps_seconds"]
        self.timestep = self.model.opt.timestep

        self.frame_skip = max(1, round(self.time_between_steps_seconds / self.timestep))
        self.time_per_step = self.frame_skip * self.timestep
        self._max_episode_steps = int(self.simulation_length_seconds / self.time_per_step)

        # -------------------------
        # Action space
        # -------------------------
        self.actuator_dim = self.model.nu
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.actuator_dim,), dtype=np.float32
        )

        # -------------------------
        # Marker positions
        # -------------------------
        all_markers = _get_geoms_positions(self.model, self.data, self.marker_names)[:]
        # Tentacle markers: marker_1 ... marker_24
        self.marker_positions_tentacle = all_markers[:24]

        # Drone markers: marker_25, marker_26, marker_27
        self.marker_positions_drone = all_markers[24:]

        # -------------------------
        # Drone body ID
        # -------------------------
        self.drone_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "x2")
        self.drone_pos = self.data.xpos[self.drone_body_id].copy()

        # -------------------------
        # Goal geometry
        # -------------------------
        site_id = self.model.site('goal').id
        self.goal_position = self.model.geom_pos[site_id].copy()
        self.model.site_pos[site_id] = self.goal_position
        self.delta_goal = self.drone_pos - self.goal_position 
        self.prev_delta_goal = None
        # -------------------------
        # State
        # -------------------------
        self._elapsed_steps = 0
        self.viewer = None
        self.renderer = None

        # -------------------------
        # Observation dimension
        # -------------------------
        # 24 tentacle markers → 24 * 3 = 72
        # 3 drone markers → 3 * 3 = 9
        # goal position → 3
        # IMU: gyro(3) + accel(3) + quat(4) = 10
        # cable lengths → 2
        self.single_frame_obs_dim = (
            72 + 9  + 3 + 10 + 2
        )

        stacked_shape = (self.num_frames * self.single_frame_obs_dim,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=stacked_shape, dtype=np.float32
        )


    # ---------------------------------------------------------
    # Observation 
    # ---------------------------------------------------------
    def _get_current_raw_obs(self):

        # Drone pozíció (globális)
        drone_pos = self.data.xpos[self.drone_body_id].copy()

        # 1) Tentacle marker pozíciók relatív drónhoz
        marker_positions_tentacle_rel = (self.marker_positions_tentacle - drone_pos).flatten()

        # 2) Drone marker pozíciók relatív drónhoz
        marker_positions_drone_rel = (self.marker_positions_drone - drone_pos).flatten()

        # 3) Relatív goal pozíció
        delta_goal_flat = (self.goal_position - drone_pos).flatten()

        # 4) IMU
        gyro = self.data.sensor("body_gyro").data.copy()
        accel = self.data.sensor("body_linacc").data.copy()
        quat = self.data.sensor("body_quat").data.copy()

        # 5) Kábelhosszok
        cable_length = self.data.actuator_length.copy()[-2:]

        # 6) Összefűzés
        obs = np.concatenate([
            marker_positions_tentacle_rel,
            marker_positions_drone_rel,
            delta_goal_flat,
            gyro,
            accel,
            quat,
            cable_length
        ])

        return obs



    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------
    def _base_reset(self):

        mujoco.mj_resetData(self.model, self.data)

        hover = self.model.keyframe('hover')
        self.data.qpos[:] = hover.qpos
        self.data.qvel[:] = hover.qvel
        self.data.ctrl[:] = hover.ctrl
        mujoco.mj_forward(self.model, self.data)
       # -------------------------
        # Marker positions
        # -------------------------
        all_markers = _get_geoms_positions(self.model, self.data, self.marker_names)[:]

        # Tentacle markers: marker_1 ... marker_24
        self.marker_positions_tentacle = all_markers[:24]

        # Drone markers: marker_25, marker_26, marker_27
        self.marker_positions_drone = all_markers[24:]
        self.drone_pos = self.data.xpos[self.drone_body_id].copy()

        random_R=np.random.uniform(0.,0.5)
        random_thetha=np.random.uniform(0,np.pi)
        random_phi=np.random.uniform(0,2*np.pi)
        self.delta_goal=np.array([random_R*np.sin(random_thetha)*np.cos(random_phi),
                                     random_R*np.sin(random_thetha)*np.sin(random_phi),
                                     random_R*np.cos(random_thetha)])

       



        
        self.prev_delta_goal = None
        self._elapsed_steps = 0

        mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------
    # Step
    # ---------------------------------------------------------
    def _base_step(self, action):
        # 1) Action szeparálása
        drone_action = action[:4]          # thrust1..4
        tentacle_action = action[4:]       # act1, act2
        # 2) Drón motorok skálázása (0–13 N)
        drone_ctrl = np.clip(drone_action, -1, 1)
        drone_ctrl = (drone_ctrl + 1) / 2 * 13.0  
        # 3) Tentacle actuator skálázása (kábelhossz)
        act_low = np.array([-1, -1])
        act_high = np.array([1, 1])

        tentacle_ctrl = np.clip(tentacle_action, -1, 1)
        tentacle_ctrl = act_low + (tentacle_ctrl + 1) / 2 * (act_high - act_low)

        # 4) Ctrl vektor összeállítása
        ctrl = np.zeros(self.model.nu)
        ctrl[0:4] = drone_ctrl
        ctrl[4:6] = tentacle_ctrl

        self.data.ctrl[:] = ctrl
       
        self.prev_delta_goal=self.delta_goal
        # 5) Fizikai léptetés
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # 6) Marker pozíciók frissítése
        all_markers = _get_geoms_positions(self.model, self.data, self.marker_names)[:]
        self.marker_positions_tentacle = all_markers[:24]
        self.marker_positions_drone = all_markers[24:27]
        # 7) Drón pozíció frissítése
        self.drone_pos = self.data.xpos[self.drone_body_id].copy()

        # 8) Goal pozíció frissítése
        site_id = self.model.site('goal').id
        self.goal_position=self.model.site_pos[site_id] 
        self.delta_goal = self.drone_pos - self.goal_position 
        # 9) Lépésszámláló
        self._elapsed_steps += 1

        mujoco.mj_forward(self.model, self.data)
        return True


    # ---------------------------------------------------------
    # Render
    # ---------------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self.viewer.is_running():
                self.viewer.sync()
            time.sleep(self.render_delay)
    def _get_info(self):

        return {
                "elapsed_steps": self._elapsed_steps,
            }
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def env_creator(env_config):
    config = load_config(env_config.get("config_path")) if "config_path" in env_config else env_config
    return BaseEnv(config=config)
