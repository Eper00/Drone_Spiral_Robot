"""Hover Aviary: single-drone hover task for RL training.

Task: hover at z=1.0 and remain stable.
Observation: 12-dim [pos(3), rpy(3), vel(3), ang_vel(3)]  (normalized)
Action: 4-dim normalized RPMs [-1, 1]
"""
import os
import numpy as np
from gymnasium import spaces
from pathlib import Path
import mujoco

from multi_drone_mujoco.envs.base_aviary import BaseAviary,DRONE_PARAMS
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType
ASSETS_PATH = Path(__file__).resolve().parent.parent / "assets"
CF2_MESH_DIR = Path(os.environ.get("MJ_DRONES_CF2_ASSETS",
                                   str(ASSETS_PATH / "cf2")))
def _generate_aviary_xml(
    num_drones: int,
    drone_model: DroneModel,
    init_xyzs: np.ndarray,
    init_rpys: np.ndarray,
    obstacles: bool = False,
    vision: bool = False,
    timestep: float = 1 / 240,
) -> str:
    """Generate MuJoCo XML for the aviary with N drones."""
    meshdir = str(CF2_MESH_DIR)
    tentacle_mesh=str(ASSETS_PATH)
    params = DRONE_PARAMS[drone_model]
    # Visual and collision meshes
    visual_meshes = "\n".join(
        f'    <mesh file="{meshdir}/cf2_{i}.obj" name="cf2_vis_{i}"/>'
        for i in range(7)
    )
    collision_meshes = "\n".join(
        f'    <mesh file="{meshdir}/cf2_collision_{i}.obj" name="cf2_col_{i}"/>'
        for i in range(32)
    )

    # Drone bodies with 4 propeller sites for force application
    mass = params["mass"]
    ixx, iyy, izz = params["ixx"], params["iyy"], params["izz"]
    L = params["arm_length"]

    drone_bodies = ""
    actuators = ""
    sensors = ""
    for d in range(num_drones):
        x, y, z = init_xyzs[d]
        # Convert RPY to quaternion for initial orientation
        r, p_angle, yaw = init_rpys[d]
        # Simple RPY to quat (small angles)
        cr, sr = np.cos(r / 2), np.sin(r / 2)
        cp, sp = np.cos(p_angle / 2), np.sin(p_angle / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        prefix = f"drone{d}"

        # Propeller positions (X-configuration for CF2X)
        if drone_model == DroneModel.CF2X:
            prop_offsets = [
                (L / np.sqrt(2), L / np.sqrt(2), 0),    # front-left
                (-L / np.sqrt(2), L / np.sqrt(2), 0),   # front-right (corrected)
                (-L / np.sqrt(2), -L / np.sqrt(2), 0),  # rear-right
                (L / np.sqrt(2), -L / np.sqrt(2), 0),   # rear-left
            ]
        elif drone_model == DroneModel.CF2P:
            prop_offsets = [
                (L, 0, 0),     # front
                (0, L, 0),     # left
                (-L, 0, 0),    # rear
                (0, -L, 0),    # right
            ]
        else:  # RACE (same as X)
            prop_offsets = [
                (L / np.sqrt(2), L / np.sqrt(2), 0),
                (-L / np.sqrt(2), L / np.sqrt(2), 0),
                (-L / np.sqrt(2), -L / np.sqrt(2), 0),
                (L / np.sqrt(2), -L / np.sqrt(2), 0),
            ]

        prop_sites = ""
        for pi, (px, py, pz) in enumerate(prop_offsets):
            prop_sites += f'      <site name="{prefix}_prop{pi}" pos="{px} {py} {pz}" group="5"/>\n'

        drone_bodies += f"""
    <body name="{prefix}" pos="{x} {y} {z}" quat="{qw} {qx} {qy} {qz}">
      <freejoint name="{prefix}_joint"/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx} {iyy} {izz}"/>
      <geom name="{prefix}_collision" type="cylinder" size="{params['collision_r']} {params['collision_h'] / 2}" rgba="0 0 0 0" contype="1" conaffinity="1"/>
      <geom mesh="cf2_vis_0" material="propeller_plastic" class="visual"/>
      <geom mesh="cf2_vis_1" material="medium_gloss_plastic" class="visual"/>
      <geom mesh="cf2_vis_2" material="polished_gold" class="visual"/>
      <geom mesh="cf2_vis_3" material="polished_plastic" class="visual"/>
      <geom mesh="cf2_vis_4" material="burnished_chrome" class="visual"/>
      <geom mesh="cf2_vis_5" material="body_frame_plastic" class="visual"/>
      <geom mesh="cf2_vis_6" material="white" class="visual"/>
      <site name="{prefix}_center" pos="0 0 0" group="5"/>
        <body name="hook" pos="0 0 -0.07">

        <inertial pos="0 0 0" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>

        <joint name="hook_joint" type="hinge" axis="1 0 0" range="-1.2 0"/>

        <geom type="capsule"
          fromto="0 0 0   0 0 -0.03"
          size="0.004"
          rgba="0.8 0.2 0.2 1"
          contype="1" conaffinity="1"/>

    <geom type="capsule"
          fromto="0 0 -0.03   0.03 0 -0.03"
          size="0.004"
          rgba="0.8 0.2 0.2 1"
          contype="1" conaffinity="1"/>

    <geom type="capsule"
          fromto="0.03 0 -0.03   0.03 0 0.06"
          size="0.004"
          rgba="0.8 0.2 0.2 1"
          contype="1" conaffinity="1"/>

</body>



{prop_sites}
"""

        # Add camera for vision
        if vision:
            drone_bodies += f'      <camera name="{prefix}_cam" pos="0.02 0 0" xyaxes="0 -1 0 0 0 1" fovy="60"/>\n'

        drone_bodies += "    </body>\n"

        # Sensors
        sensors += f"""
    <gyro name="{prefix}_gyro" site="{prefix}_center"/>
    <accelerometer name="{prefix}_acc" site="{prefix}_center"/>
    <framequat name="{prefix}_quat" objtype="site" objname="{prefix}_center"/>
    <framepos name="{prefix}_pos" objtype="site" objname="{prefix}_center"/>
    <framelinvel name="{prefix}_vel" objtype="site" objname="{prefix}_center"/>
    <frameangvel name="{prefix}_angvel" objtype="site" objname="{prefix}_center"/>"""

    # Obstacles
    obstacle_bodies = ""
    if obstacles:
        obstacle_bodies = """
    <body name="obstacle_box" pos="0.5 0.5 0.3">
      <geom type="box" size="0.1 0.1 0.3" rgba="0.8 0.2 0.2 1"/>
    </body>
    <body name="obstacle_sphere" pos="-0.5 0.5 0.5">
      <geom type="sphere" size="0.1" rgba="0.2 0.8 0.2 1"/>
    </body>
    <body name="obstacle_cylinder" pos="0 -0.5 0.4">
      <geom type="cylinder" size="0.05 0.3" rgba="0.2 0.2 0.8 1"/>
    </body>"""

    xml = f"""<mujoco model="aviary_{num_drones}x_{drone_model.value}">
  <option integrator="RK4" density="1.225" viscosity="1.8e-5" timestep="{timestep}"/>
  <compiler inertiafromgeom="false" autolimits="true"/>

  <default>
    <default class="cf2">
      <default class="visual">
        <geom group="2" type="mesh" contype="0" conaffinity="0"/>
      </default>
    </default>
  </default>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-20" elevation="-20"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
  
    <material name="polished_plastic" rgba="0.631 0.659 0.678 1"/>
    <material name="polished_gold" rgba="0.969 0.878 0.6 1"/>
    <material name="medium_gloss_plastic" rgba="0.109 0.184 0.0 1"/>
    <material name="propeller_plastic" rgba="0.792 0.820 0.933 1"/>
    <material name="white" rgba="1 1 1 1"/>
    <material name="body_frame_plastic" rgba="0.102 0.102 0.102 1"/>
    <material name="burnished_chrome" rgba="0.898 0.898 0.898 1"/>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    
{visual_meshes}
{collision_meshes}
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true" castshadow="false"/>
    <geom name="floor" size="10 10 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
{drone_bodies}{obstacle_bodies}
  </worldbody>

  <sensor>
{sensors}
  </sensor>
</mujoco>
"""
    return xml


class TentacleAviary(BaseAviary):
    """Single drone hover task — matches gym-pybullet-drones HoverAviary."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.BB,
        physics: Physics = Physics.MJC,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        obstacles: bool = False,
        target_height: float = 1.0,
        initial_xyzs=None,
        render_mode=None,
    ):
        self.TARGET_HEIGHT = target_height
        self.EPISODE_LEN_SEC = 10
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 0.1]])

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            physics=physics,
            sim_freq=sim_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=obstacles,
            obs_type=ObservationType.KIN,
            act_type=ActionType.RPM,
            initial_xyzs=initial_xyzs,
            render_mode=render_mode,
        )
        xml_str = _generate_aviary_xml(
            num_drones=1,
            drone_model=drone_model,
            init_xyzs=self.INIT_XYZS,
            init_rpys=self.INIT_RPYS,
            obstacles=obstacles,
            vision=False,
            timestep=self.SIM_TIMESTEP,
        )
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)

    def _actionSpace(self):
        """Normalized [-1, 1] → mapped to RPM internally."""
        return spaces.Box(low=-np.ones(4, dtype=np.float32), high=np.ones(4, dtype=np.float32))

    def _observationSpace(self):
        """12-dim observation: pos, rpy, vel, ang_vel."""
        return spaces.Box(
            low=-np.inf * np.ones(12, dtype=np.float32),
            high=np.inf * np.ones(12, dtype=np.float32),
        )

    def _preprocessAction(self, action):
        """Convert normalized action to RPMs."""
        action = np.clip(np.array(action).flatten(), -1, 1)
        rpms = self._normalizedActionToRPM(action).reshape(1, 4)
        return rpms

    def _computeObs(self):
        """12-dim observation."""
        state = self._getDroneStateVector(0)
        # pos(3), rpy(3), vel(3), ang_vel(3)
        obs = np.hstack([state[0:3], state[7:10], state[10:13], state[13:16]])
        return obs.astype(np.float32)

    def _computeReward(self):
        """Dense reward: penalize distance to target height and attitude."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        rpy = state[7:10]

        # Reward for being at target height
        height_error = abs(pos[2] - self.TARGET_HEIGHT)
        xy_error = np.linalg.norm(pos[0:2])

        reward = -height_error - 0.1 * xy_error
        reward -= 0.05 * np.linalg.norm(vel)
        reward -= 0.05 * (abs(rpy[0]) + abs(rpy[1]))

        # Bonus for being close
        if height_error < 0.05 and xy_error < 0.05:
            reward += 1.0

        if self._computeTerminated():
            reward -= 100.0

        return float(reward)

    def _computeTerminated(self):
        """Terminate if drone crashes or flips."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]

        if pos[2] < 0.0:
            return True
        if abs(rpy[0]) > np.pi / 2 or abs(rpy[1]) > np.pi / 2:
            return True
        if pos[2] > 3.0:
            return True
        return False

    def _computeTruncated(self):
        """Truncate after episode time limit."""
        return self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC

    def _computeInfo(self):
        return {
            "position": self.pos[0].tolist(),
            "height_error": abs(self.pos[0, 2] - self.TARGET_HEIGHT),
        }
