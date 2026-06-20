import mujoco
import numpy as np
import matplotlib.pyplot as plt
import yaml
def _action_to_ctrl(action,actuator_low,actuator_high):
        return actuator_low + (action + 1.0) * 0.5 * (
            actuator_high - actuator_low
        )
def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

from scipy.spatial.transform import Rotation as R

def get_rpy_from_quat(quat):
    # MuJoCo: [w, x, y, z] → SciPy: [x, y, z, w]
    q = np.array([quat[1], quat[2], quat[3], quat[0]])

    # Euler szögek (roll, pitch, yaw)
    roll, pitch, yaw = R.from_quat(q).as_euler('xyz', degrees=False)

    return roll, pitch, yaw

def _get_geoms_positions(model, data, geom_names) -> np.ndarray:
    """
    Returns:
        (N, 3) array of geom positions (x,y,z)
    """
    positions = []
    geom_names = [geom_names] if isinstance(geom_names, str) else geom_names

    for name in geom_names:
        geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name
        )
        positions.append(data.geom_xpos[geom_id].copy())

    return np.array(positions)





def sample_target(workspace_center,
                  workspace_inner_radius,
                  workspace_outer_radius):

    r = np.sqrt(
        np.random.uniform(
            workspace_inner_radius**2,
            workspace_outer_radius**2
        )
    )

    theta = np.random.uniform(np.pi+np.pi/18, 2*np.pi-np.pi/18)  # alsó félkör

    x = workspace_center[0] + r * np.cos(theta)
    y = workspace_center[1] + r * np.sin(theta)

    return np.array([x, y])


def _normalize_position(positions,workspace_center,workspace_scale) -> np.ndarray:
    positions = np.asarray(positions)


    return (positions - workspace_center[None ,:]) / workspace_scale[None, :]

def _normalize_actuator_lengths(lengths,actuator_low,actuator_high) -> np.ndarray:
    lengths = np.asarray(lengths)
    actuator_low = np.asarray(actuator_low)
    actuator_high = np.asarray(actuator_high)
    return (
            2.0
            * (lengths - actuator_low)
            / (actuator_high - actuator_low)
            - 1.0
        )

