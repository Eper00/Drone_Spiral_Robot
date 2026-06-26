from stable_baselines3 import PPO
from multi_drone_mujoco.envs.hover_aviary import HoverAviary

env = HoverAviary(ctrl_freq=48)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
model.save("hover_ppo")