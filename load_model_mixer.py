import sys
import os
import gymnasium

sys.modules["gym"] = gymnasium

from stable_baselines3 import PPO
from mixing_env import MixEnv

models_dir = "mixing_demo/models/PPO"

env = MixEnv()
env.reset()

model_path = f"{models_dir}/135000.zip"

model = PPO.load(model_path, env=env)

episodes = 1

for _ in range(episodes):
    obs, info = env.reset()
    terminated, truncated = False, False

    while not terminated and not truncated:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        # env.render()
        print(rewards)

env.visualize(["t", "volume", "ca", "rtime", "qs", "qa"])
