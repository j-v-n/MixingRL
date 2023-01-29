import sys
import os
import gymnasium

sys.modules["gym"] = gymnasium

from stable_baselines3 import PPO
from mixing_env import MixEnv

models_dir = "mixing_demo/models/PPO"
logdir = "mixing_demo/logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = MixEnv()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 5000

for i in range(1, 1000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()
