from stable_baselines3.common.env_checker import check_env
from mixing_env import MixEnv

env = MixEnv()
check_env(env)
