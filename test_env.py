from mixing_env import MixEnv

env = MixEnv()

# episodes = 5

# for episode in range(episodes):
terminated, truncated = False, False
obs, _ = env.reset()
while (not terminated) and (not truncated):
    # for i in range(500):
    # i += 1
    random_action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(random_action)

env.visualize(["t", "volume", "ca", "rtime", "qs", "qa"])
