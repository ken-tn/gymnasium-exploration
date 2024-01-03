import gymnasium as gym
env = gym.make("ALE/Tetris-v5", render_mode="human")

observation, info = env.reset(seed=42)
for steps in range(1000):
    print(steps)
    action = env.action_space.sample()
    #print(action)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
