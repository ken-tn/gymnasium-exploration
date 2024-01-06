import gym_totris
import gymnasium as gym
env = gym.make("TOTRIS-v0")

observation, info = env.reset()
for steps in range(1000):
    print(steps)
    action = env.action_space.sample()
    #print(action)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
