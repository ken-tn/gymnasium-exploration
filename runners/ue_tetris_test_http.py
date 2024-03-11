import requests
import json
import numpy as np
import gymnasium as gym
import gym_totris
from gymnasium import spaces

env = gym.make("TOTRIS-v0")
print(env.reset())

def main():
    while True:
        observation, reward, terminated, truncated, info = env.step(1)
        # observation = response['Observation']
        # reward = response['Reward']
        # truncated = response['Truncated']
        # info = response['Info']
        if terminated: # terminated
            env.reset()

if __name__ == "__main__":
    main()