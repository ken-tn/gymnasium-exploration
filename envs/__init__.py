from gymnasium.envs.registration import register

register(
    id="gym/TOTRIS-v0",
    entry_point="gym.envs:TetrisEnv",
    max_episode_steps=300,
)