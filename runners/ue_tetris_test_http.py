import requests
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TotrisEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, gameModeObjectPath="/Game/Levels/UEDPIE_0_Start.Start:PersistentLevel.TOTRISGameModeBP_C_0", pawnObjectPath="/Game/Levels/UEDPIE_0_Start.Start:PersistentLevel.ControllerPawnBP_C_1"):
        self.API = "http://127.0.0.1:30010/"
        self.gamemodeObjectPath = gameModeObjectPath
        self.pawnObjectPath = pawnObjectPath
        self._call = self.API+"remote/object/call"
        self._headers = {"Content-Type": "application/json"}

        # Arguments here for API calls, pre-computed and reusable for efficiency
        self._resetArgs = json.dumps({
            "objectPath" : self.gamemodeObjectPath,
            "functionName" : "Reset",
            "generateTransaction" : False
        })

        self._stepArgs = json.dumps({
            "objectPath" : self.gamemodeObjectPath,
            "functionName" : "GameTick",
            "generateTransaction" : False
        })

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """ The following dictionary maps abstract actions from `self.action_space` to
        the move performed if that action is taken.
        """
        self._action_to_move = {
            0: "TriggerRotate",
            1: "TriggerDown",
            2: "TriggerLeft",
            3: "TriggerRight"
        }

        self._moveArgs = {
            0: json.dumps({
            "objectPath" : self.pawnObjectPath,
            "functionName" : self._action_to_move[0],
            "generateTransaction" : False
            }),
            1: json.dumps({
                "objectPath" : self.pawnObjectPath,
                "functionName" : self._action_to_move[1],
                "generateTransaction" : False
            }),
            2: json.dumps({
                "objectPath" : self.pawnObjectPath,
                "functionName" : self._action_to_move[2],
                "generateTransaction" : False
            }),
            3: json.dumps({
                "objectPath" : self.pawnObjectPath,
                "functionName" : self._action_to_move[3],
                "generateTransaction" : False
            }),
        }
        
        # Observations are dictionaries with the board's state.
        self.observation_space = spaces.Dict(
            {
                # low: 0, high: 1, 1d of length 200 (board is 10x20)
                "board": spaces.Box(0, 1, shape=(200,), dtype=int),
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _UE_Call(self, args):
        return requests.put(self._call, data=args, headers=self._headers)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)
        self._UE_Call(self._resetArgs)

        return {"board": np.zeros(200, dtype=int)}, {}
    
    def step(self, action):
        self._UE_Call(self._moveArgs[action])
        response = self._UE_Call(self._stepArgs).json()['ReturnValue']

        # observation, reward, terminated, truncated, info
        return {"board": np.array(response['Observation'])}, response['Reward'], response['Terminated'], response['Truncated'], {"state": response['Info']}
        
def main():
    env = TotrisEnv()
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