import requests
import json
import gymnasium as gym
from gymnasium import spaces


class TetrisEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.API = "http://127.0.0.1:30010/"
        self.size = size  # The size of the square grid

        # Arguments here for API calls, pre-computed and reusable for efficiency
        self._resetArgs = json.dumps({
            "objectPath" : "/Game/Levels/UEDPIE_0_Start.Start:PersistentLevel.TOTRISGameModeBase_0",
            "functionName" : "Reset",
            "generateTransaction" : False
        })

        self._stepArgs = json.dumps({
            "objectPath" : "/Game/Levels/UEDPIE_0_Start.Start:PersistentLevel.TOTRISGameModeBase_0",
            "functionName" : "GameTick",
            "generateTransaction" : True
        })

        self._moveArgs = {
            0: json.dumps({
            "objectPath" : "/Game/Levels/UEDPIE_0_Start.Start:PersistentLevel.ControllerPawn_0",
            "functionName" : "AIMove",
            "parameters": {"KeyName": self._action_to_move[0]},
            "generateTransaction" : False
            }),
            1: json.dumps({
                "objectPath" : "/Game/Levels/UEDPIE_0_Start.Start:PersistentLevel.ControllerPawn_0",
                "functionName" : "AIMove",
                "parameters": {"KeyName": self._action_to_move[1]},
                "generateTransaction" : False
            }),
            2: json.dumps({
                "objectPath" : "/Game/Levels/UEDPIE_0_Start.Start:PersistentLevel.ControllerPawn_0",
                "functionName" : "AIMove",
                "parameters": {"KeyName": self._action_to_move[2]},
                "generateTransaction" : False
            }),
            3: json.dumps({
                "objectPath" : "/Game/Levels/UEDPIE_0_Start.Start:PersistentLevel.ControllerPawn_0",
                "functionName" : "AIMove",
                "parameters": {"KeyName": self._action_to_move[3]},
                "generateTransaction" : False
            }),
        }
        
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """ The following dictionary maps abstract actions from `self.action_space` to
        the move performed if that action is taken.
        """
        self._action_to_move = {
            0: "up",
            1: "down",
            2: "left",
            3: "right"
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _UE_Call(self, args):
        headers = {"Content-Type": "application/json"}
        return requests.put(self.API+"remote/object/call", data=args, headers=headers)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        return self._UE_Call(self._resetArgs)
    
    def step(self, action):
        self._UE_Call(self._moveArgs[action])
        response = self._UE_Call(self.stepArgs).json()['ReturnValue']

        # observation, reward, terminated, truncated, info
        return response['Observation'], response['Reward'], response['Terminated'], response['Truncated'], response['Info']
        
