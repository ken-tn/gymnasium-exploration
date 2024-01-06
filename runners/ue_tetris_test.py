import json
import requests
import time
import random

API = "http://127.0.0.1:30010/"
def UE_Call(args):
    headers = {"Content-Type": "application/json"}
    return requests.put(API+"remote/object/call", data=json.dumps(args), headers=headers)

def tick():
    args = {
        "objectPath" : "/Game/Levels/UEDPIE_0_Start.Start:PersistentLevel.TOTRISGameModeBase_0",
        "functionName" : "GameTick",
        "generateTransaction" : True
    }

    return UE_Call(args)

def reset():
    args = {
        "objectPath" : "/Game/Levels/UEDPIE_0_Start.Start:PersistentLevel.TOTRISGameModeBase_0",
        "functionName" : "Reset",
        "generateTransaction" : False
    }

    return UE_Call(args)

def move(key):
    args = {
        "objectPath" : "/Game/Levels/UEDPIE_0_Start.Start:PersistentLevel.ControllerPawn_0",
        "functionName" : "AIMove",
        "parameters": {"KeyName": key},
        "generateTransaction" : False
    }

    return UE_Call(args)

if __name__ == "__main__":
    terminated = False
    while True:
        move(random.choice(action_space))
        response = tick()
        response = response.json()['ReturnValue']
        # observation = response['Observation']
        # reward = response['Reward']
        terminated = response['Terminated']
        # truncated = response['Truncated']
        # info = response['Info']
        if terminated: # terminated
            reset()
        #print("response took " + str(time.time() - currentTime))