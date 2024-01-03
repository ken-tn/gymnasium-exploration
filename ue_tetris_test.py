import json
import requests
import time

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

if __name__ == "__main__":
    for i in range(0, 1000):
        currentTime = time.time()
        response = tick()
        response.json()
        print(response.content)
        print("response took " + str(time.time() - currentTime))