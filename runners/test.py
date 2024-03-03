import asyncio
import websockets
import json

msg = {
    "MessageName": "http",
    "Parameters": {
        "Url": "/remote/object/call",
        "Verb": "PUT",
        "Body": {
            "objectPath": "/Game/Levels/UEDPIE_0_Start.Start:PersistentLevel.TOTRISGameModeBP_C_0",
            "functionName": "GameTick",
            "generateTransaction": False
        }
    }
}

async def hello():
    uri = "ws://127.0.0.1:30020"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(msg))

        greeting = await websocket.recv()
        print(f"<<< {greeting}")

if __name__ == "__main__":
    asyncio.run(hello())