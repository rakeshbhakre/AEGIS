"""WebSocket telemetry and status streaming."""
from __future__ import annotations

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect


def register_websocket(app: FastAPI, engine_getter, state: dict, tick: dict) -> None:
    @app.websocket("/ws")
    async def ws_ep(ws: WebSocket):
        await ws.accept()
        while engine_getter() is None:
            await ws.send_json({"type": "status", "stats": {"warming": True, "note": "engine calibrating…"}})
            await asyncio.sleep(1)
        engine = engine_getter()
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        engine.subscribers.append(queue)
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=0.5)
                    await ws.send_json(msg)
                except asyncio.TimeoutError:
                    stats = engine.stats() | {
                        "warm": state["warm"], "k": tick["n"],
                        "mode": state["mode"], "speed": state["speed"],
                    }
                    await ws.send_json({"type": "status", "stats": stats})
        except WebSocketDisconnect:
            pass
        finally:
            if queue in engine.subscribers:
                engine.subscribers.remove(queue)
