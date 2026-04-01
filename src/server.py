"""
OpenEnv-compliant WebSocket server for the Data Pipeline Incident Response environment.

Protocol:
  Client → Server: JSON message with "action" field
    reset:  {"action": "reset", "task_id": "easy|medium|hard"}
    step:   {"action": {"action_type": "...", "params": {...}}}
    state:  {"action": "state"}

  Server → Client: JSON with observation / reward / done / info
"""
from __future__ import annotations
import json
import os
from typing import Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.environment import DataPipelineEnv
from src.models import PipelineAction

app = FastAPI(title="Data Pipeline Incident Response — OpenEnv")


# ------------------------------------------------------------------ #
# Health check
# ------------------------------------------------------------------ #

@app.get("/health")
async def health():
    return {"status": "ok", "env": "data-pipeline-incident-response"}


@app.get("/")
async def root():
    return {
        "name": "Data Pipeline Incident Response",
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
        "openenv_spec": "0.1",
        "websocket_endpoint": "/ws",
    }


# ------------------------------------------------------------------ #
# WebSocket endpoint — one session per connection
# ------------------------------------------------------------------ #

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    env: DataPipelineEnv | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send_error(websocket, "Invalid JSON.")
                continue

            action_field = msg.get("action")

            # ---- RESET ----
            if action_field == "reset":
                task_id = msg.get("task_id", "easy")
                env = DataPipelineEnv(task_id=task_id)
                obs = env.reset()
                await websocket.send_text(json.dumps({
                    "observation": obs.model_dump(),
                    "reward": 0.0,
                    "done": False,
                    "info": {"task_id": task_id},
                }))

            # ---- STATE ----
            elif action_field == "state":
                if env is None:
                    await _send_error(websocket, "Call reset first.")
                    continue
                await websocket.send_text(json.dumps({"state": env.state()}))

            # ---- STEP ----
            elif isinstance(action_field, dict):
                if env is None:
                    await _send_error(websocket, "Call reset first.")
                    continue
                try:
                    action = PipelineAction(**action_field)
                except Exception as e:
                    await _send_error(websocket, f"Invalid action: {e}")
                    continue

                result = env.step(action)
                await websocket.send_text(json.dumps({
                    "observation": result.observation.model_dump(),
                    "reward": result.reward,
                    "done": result.done,
                    "info": result.info,
                }))

            else:
                await _send_error(websocket, f"Unknown action: {action_field!r}")

    except WebSocketDisconnect:
        pass


async def _send_error(ws: WebSocket, msg: str):
    await ws.send_text(json.dumps({"error": msg}))


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("src.server:app", host="0.0.0.0", port=port, workers=1)