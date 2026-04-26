from __future__ import annotations
import os
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.environment import DataPipelineEnv
from src.models import PipelineAction


# ---------------- LOGGING ---------------- #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openenv")


# ---------------- APP ---------------- #

app = FastAPI(
    title="Data Pipeline Incident Response — OpenEnv",
    description="AI agent environment for debugging data pipelines.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- SESSION STORE ---------------- #

_sessions: Dict[str, DataPipelineEnv] = {}
current_session_id: Optional[str] = None


# ---------------- REQUEST ---------------- #

class ResetRequest(BaseModel):
    task_id: str = "easy"


# ---------------- ROOT ---------------- #

@app.get("/")
def root():
    return {"status": "ok", "env": "data-pipeline", "version": "1.1.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------------- RESET ---------------- #

@app.post("/reset")
def reset(req: Optional[ResetRequest] = Body(default=None)):
    global current_session_id

    task_id = req.task_id if req else "easy"

    env = DataPipelineEnv(task_id=task_id)
    session_id = f"{task_id}_{id(env)}"

    _sessions[session_id] = env
    current_session_id = session_id

    obs, _ = env.reset()

    logger.info(f"[RESET] task={task_id} session={session_id}")

    return {
        "session_id": session_id,
        "observation": obs.model_dump()
    }


# ---------------- STEP ---------------- #

@app.post("/step")
def step(body: Dict[str, Any] = Body(...)):
    global current_session_id

    payload = dict(body)
    session_id = payload.pop("session_id", None) or current_session_id

    if session_id is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        # Accept both shapes:
        # 1) {"action_type": "...", "params": {...}}
        # 2) {"action": {"action_type": "...", "params": {...}}}
        action_payload = payload.get("action") if isinstance(payload.get("action"), dict) else payload
        action = PipelineAction(**action_payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = env.step(action)

    logger.info(
        f"[STEP] session={session_id} action={action.action_type} "
        f"reward={result.reward} done={result.done}"
    )

    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "terminated": result.terminated,
        "truncated": result.truncated,
    }


# ---------------- WEBSOCKET ---------------- #

@app.websocket("/ws")
async def websocket_env(websocket: WebSocket):
    global current_session_id

    await websocket.accept()
    logger.info("[WS] client connected")

    try:
        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")

            # reset path: {"action":"reset", "task_id":"easy"}
            if action == "reset":
                task_id = msg.get("task_id", "easy")
                env = DataPipelineEnv(task_id=task_id)
                session_id = f"{task_id}_{id(env)}"
                _sessions[session_id] = env
                current_session_id = session_id
                obs, _ = env.reset()

                await websocket.send_json({
                    "session_id": session_id,
                    "observation": obs.model_dump(),
                    "reward": 0.0,
                    "done": False,
                    "terminated": False,
                    "truncated": False,
                    "info": {"message": "Environment reset"},
                })
                continue

            # state path: {"action":"state", "session_id":"..."}
            if action == "state":
                session_id = msg.get("session_id") or current_session_id
                if session_id is None or session_id not in _sessions:
                    await websocket.send_json({"error": "Call reset first"})
                    continue
                await websocket.send_json(_sessions[session_id].state())
                continue

            # step path:
            # {"action": {"action_type":"...", "params": {...}}, "session_id":"..."}
            session_id = msg.get("session_id") or current_session_id
            if session_id is None or session_id not in _sessions:
                await websocket.send_json({"error": "Call reset first"})
                continue

            env = _sessions[session_id]

            try:
                action_payload = action if isinstance(action, dict) else msg
                action_obj = PipelineAction(**action_payload)
            except Exception as exc:
                await websocket.send_json({"error": f"Invalid action payload: {exc}"})
                continue

            result = env.step(action_obj)
            await websocket.send_json({
                "session_id": session_id,
                "observation": result.observation.model_dump(),
                "reward": result.reward,
                "done": result.done,
                "terminated": result.terminated,
                "truncated": result.truncated,
                "info": result.info,
            })

    except WebSocketDisconnect:
        logger.info("[WS] client disconnected")
    except Exception as exc:
        logger.exception(f"[WS] unexpected server error: {exc}")
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass


# ---------------- STATE ---------------- #

@app.get("/state")
def state():
    global current_session_id

    if current_session_id is None:
        return {"status": "not_initialized"}

    return _sessions[current_session_id].state()


# ---------------- TASKS ---------------- #

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"task_id": "easy", "difficulty": "easy"},
            {"task_id": "medium", "difficulty": "medium"},
            {"task_id": "hard", "difficulty": "hard"},
            {"task_id": "hard2", "difficulty": "hard"},
        ]
    }


# ---------------- ENTRY ---------------- #

def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()