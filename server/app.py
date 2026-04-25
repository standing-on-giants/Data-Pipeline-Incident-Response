from __future__ import annotations
import os
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Body
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

    session_id = body.pop("session_id", None) or current_session_id

    if session_id is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        action = PipelineAction(**body)
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
        "done": result.done
    }


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