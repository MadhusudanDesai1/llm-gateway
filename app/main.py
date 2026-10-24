from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import router
from .settings import CFG
from .store import summarize

CFG["rpm"] = 0  # throttle is only for free-tier benchmarking

app = FastAPI(title="LLM Gateway")


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    threshold: float | None = Field(default=None, ge=0, le=2)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        return router.handle(req.prompt, req.threshold)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {type(e).__name__}")


@app.get("/stats")
def stats():
    return summarize(router.store.all())
