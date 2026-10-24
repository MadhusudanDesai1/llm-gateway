from __future__ import annotations

import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import router
from .settings import CFG
from .store import summarize

CFG["rpm"] = 0  # throttle is only for free-tier benchmarking

app = FastAPI(title="LLM Gateway")


def require_key(x_api_key: str | None = Header(default=None)):
    """If GATEWAY_API_KEY is set, /chat and /stats require a matching x-api-key header."""
    expected = os.getenv("GATEWAY_API_KEY")
    if expected and not (x_api_key and hmac.compare_digest(x_api_key.encode(), expected.encode())):
        raise HTTPException(status_code=401, detail="invalid or missing x-api-key")


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    threshold: float | None = Field(default=None, ge=0, le=2)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", dependencies=[Depends(require_key)])
def chat(req: ChatRequest):
    try:
        return router.handle(req.prompt, req.threshold)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {type(e).__name__}")


@app.get("/stats", dependencies=[Depends(require_key)])
def stats():
    return summarize(router.store.all())
