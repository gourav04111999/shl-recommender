"""
SHL Assessment Recommender - FastAPI Service
Stateless conversational agent that guides hiring managers to relevant SHL assessments.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import List

from agent import run_agent, AgentError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models (schema is non-negotiable per assignment spec)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    messages: List[Message]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("messages list cannot be empty")
        # Enforce the 8-turn cap from the assignment spec
        if len(v) > 8:
            raise ValueError("conversation exceeds the 8-turn cap")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SHL Recommender starting up — catalog loaded.")
    yield
    logger.info("SHL Recommender shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL Individual Test Solution selection",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Global exception handler — always return valid JSON
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "reply": "I encountered an unexpected error. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Readiness probe. Returns 200 immediately after startup."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Stateless conversational endpoint.

    The caller must send the full conversation history on every call.
    The service holds no per-conversation state.
    """
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    try:
        result = run_agent(messages)
    except AgentError as exc:
        logger.warning(f"AgentError: {exc}")
        return ChatResponse(
            reply=str(exc),
            recommendations=[],
            end_of_conversation=False,
        )
    return ChatResponse(**result)
