"""A2A server factory — exposes specialist agents via JSON-RPC over HTTP."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rip.models.messages import (
    A2AMessage,
    A2ATask,
    AgentCard,
    JSONRPCRequest,
    JSONRPCResponse,
    MessagePart,
    TaskState,
)

TaskHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
StartupHook = Callable[[], Awaitable[None]]


def create_a2a_app(
    agent_card: AgentCard,
    handler: TaskHandler,
    on_startup: StartupHook | None = None,
) -> FastAPI:
    """Create a FastAPI app that serves an A2A-compliant agent endpoint."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if on_startup:
            await on_startup()
        yield

    app = FastAPI(title=agent_card.name, version=agent_card.version, lifespan=lifespan)

    @app.get("/.well-known/agent.json")
    async def well_known() -> AgentCard:
        return agent_card

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "message": f"This is the {agent_card.name} API — not the web UI.",
            "ui": "Open http://localhost:8000 in your browser",
            "health": "/health",
            "agent_card": "/.well-known/agent.json",
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "agent": agent_card.name}

    @app.post("/a2a")
    async def a2a_endpoint(request: Request) -> JSONResponse:
        body = await request.json()
        rpc = JSONRPCRequest.model_validate(body)

        if rpc.method != "message/send":
            return JSONResponse(
                content=JSONRPCResponse(
                    id=rpc.id,
                    error={"code": -32601, "message": f"Method not found: {rpc.method}"},
                ).model_dump()
            )

        message_data = rpc.params.get("message", {})
        message = A2AMessage.model_validate(message_data)
        metadata = rpc.params.get("metadata", {})

        query = ""
        for part in message.parts:
            if part.text:
                query = part.text
                break

        task_id = str(rpc.id or uuid.uuid4())

        try:
            result_data = await handler(query, metadata)
            task = A2ATask(
                id=task_id,
                status=TaskState.COMPLETED,
                messages=[
                    message,
                    A2AMessage(
                        role="agent",
                        parts=[MessagePart(kind="data", data=result_data)],
                    ),
                ],
                artifacts=[{"name": "result", "data": result_data}],
            )
            return JSONResponse(
                content=JSONRPCResponse(id=rpc.id, result=task.model_dump(mode="json")).model_dump(
                    mode="json"
                )
            )
        except Exception as e:
            return JSONResponse(
                content=JSONRPCResponse(
                    id=rpc.id,
                    error={"code": -32000, "message": str(e)},
                ).model_dump(),
                status_code=500,
            )

    return app
