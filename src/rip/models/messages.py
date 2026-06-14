"""A2A protocol message types (JSON-RPC 2.0 over HTTP)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class MessagePart(BaseModel):
    kind: Literal["text", "data"] = "text"
    text: str | None = None
    data: dict[str, Any] | None = None


class A2AMessage(BaseModel):
    role: Literal["user", "agent"] = "user"
    parts: list[MessagePart]


class A2ATask(BaseModel):
    id: str
    session_id: str | None = None
    status: TaskState = TaskState.SUBMITTED
    messages: list[A2AMessage] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class AgentCard(BaseModel):
    """Published at /.well-known/agent.json for A2A discovery."""

    name: str
    description: str
    version: str = "0.1.0"
    url: str
    capabilities: dict[str, bool] = Field(
        default_factory=lambda: {"streaming": False, "pushNotifications": False}
    )
    skills: list[AgentSkill] = Field(default_factory=list)
    default_input_modes: list[str] = Field(default_factory=lambda: ["text"])
    default_output_modes: list[str] = Field(default_factory=lambda: ["text"])


class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    id: str | int | None = None


class JSONRPCResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    id: str | int | None = None
