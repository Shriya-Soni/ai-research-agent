"""A2A client for the LangGraph coordinator to call specialist agents."""

from __future__ import annotations

import uuid
from typing import Any

import httpx

from rip.models.messages import (
    A2AMessage,
    A2ATask,
    AgentCard,
    JSONRPCRequest,
    JSONRPCResponse,
    MessagePart,
    TaskState,
)


class A2AClient:
    """Client agent that discovers and delegates tasks to remote A2A agents."""

    def __init__(self, agent_url: str, timeout: float = 120.0):
        self.agent_url = agent_url.rstrip("/")
        self.timeout = timeout

    async def get_agent_card(self) -> AgentCard:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.agent_url}/.well-known/agent.json")
            resp.raise_for_status()
            return AgentCard.model_validate(resp.json())

    async def send_message(
        self,
        query: str,
        metadata: dict[str, Any] | None = None,
    ) -> A2ATask:
        task_id = str(uuid.uuid4())
        request = JSONRPCRequest(
            method="message/send",
            id=task_id,
            params={
                "message": A2AMessage(
                    role="user",
                    parts=[MessagePart(kind="text", text=query)],
                ).model_dump(),
                "metadata": metadata or {},
            },
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.agent_url}/a2a",
                json=request.model_dump(),
            )
            resp.raise_for_status()
            rpc_response = JSONRPCResponse.model_validate(resp.json())

            if rpc_response.error:
                raise RuntimeError(f"A2A error: {rpc_response.error}")

            return A2ATask.model_validate(rpc_response.result)

    async def execute(self, query: str, **metadata: Any) -> dict[str, Any]:
        """High-level: send query and extract artifacts from completed task."""
        task = await self.send_message(query, metadata=metadata)
        if task.status == TaskState.FAILED:
            raise RuntimeError(f"Agent task failed: {task.id}")

        artifacts = task.artifacts
        if artifacts:
            return artifacts[0].get("data", {})
        return {"messages": [m.model_dump() for m in task.messages]}
