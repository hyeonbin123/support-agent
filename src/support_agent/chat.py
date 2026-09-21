"""Provider-neutral chat types, the ChatProvider interface and a scripted provider for tests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    name: str
    # Providers that send a JSON string parse it; unparsable text becomes {"_raw": text}.
    arguments: dict[str, Any]
    id: str = ""  # "" when the provider sent none


@dataclass(frozen=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()  # assistant only
    tool_name: str | None = None  # tool only (Ollama's native API uses the name)
    tool_call_id: str | None = None  # tool only (OpenAI-style and Anthropic APIs use the id)
    delivered: bool = True  # assistant only: False when the text never reached the customer
    harness: bool = False  # user only: True when the harness wrote it, not the customer

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {"name": c.name, "arguments": c.arguments, "id": c.id} for c in self.tool_calls
            ]
        if self.tool_name:
            out["tool_name"] = self.tool_name
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if not self.delivered:
            out["delivered"] = False
        if self.harness:
            out["harness"] = True
        return out

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Message:
        """The inverse of to_dict (the service keeps a conversation as JSON between requests)."""
        calls = tuple(
            ToolCall(c["name"], c["arguments"], c.get("id", "")) for c in data.get("tool_calls", ())
        )
        return Message(
            data["role"],
            data.get("content", ""),
            calls,
            data.get("tool_name"),
            data.get("tool_call_id"),
            data.get("delivered", True),
            data.get("harness", False),
        )


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_ms: float = 0.0  # measured by the client around the HTTP call
    load_ms: float = 0.0  # model load time reported by the server
    prompt_eval_ms: float = 0.0
    eval_ms: float = 0.0


@dataclass(frozen=True)
class ChatResponse:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None  # "length" means the reply was cut off
    thinking: str = ""  # reasoning text of thinking models; never sent back


class ProviderError(RuntimeError):
    """The model server failed (timeout, HTTP error, bad JSON). Not the agent's fault."""


class ChatProvider(ABC):
    name: str = "provider"

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Send the whole history and return one assistant turn.

        `tools` items are neutral: {"name", "description", "parameters"}; each provider wraps them itself.
        Failures of the server are raised as ProviderError.
        """

    def describe(self) -> dict[str, Any]:
        """What goes into the run manifest (model, digest, options)."""
        return {"provider": self.name}


class ScriptExhaustedError(RuntimeError):
    """The scripted provider was asked for more turns than the test prepared."""


class ScriptedProvider(ChatProvider):
    """Replays prepared turns. A str is a text reply, a ToolCall is a single tool call."""

    name = "scripted"

    def __init__(self, script: Sequence[ChatResponse | ToolCall | str]):
        self._script = list(script)
        self.requests: list[dict[str, Any]] = []

    def chat(self, messages, tools=(), *, temperature=0.0, seed=None, max_tokens=1024):
        self.requests.append(
            {"messages": list(messages), "tools": list(tools), "temperature": temperature, "seed": seed}
        )
        if not self._script:
            raise ScriptExhaustedError(f"script ran out after {len(self.requests) - 1} calls")
        item = self._script.pop(0)
        if isinstance(item, ChatResponse):
            return item
        if isinstance(item, ToolCall):
            return ChatResponse(tool_calls=(item,))
        return ChatResponse(text=item)
