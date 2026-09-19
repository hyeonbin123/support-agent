"""ChatProvider for Ollama's native /api/chat, called with httpx directly."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from typing import Any

import httpx

from support_agent.chat import ChatProvider, ChatResponse, Message, ProviderError, ToolCall, Usage

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_META_TIMEOUT_S = 10.0  # /api/tags, /api/version, /api/ps answer at once or not at all
_UNSET: Any = object()


def _clean(value: Any) -> Any:
    """Replace lone surrogates in every string; they cannot be written as UTF-8 later."""
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, dict):
        return {_clean(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _ms(raw: dict[str, Any], key: str) -> float:
    return (raw.get(key, 0) or 0) / 1e6  # the server reports nanoseconds


def _wire_message(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_name": m.tool_name or "", "content": m.content}
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.role == "assistant" and m.tool_calls:
        # arguments must be a JSON object: the server rejects a string here
        out["tool_calls"] = [
            {"function": {"name": c.name, "arguments": dict(c.arguments)}} for c in m.tool_calls
        ]
    return out


def _wire_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        },
    }


def _parse_arguments(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except ValueError:
            return {"_raw": _clean(args)}
        return _clean(parsed) if isinstance(parsed, dict) else {"_raw": _clean(args)}
    if isinstance(args, dict):
        return _clean(args)
    return {"_raw": _clean(args)}


def _parse_tool_calls(items: Any) -> tuple[ToolCall, ...]:
    if not items:
        return ()
    if not isinstance(items, list):
        raise ProviderError(f"ollama: tool_calls is not a list: {str(items)[:300]}")
    calls = []
    for item in items:
        fn = item.get("function") if isinstance(item, dict) else None
        if not isinstance(fn, dict):
            raise ProviderError(f"ollama: malformed tool call: {str(item)[:300]}")
        calls.append(
            ToolCall(
                name=_clean(str(fn.get("name") or "")),
                arguments=_parse_arguments(fn.get("arguments")),
                id=_clean(str(item.get("id") or "")),
            )
        )
    return tuple(calls)


class OllamaProvider(ChatProvider):
    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        num_ctx: int = 16384,
        keep_alive: str | int = "60m",
        think: bool | None = None,
        base_url: str | None = None,
        timeout_s: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        # Fixed per instance: different runner options between requests make the server reload the model.
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.think = think
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        # The first request may have to load the model from a slow disk, hence the long read timeout.
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
            transport=transport,
        )
        self._digest: Any = _UNSET
        self._version: Any = _UNSET

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OllamaProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
        try:
            resp = self._http.request(method, path, json=body, **kwargs)
        except httpx.HTTPError as e:
            raise ProviderError(f"ollama: {method} {path} failed: {type(e).__name__}: {e}") from e
        if not resp.is_success:
            raise ProviderError(
                f"ollama: {method} {path} returned HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderError(f"ollama: {method} {path} returned invalid JSON: {resp.text[:300]}") from e
        if not isinstance(data, dict):
            raise ProviderError(f"ollama: {method} {path} returned a non-object body: {resp.text[:300]}")
        return data

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        options: dict[str, Any] = {
            "num_ctx": self.num_ctx,
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if seed is not None:
            options["seed"] = seed
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_wire_message(m) for m in messages],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        if tools:
            body["tools"] = [_wire_tool(t) for t in tools]
        if self.think is not None:  # models without thinking answer 400 to any think value
            body["think"] = self.think

        start = time.perf_counter()
        raw = self._request("POST", "/api/chat", body)
        wall_ms = (time.perf_counter() - start) * 1000.0

        msg = raw.get("message")
        if not isinstance(msg, dict):
            detail = raw.get("error") or json.dumps(raw, ensure_ascii=False)[:300]
            raise ProviderError(f"ollama: response has no message: {detail}")
        return ChatResponse(
            text=_clean(msg.get("content") or ""),
            tool_calls=_parse_tool_calls(msg.get("tool_calls")),
            usage=Usage(
                prompt_tokens=raw.get("prompt_eval_count", 0) or 0,
                completion_tokens=raw.get("eval_count", 0) or 0,
                wall_ms=wall_ms,
                load_ms=_ms(raw, "load_duration"),
                prompt_eval_ms=_ms(raw, "prompt_eval_duration"),
                eval_ms=_ms(raw, "eval_duration"),
            ),
            finish_reason=raw.get("done_reason"),
            thinking=_clean(msg.get("thinking") or ""),
        )

    def preload(self) -> float:
        """Load the model with this provider's runner options. Returns the load time in ms."""
        body = {
            "model": self.model,
            "messages": [],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"num_ctx": self.num_ctx},
        }
        return _ms(self._request("POST", "/api/chat", body), "load_duration")

    def loaded_models(self) -> list[dict[str, Any]]:
        """Models the server holds in memory right now; [] when the server cannot be asked."""
        try:
            models = self._request("GET", "/api/ps", timeout=_META_TIMEOUT_S).get("models") or []
            return [
                {"name": m.get("name") or m.get("model") or "", "size_vram": m.get("size_vram", 0) or 0}
                for m in models
                if isinstance(m, dict)
            ]
        except Exception:
            return []

    def _fetch_digest(self) -> str | None:
        models = self._request("GET", "/api/tags", timeout=_META_TIMEOUT_S).get("models") or []
        wanted = {self.model} if ":" in self.model else {self.model, f"{self.model}:latest"}
        for m in models:
            if isinstance(m, dict) and (m.get("name") in wanted or m.get("model") in wanted):
                return m.get("digest")
        return None

    def _fetch_version(self) -> str | None:
        return self._request("GET", "/api/version", timeout=_META_TIMEOUT_S).get("version")

    def describe(self) -> dict[str, Any]:
        # A failed lookup is not cached, so a later describe() can still fill it in.
        if self._digest is _UNSET:
            try:
                self._digest = self._fetch_digest()
            except Exception:
                pass
        if self._version is _UNSET:
            try:
                self._version = self._fetch_version()
            except Exception:
                pass
        return {
            "provider": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "num_ctx": self.num_ctx,
            "keep_alive": self.keep_alive,
            "think": self.think,
            "digest": None if self._digest is _UNSET else self._digest,
            "ollama_version": None if self._version is _UNSET else self._version,
        }
