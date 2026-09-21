"""The shop tools as an MCP server, built from the same registry as the agent.

    uv run python -m support_agent.mcp_server                      # stdio, read-only tools
    uv run python -m support_agent.mcp_server --scope write        # also the tools that change the database
    uv run python -m support_agent.mcp_server --http --port 8063   # Streamable HTTP (SUPPORT_AGENT_MCP_TOKEN)

A call takes the service's path (`ChatService.call_tool`): the tools enforce the policy, a large refund waits
in the approval queue, and every call leaves an audit row.

The server holds one service session at a time: a customer verified with `find_customer` stays verified
until `new_conversation` is called (a session belongs to one customer, as in the chat). The state is not
tied to the MCP connection because the current protocol revision is stateless (the SDK builds a new server
session for every request). Over stdio the client starts its own process, so this is "per client"; the HTTP
server is for one operator per token.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import threading
from typing import Any, Literal

import anyio
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
    ToolAnnotations,
)

from support_agent.chat import ScriptedProvider
from support_agent.config import THINK_TOOL
from support_agent.service.bootstrap import prepare_database
from support_agent.service.core import TOOL_LABELS, BusyError, ChatService
from support_agent.service.settings import PREFIX, Settings
from support_agent.toolkit import ToolSpec

Scope = Literal["read", "write"]
MAX_ARGUMENT_BYTES = 20_000  # of one call's arguments as JSON; the tools' free-text fields have no limit
MAX_HTTP_BODY_BYTES = 262_144
NEW_CONVERSATION = Tool(
    name="new_conversation",
    title="새 상담 시작",
    description=(
        "지금 상담을 끝내고 새 상담을 시작한다. 본인 확인 상태가 지워진다. "
        "다른 고객을 응대하기 전에 호출한다."
    ),
    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False),
)
INSTRUCTIONS = (
    "가상의 온라인 쇼핑몰 고객센터 도구입니다. 주문·고객 정보를 보려면 먼저 find_customer로 본인 확인을 "
    "해야 합니다. 규정에 어긋나는 처리는 도구가 거절하고, 환불액이 큰 취소·반품은 담당자 승인을 기다립니다."
)


def exposed(service: ChatService, scope: Scope) -> list[ToolSpec]:
    """`think` is a prompt device of the evaluation, not a shop tool. Writes need the write scope."""
    return [
        spec
        for spec in service.registry.values()
        if spec.name != THINK_TOOL and (scope == "write" or not spec.write)
    ]


def as_mcp_tool(spec: ToolSpec) -> Tool:
    return Tool(
        name=spec.name,
        title=TOOL_LABELS.get(spec.name),
        description=spec.description,
        input_schema=spec.schema()["parameters"],
        annotations=ToolAnnotations(
            read_only_hint=not spec.write,
            destructive_hint=spec.write,
            idempotent_hint=not spec.write,
            open_world_hint=False,
        ),
    )


def _text(text: str, *, is_error: bool = False) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=is_error)


def build_server(service: ChatService, scope: Scope = "read") -> Server:
    allowed = {spec.name: spec for spec in exposed(service, scope)}
    session_id: list[str] = []  # made on the first call, so that listing tools leaves no empty session
    guard = threading.Lock()

    async def list_tools(
        _ctx: ServerRequestContext, _params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        return ListToolsResult(tools=[*(as_mcp_tool(spec) for spec in allowed.values()), NEW_CONVERSATION])

    def call(name: str, arguments: dict[str, Any]) -> CallToolResult:
        if name == NEW_CONVERSATION.name:
            with guard:
                session_id.clear()
            return _text('{"ok": true}')
        if name not in allowed:
            hidden = name in service.registry and name != THINK_TOOL
            reason = "쓰기 권한(--scope write)이 없습니다" if hidden else "그런 도구는 없습니다"
            return _text(f"Error: [not_allowed] {name}: {reason}.", is_error=True)
        if len(json.dumps(arguments, ensure_ascii=False).encode()) > MAX_ARGUMENT_BYTES:
            return _text(
                f"Error: [invalid_arguments] 인자가 너무 큽니다 ({MAX_ARGUMENT_BYTES}바이트까지).",
                is_error=True,
            )
        with guard:
            if not session_id:
                session_id.append(service.new_session("mcp")["session_id"])
            current = session_id[0]  # new_conversation may clear the list right after the lock
        try:
            result = service.call_tool(current, name, arguments)
        except BusyError:
            return _text("Error: [busy] 앞선 호출이 끝나지 않았습니다.", is_error=True)
        return _text(result.content, is_error=not result.ok)

    async def call_tool(_ctx: ServerRequestContext, params: CallToolRequestParams) -> CallToolResult:
        # The tools are synchronous (SQLAlchemy); keep the event loop free while one runs.
        return await anyio.to_thread.run_sync(call, params.name, params.arguments or {})

    return Server(
        "support-agent",
        version="0.1.0",
        instructions=INSTRUCTIONS,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


class BearerToken:
    """ASGI wrapper: every HTTP request must carry `Authorization: Bearer <token>`."""

    def __init__(self, app, token: str):
        self.app, self.expected = app, f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            given = dict(scope["headers"]).get(b"authorization", b"")
            if not secrets.compare_digest(given, self.expected):
                await send({"type": "http.response.start", "status": 401, "headers": []})
                await send({"type": "http.response.body", "body": b"unauthorized"})
                return
        await self.app(scope, receive, send)


def make_service(settings: Settings) -> ChatService:
    engine = prepare_database(settings.database_url, load_seed=settings.load_seed)
    return ChatService(settings, engine, ScriptedProvider([]))  # no model here: the client is the model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scope", choices=["read", "write"], default="read", help="write also exposes writes"
    )
    parser.add_argument("--http", action="store_true", help="Streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8063)
    args = parser.parse_args()

    server = build_server(make_service(Settings.from_env()), args.scope)
    if not args.http:

        async def run_stdio() -> None:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(read_stream, write_stream, server.create_initialization_options())

        anyio.run(run_stdio)
        return

    token = os.environ.get(PREFIX + "MCP_TOKEN", "")
    if not token:
        sys.exit(
            f"--http needs {PREFIX}MCP_TOKEN: the write tools must not be open to whoever reaches the port"
        )
    import uvicorn

    http_app = server.streamable_http_app(host=args.host, max_request_body_size=MAX_HTTP_BODY_BYTES)
    app = BearerToken(http_app, token)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
