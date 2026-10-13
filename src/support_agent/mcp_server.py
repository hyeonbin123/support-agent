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

# [파일 역할] 4단계 MCP 서버. 에이전트가 쓰는 도구 등록부를 그대로 MCP 도구로 내놓아, MCP 클라이언트(Claude
# Desktop, Claude Code 등)의 모델이 상담원 역할을 하게 한다.
# [설계 이유] (docs/mcp.md '정한 것과 이유')
# - 서비스와 같은 길: 호출은 ChatService.call_tool → toolkit.execute()를 지나므로 규정 검사(P1), 큰 환불의
#   사람 승인, 감사 로그가 채팅과 똑같이 걸린다. MCP로 붙었다고 승인을 건너뛸 수 없다.
# - 읽기와 쓰기 분리: 기본은 읽기 도구만, --scope write여야 쓰기 도구가 보인다.
# - 도구 정의는 에이전트가 보는 것과 같은 ToolSpec.schema()에서 나온다. 스키마를 고치면 지금까지의 측정과
#   비교할 수 없게 되므로, 크기 제한 같은 방어는 MCP 경계에서 따로 한다.
# - 저수준 Server를 쓴다. 도구가 등록부에서 동적으로 만들어져 함수마다 데코레이터를 다는 고수준 API보다 맞다.
# [연결] service/core.py의 ChatService·TOOL_LABELS, service/bootstrap.py의 prepare_database,
# service/settings.py의 Settings를 쓴다. 테스트는 tests/test_mcp_server.py(SDK의 메모리 내 클라이언트).

from __future__ import annotations

# secrets: HTTP 토큰을 시간 차 공격에 안전하게 비교한다(compare_digest). threading: 여러 요청이 서버의 세션
# 상태를 함께 건드리지 않게 잠근다.
import argparse
import json
import os
import secrets
import sys
import threading
from typing import Any, Literal

# 공식 MCP Python SDK(mcp>=2.2,<3로 고정, docs/mcp.md)의 저수준 Server와 stdio 전송, 프로토콜 타입. anyio는
# 동기 도구 호출을 작업 스레드로 넘기고 stdio 서버를 돌리는 데 쓴다.
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

# 서비스 계층을 그대로 재사용한다. ScriptedProvider는 빈 대본으로 넣는다(아래 make_service): MCP에서는
# 클라이언트의 모델이 도구를 직접 부르므로 이 서버에는 LLM이 필요 없다.
from support_agent.chat import ScriptedProvider
from support_agent.config import THINK_TOOL
from support_agent.service.bootstrap import prepare_database
from support_agent.service.core import TOOL_LABELS, BusyError, ChatService
from support_agent.service.settings import PREFIX, Settings
from support_agent.toolkit import ToolSpec

# 크기 제한: 한 호출의 인자는 JSON으로 20,000바이트, HTTP 요청 본문은 256 KB까지. 도구의 자유 서술 인자(티켓
# 본문, 이관 요약)에는 길이 제한이 없어서 MCP 경계에서 막는다(docs/mcp.md).
Scope = Literal["read", "write"]
MAX_ARGUMENT_BYTES = 20_000  # of one call's arguments as JSON; the tools' free-text fields have no limit
MAX_HTTP_BODY_BYTES = 262_144
# MCP 전용 도구. 본인 확인 상태는 서버가 들고 있으므로, 다른 고객을 응대하려면 이 도구로 상담(서비스 세션)을
# 새로 시작한다. 상담 하나는 고객 한 명의 것이라는 채팅과 같은 규칙이다.
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
# MCP 초기화 때 클라이언트에 주는 서버 안내문. 본인 확인 먼저, 규정 위반은 도구가 거절, 큰 환불은 승인
# 대기라는 세 가지만 알린다.
INSTRUCTIONS = (
    "가상의 온라인 쇼핑몰 고객센터 도구입니다. 주문·고객 정보를 보려면 먼저 find_customer로 본인 확인을 "
    "해야 합니다. 규정에 어긋나는 처리는 도구가 거절하고, 환불액이 큰 취소·반품은 담당자 승인을 기다립니다."
)


# 범위(scope)에 따라 내놓을 도구. think는 평가에서 프롬프트 장치로 쓰는 도구라 쇼핑몰 도구가 아니므로 빼고,
# 쓰기 도구는 write 범위에서만 넣는다.
def exposed(service: ChatService, scope: Scope) -> list[ToolSpec]:
    """`think` is a prompt device of the evaluation, not a shop tool. Writes need the write scope."""
    return [
        spec
        for spec in service.registry.values()
        if spec.name != THINK_TOOL and (scope == "write" or not spec.write)
    ]


# ToolSpec → MCP Tool. 이름·설명·인자 스키마는 에이전트에게 보여 주는 것과 같다. 쓰기 여부로 MCP
# 주석(read_only_hint, destructive_hint, idempotent_hint)을 단다. 제목은 서비스 화면과 같은 한국어 도구
# 이름(TOOL_LABELS)이다.
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


# 도구 결과를 텍스트 하나로 감싼다. 도구가 거절하면 is_error=True와 함께 에이전트가 받는 것과 같은 'Error:
# [코드] 설명' 문장을 돌려준다. 클라이언트의 모델이 읽고 고칠 수 있게 하기 위해서다(docs/mcp.md).
def _text(text: str, *, is_error: bool = False) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=is_error)


# MCP Server 객체를 만든다. 서버는 서비스 세션 하나를 들고 있다. 지금 프로토콜 개정판은 상태가 없는 방식이고
# SDK가 요청마다 서버 세션 객체를 새로 만들기 때문에, 본인 확인 상태를 MCP 연결이 아니라 이 클로저의
# session_id에 둔다(모듈 docstring, docs/mcp.md).
def build_server(service: ChatService, scope: Scope = "read") -> Server:
    allowed = {spec.name: spec for spec in exposed(service, scope)}
    # 변경 가능한 칸으로 쓰려고 list에 담는다. 첫 도구 호출 때 만들어, 도구 목록만 조회한 클라이언트가 빈
    # 세션을 남기지 않게 한다.
    session_id: list[str] = []  # made on the first call, so that listing tools leaves no empty session
    guard = threading.Lock()

    # 도구 목록 요청: 허용된 도구 + new_conversation.
    async def list_tools(
        _ctx: ServerRequestContext, _params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        return ListToolsResult(tools=[*(as_mcp_tool(spec) for spec in allowed.values()), NEW_CONVERSATION])

    # 도구 호출 하나를 처리하는 동기 함수(작업 스레드에서 돈다).
    def call(name: str, arguments: dict[str, Any]) -> CallToolResult:
        if name == NEW_CONVERSATION.name:
            with guard:
                session_id.clear()
            return _text('{"ok": true}')
        # 허용 목록에 없는 이름: 등록부에는 있는데 범위 때문에 숨긴 쓰기 도구면 권한이 없다고, 아니면 없는
        # 도구라고 알린다. 둘 다 not_allowed 오류다.
        if name not in allowed:
            hidden = name in service.registry and name != THINK_TOOL
            reason = "쓰기 권한(--scope write)이 없습니다" if hidden else "그런 도구는 없습니다"
            return _text(f"Error: [not_allowed] {name}: {reason}.", is_error=True)
        # 인자 크기 제한. UTF-8 바이트로 재므로 한글은 글자당 3바이트로 센다.
        if len(json.dumps(arguments, ensure_ascii=False).encode()) > MAX_ARGUMENT_BYTES:
            return _text(
                f"Error: [invalid_arguments] 인자가 너무 큽니다 ({MAX_ARGUMENT_BYTES}바이트까지).",
                is_error=True,
            )
        # 세션 id를 잠금 안에서 읽어 두고 잠금 밖에서 호출한다. 호출 도중 new_conversation이 목록을 비워도 이
        # 호출은 시작할 때의 세션으로 끝난다.
        with guard:
            if not session_id:
                session_id.append(service.new_session("mcp")["session_id"])
            current = session_id[0]  # new_conversation may clear the list right after the lock
        # 서비스의 단일 실행 경로로 호출한다. 같은 세션의 잠금을 30초 안에 얻지 못하면(앞선 호출이나 그 세션의
        # 승인 처리가 진행 중) 서비스가 BusyError를 내고, busy 오류로 돌려준다.
        try:
            result = service.call_tool(current, name, arguments)
        except BusyError:
            return _text("Error: [busy] 앞선 호출이 끝나지 않았습니다.", is_error=True)
        return _text(result.content, is_error=not result.ok)

    # 도구는 SQLAlchemy를 쓰는 동기 코드라, 실행하는 동안 이벤트 루프가 막히지 않게 작업 스레드로 넘긴다.
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


# HTTP 모드의 인증. ASGI 앱을 감싸 모든 HTTP 요청에 'Authorization: Bearer <토큰>'이 있는지 본다.
# secrets.compare_digest로 비교해 비교 시간으로 토큰이 새지 않게 한다. HTTP가 아닌 ASGI 이벤트(lifespan 등)는
# 그대로 통과시킨다.
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


# 서비스와 같은 설정(SUPPORT_AGENT_DATABASE_URL 등)으로 DB를 준비하고 ChatService를 만든다. Compose의
# PostgreSQL을 가리키면 채팅 서비스와 같은 DB, 같은 승인 대기열, 같은 감사 로그를 쓴다(docs/mcp.md).
def make_service(settings: Settings) -> ChatService:
    engine = prepare_database(settings.database_url, load_seed=settings.load_seed)
    return ChatService(settings, engine, ScriptedProvider([]))  # no model here: the client is the model


# 명령줄 진입점. 기본은 stdio(클라이언트가 이 프로세스를 직접 띄우므로 프로세스 하나 = 클라이언트 하나)이고,
# --http면 Streamable HTTP로 연다.
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

    # HTTP는 토큰 없이는 뜨지 않는다. 쓰기 도구가 포트에 닿는 누구에게나 열리면 안 되기 때문이다. 기본 주소도
    # 127.0.0.1이다. uvicorn은 HTTP 모드에서만 필요해 여기서 가져온다.
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
