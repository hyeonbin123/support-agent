"""The MCP server, driven by the SDK's in-memory client."""

# [테스트 묶음] 4단계 MCP 서버(mcp_server.py)를 공식 MCP SDK의 메모리 내 클라이언트로 직접 부려 확인한다:
# 범위별 도구 목록과 주석(hint), 서버가 들고 있는 본인 확인 상태, 읽기 서버의 쓰기 거절, 쓰기의 실행·감사
# 로그·사람 승인, HTTP 토큰.
# [왜 중요한가] MCP로 붙어도 서비스와 같은 길(ChatService → toolkit.execute)을 지나야 규정 검사, 큰 환불의
# 사람 승인, 감사 로그를 건너뛸 수 없다 (docs/mcp.md "정한 것과 이유"). 이 테스트가 그 약속을 고정한다.
# [방법] 서비스와 같은 마이그레이션·seed 데이터를 담은 SQLite 파일을 모듈마다 한 번 만들고, 테스트마다 파일을
# 복사해 쓴다. 모델은 쓰지 않는다(빈 대본 제공자).

from __future__ import annotations

# httpx: HTTP 감싸개(BearerToken)를 실제 서버 없이 ASGI로 부른다. mcp.Client: SDK의 메모리 내 클라이언트.
# 서비스 쪽은 DB 준비(prepare_database), 한 턴·도구 호출 처리(ChatService), 서비스 테이블(store),
# 설정(Settings)을 가져온다.
import json

import httpx
import pytest
from mcp import Client
from sqlalchemy import select
from sqlalchemy.orm import Session

from support_agent import db
from support_agent.chat import ScriptedProvider
from support_agent.mcp_server import BearerToken, build_server, exposed
from support_agent.service import store
from support_agent.service.bootstrap import prepare_database
from support_agent.service.core import ChatService
from support_agent.service.settings import Settings

# SDK 클라이언트가 비동기라 이 파일의 테스트는 모두 anyio로 돌리고, 백엔드는 asyncio 하나로 정한다.
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# 마이그레이션과 seed 복사를 마친 SQLite 파일을 모듈에서 한 번만 만든다 (seed 데이터 복사가 테스트마다
# 반복되지 않게).
@pytest.fixture(scope="module")
def seeded_template(tmp_path_factory):
    path = tmp_path_factory.mktemp("mcp") / "template.db"
    prepare_database(f"sqlite:///{path.as_posix()}").dispose()
    return path


# 테스트마다 템플릿 파일을 복사한 새 DB로 ChatService를 만든다. 앞 테스트의 쓰기와 승인 대기열이 다음 테스트에
# 남지 않는다.
@pytest.fixture
def service(seeded_template, tmp_path):
    path = tmp_path / "service.db"
    path.write_bytes(seeded_template.read_bytes())
    engine = prepare_database(f"sqlite:///{path.as_posix()}")
    yield ChatService(Settings(), engine, ScriptedProvider([]))
    engine.dispose()


# MCP 도구 결과의 첫 글 내용.
def text_of(result) -> str:
    return result.content[0].text


# 기본 읽기 범위에서는 읽기 도구와 new_conversation만 보이고, 쓰기 도구와 평가용 think는 보이지 않는다. 도구
# 이름·설명·인자 스키마는 에이전트가 보는 정의(ToolSpec.schema())와 같아야 하고, 읽기 도구에는
# read_only_hint와 한국어 제목이 붙는다.
async def test_read_scope_lists_only_reads_and_never_the_think_tool(service):
    async with Client(build_server(service, "read")) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert "think" not in tools and "cancel_order" not in tools
    reads = {spec.name for spec in service.registry.values() if not spec.write} - {"think"}
    assert set(tools) == reads | {"new_conversation"}
    order = tools["get_order"]
    assert order.annotations.read_only_hint is True and order.title == "주문 조회"
    assert order.input_schema == service.registry["get_order"].schema()["parameters"]
    assert order.description == service.registry["get_order"].description


# 쓰기 범위에서는 think를 뺀 모든 도구 + new_conversation이 보이고, 쓰기 도구에는 destructive_hint가 붙는다.
# 클라이언트가 위험한 도구를 사람에게 확인받을 수 있게 하는 표시다.
async def test_write_scope_adds_the_writes_with_their_hints(service):
    async with Client(build_server(service, "write")) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert len(tools) - 1 == len(exposed(service, "write")) == len(service.registry) - 1
    assert tools["cancel_order"].annotations.destructive_hint is True
    assert tools["cancel_order"].annotations.read_only_hint is False


# 본인 확인 전에는 주문 조회가 거절되고, find_customer로 확인한 뒤에는 그 고객의 주문만 보인다(다른 고객
# 주문은 거절). 확인 상태는 연결이 아니라 서버 객체가 들고 있어, 같은 서버에 새로 붙은 클라이언트도 확인된
# 상태이고 다른 서버는 확인 전이다. 지금 프로토콜 개정판이 상태가 없는 방식이라 이렇게 정했다 (docs/mcp.md).
async def test_identity_is_kept_by_the_server_and_checked_by_the_tools(service):
    server = build_server(service, "read")
    async with Client(server) as client:
        refused = await client.call_tool("get_order", {"order_id": "O-90005"})
        assert refused.is_error and "Error: [" in text_of(refused)
        found = await client.call_tool("find_customer", {"name": "정예준", "contact": "010-0000-9005"})
        assert not found.is_error and json.loads(text_of(found))["customer_id"] == "C-9005"
        order = await client.call_tool("get_order", {"order_id": "O-90005"})
        assert json.loads(text_of(order))["total_won"] == 89_100
        other = await client.call_tool("get_order", {"order_id": "O-10097"})  # somebody else's order
        assert other.is_error
    async with Client(server) as second:  # the protocol is stateless: the state belongs to the server process
        assert not (await second.call_tool("get_order", {"order_id": "O-90005"})).is_error
    async with Client(build_server(service, "read")) as fresh:  # another server starts unverified
        assert (await fresh.call_tool("get_order", {"order_id": "O-90005"})).is_error


# 읽기 서버는 쓰기 도구(cancel_order)를 not_allowed로 거절하고, 없는 도구와 think도 오류, 잘못된 인자는
# invalid_arguments다. 거절된 뒤 DB의 주문은 그대로(preparing)여야 한다.
async def test_a_read_only_server_refuses_writes_and_unknown_tools(service):
    async with Client(build_server(service, "read")) as client:
        await client.call_tool("find_customer", {"name": "정예준", "contact": "01000009005"})
        refused = await client.call_tool("cancel_order", {"order_id": "O-90005", "reason": "changed_mind"})
        assert refused.is_error and "not_allowed" in text_of(refused)
        assert (await client.call_tool("no_such_tool", {})).is_error
        assert (await client.call_tool("think", {"thought": "x"})).is_error
        bad = await client.call_tool("get_order", {"order": "O-90005"})
        assert bad.is_error and "invalid_arguments" in text_of(bad)
    with Session(service.engine) as session:
        assert session.get(db.Order, "O-90005").status == db.OrderStatus.PREPARING


# 쓰기 서버에서: 작은 취소는 바로 실행되고 다시 취소하면 규정으로 거절된다. 한 상담은 고객 한 명이라 다른
# 고객을 확인하려면 new_conversation이 필요하다. 환불액 10만 원 이상인 큰 취소(O-10097, 108,200원)는
# approval_required로 승인 대기열에 올라가고 주문은 그대로다.
# 그 뒤 DB에서 세션 두 개(mcp- 접두사, 각자의 고객), 대기 중인 승인 하나, 도구 호출 감사 기록과 승인 요청
# 기록을 확인하고, 채팅과 같은 대기열에서 승인하면 그때 취소가 실행되는 것까지 본다.
async def test_writes_are_carried_out_audited_and_gated_like_in_the_chat(service):
    async with Client(build_server(service, "write")) as client:
        await client.call_tool("find_customer", {"name": "정예준", "contact": "01000009005"})
        done = await client.call_tool("cancel_order", {"order_id": "O-90005", "reason": "changed_mind"})
        assert not done.is_error and json.loads(text_of(done))["refund_won"] == 89_100
        # The policy is enforced here too: a cancelled order cannot be cancelled again.
        assert (
            await client.call_tool("cancel_order", {"order_id": "O-90005", "reason": "changed_mind"})
        ).is_error

        other = await client.call_tool("find_customer", {"name": "강정우", "contact": "01000006189"})
        assert other.is_error and "already_verified" in text_of(other)  # one session, one customer
        assert not (await client.call_tool("new_conversation", {})).is_error
        await client.call_tool("find_customer", {"name": "강정우", "contact": "01000006189"})
        big = await client.call_tool("cancel_order", {"order_id": "O-10097", "reason": "changed_mind"})
        assert big.is_error and "approval_required" in text_of(big)

    with Session(service.engine) as session:
        assert session.get(db.Order, "O-90005").status == db.OrderStatus.CANCELLED
        assert session.get(db.Order, "O-10097").status == db.OrderStatus.PREPARING
        first, row = session.scalars(select(store.ChatSession).order_by(store.ChatSession.created_at)).all()
        rows = {first.id: first.verified_customer_id, row.id: row.verified_customer_id}
        assert sorted(rows.values()) == ["C-0001", "C-9005"] and all(i.startswith("mcp-") for i in rows)
        row = first if first.verified_customer_id == "C-0001" else row
        assert row.state["messages"] == []
        approval = session.scalars(select(store.Approval)).one()
        assert (approval.session_id, approval.status, approval.refund_won) == (row.id, "pending", 108_200)
        kinds = [e.kind for e in session.scalars(select(store.AuditEvent).order_by(store.AuditEvent.id))]
    assert kinds.count("tool_call") == 6 and "approval_requested" in kinds

    decided = service.decide(approval.id, approve=True, by="kim")  # the same queue as the chat
    assert decided["status"] == "approved"
    with Session(service.engine) as session:
        assert session.get(db.Order, "O-10097").status == db.OrderStatus.CANCELLED


# HTTP로 띄울 때 모든 요청에 'Authorization: Bearer <토큰>'이 있어야 한다. 토큰이 없거나 틀리면 401이고,
# 맞으면 안쪽 앱(여기서는 204를 돌려주는 가짜)까지 간다.
async def test_the_http_wrapper_wants_the_bearer_token():
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    transport = httpx.ASGITransport(app=BearerToken(inner, "s3cret"))
    async with httpx.AsyncClient(transport=transport, base_url="http://mcp.test") as http:
        assert (await http.get("/mcp")).status_code == 401
        assert (await http.get("/mcp", headers={"Authorization": "Bearer wrong"})).status_code == 401
        assert (await http.get("/mcp", headers={"Authorization": "Bearer s3cret"})).status_code == 204
