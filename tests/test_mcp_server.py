"""The MCP server, driven by the SDK's in-memory client."""

from __future__ import annotations

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

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def seeded_template(tmp_path_factory):
    path = tmp_path_factory.mktemp("mcp") / "template.db"
    prepare_database(f"sqlite:///{path.as_posix()}").dispose()
    return path


@pytest.fixture
def service(seeded_template, tmp_path):
    path = tmp_path / "service.db"
    path.write_bytes(seeded_template.read_bytes())
    engine = prepare_database(f"sqlite:///{path.as_posix()}")
    yield ChatService(Settings(), engine, ScriptedProvider([]))
    engine.dispose()


def text_of(result) -> str:
    return result.content[0].text


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


async def test_write_scope_adds_the_writes_with_their_hints(service):
    async with Client(build_server(service, "write")) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert len(tools) - 1 == len(exposed(service, "write")) == len(service.registry) - 1
    assert tools["cancel_order"].annotations.destructive_hint is True
    assert tools["cancel_order"].annotations.read_only_hint is False


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


async def test_the_http_wrapper_wants_the_bearer_token():
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    transport = httpx.ASGITransport(app=BearerToken(inner, "s3cret"))
    async with httpx.AsyncClient(transport=transport, base_url="http://mcp.test") as http:
        assert (await http.get("/mcp")).status_code == 401
        assert (await http.get("/mcp", headers={"Authorization": "Bearer wrong"})).status_code == 401
        assert (await http.get("/mcp", headers={"Authorization": "Bearer s3cret"})).status_code == 204
