import json

import pytest
from conftest import NOW
from sqlalchemy.orm import Session
from toy_tools import TOY_REGISTRY, OrderArgs, toy_get_order

from support_agent import db
from support_agent.toolkit import (
    ToolArgs,
    ToolBugError,
    ToolContext,
    ToolResult,
    canonical_args,
    execute,
    make_registry,
    tool,
)


def run(engine, name, args, **kw):
    ctx = ToolContext(now=NOW, **kw)
    return execute(TOY_REGISTRY, engine, ctx, name, args), ctx


def test_a_read_tool_returns_json(tiny_engine):
    result, _ = run(tiny_engine, "toy_get_order", {"order_id": "O-1"})
    assert result.ok and json.loads(result.content)["total_won"] == 38900


def test_unknown_tool_and_bad_arguments_are_reported_not_raised(tiny_engine):
    result, _ = run(tiny_engine, "nope", {})
    assert (result.ok, result.error_code) == (False, "unknown_tool")
    result, _ = run(tiny_engine, "toy_get_order", {"order_id": "O-1", "invented": 1})
    assert result.error_code == "invalid_arguments"
    assert result.content.startswith("Error: [invalid_arguments]")


def test_p1_refuses_a_policy_violation_and_leaves_the_db_alone(tiny_engine):
    engine = db.memory_engine(tiny_engine)
    before = db.dump_db(engine)
    result, ctx = run(engine, "toy_cancel_order", {"order_id": "O-2"}, enforce_policy=True)
    assert (result.ok, result.error_code, result.policy_blocked) == (False, "cancel_not_allowed_status", True)
    assert db.dump_db(engine) == before and ctx.violations == []


def test_p0_lets_the_violation_through_and_records_it(tiny_engine):
    engine = db.memory_engine(tiny_engine)
    result, ctx = run(engine, "toy_cancel_order", {"order_id": "O-2"}, enforce_policy=False)
    assert result.ok and result.violations == ("cancel_not_allowed_status",)
    changed = db.diff_dumps(db.dump_db(tiny_engine), db.dump_db(engine))
    assert changed["orders"]["only_actual"][0]["status"] == "cancelled"


def test_integrity_checks_refuse_in_every_mode(tiny_engine):
    result, _ = run(tiny_engine, "toy_get_order", {"order_id": "O-404"}, enforce_policy=False)
    assert (result.error_code, result.policy_blocked) == ("order_not_found", False)


def test_a_handler_bug_is_raised_as_tool_bug(tiny_engine):
    with pytest.raises(ToolBugError):
        run(tiny_engine, "toy_bug", {"order_id": "O-1"})


def test_before_write_can_refuse_a_write_but_is_not_asked_about_reads(tiny_engine):
    seen = []

    def gate(spec, args):
        seen.append((spec.name, args))
        return ToolResult.error("confirmation_required", "고객 확인이 필요합니다.")

    ctx = ToolContext(now=NOW)
    assert execute(TOY_REGISTRY, tiny_engine, ctx, "toy_get_order", {"order_id": "O-1"}, before_write=gate).ok
    refused = execute(
        TOY_REGISTRY, tiny_engine, ctx, "toy_cancel_order", {"order_id": "O-1"}, before_write=gate
    )
    assert refused.error_code == "confirmation_required"
    assert seen == [("toy_cancel_order", {"order_id": "O-1"})]


def test_the_episode_copy_is_private(tiny_engine):
    engine = db.memory_engine(tiny_engine)
    run(engine, "toy_cancel_order", {"order_id": "O-1"})
    assert db.state_hash(db.dump_db(engine)) != db.state_hash(db.dump_db(tiny_engine))


def test_two_copies_that_run_the_same_calls_end_in_the_same_state(tiny_engine):
    hashes = []
    for _ in range(2):
        engine = db.memory_engine(tiny_engine)
        run(engine, "toy_cancel_order", {"order_id": "O-1"})
        hashes.append(db.state_hash(db.dump_db(engine)))
    assert hashes[0] == hashes[1]


def test_registry_refuses_tools_without_a_required_argument():
    class Loose(ToolArgs):
        note: str = ""

    @tool(write=False)
    def loose(session, ctx, args: Loose) -> dict:
        """doc"""
        return {}

    with pytest.raises(ValueError, match="required"):
        make_registry(loose)


def test_canonical_args_sorts_order_free_lists():
    class LinesArgs(ToolArgs):
        line_nos: list[int]

    @tool(write=True, unordered_args=("line_nos",))
    def lines(session, ctx, args: LinesArgs) -> dict:
        """doc"""
        return {}

    assert canonical_args(lines, LinesArgs(line_nos=[2, 1])) == {"line_nos": [1, 2]}


def test_a_tool_spec_is_callable_and_has_a_flat_schema(tiny_engine):
    with Session(tiny_engine) as session:
        assert toy_get_order(session, ToolContext(now=NOW), OrderArgs(order_id="O-1"))["status"] == "paid"
    schema = toy_get_order.schema()
    assert schema["parameters"]["required"] == ["order_id"] and "title" not in json.dumps(schema)
