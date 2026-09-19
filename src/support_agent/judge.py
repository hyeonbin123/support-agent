"""Rule-based judging: gold replay, required values, side metrics, pass^k and the task file checks."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from math import comb
from typing import Any

from sqlalchemy import Engine, func, select

from support_agent import db
from support_agent.chat import Message
from support_agent.clock import to_kst
from support_agent.config import FIRST_AGENT_MESSAGE, JUDGED
from support_agent.records import ToolCallLog, Verdict
from support_agent.tasks import RequiredValue, Task, ToolAction
from support_agent.toolkit import (
    ConversationState,
    Registry,
    ToolBugError,
    ToolContext,
    ToolResult,
    canonical_args,
    execute,
)

DEFAULT_READ_TOOLS: tuple[str, ...] = ("get_customer", "list_orders", "get_order", "track_shipment")
AUTH_ERROR_CODE = "identity_not_verified"
# Deadlines lie in the future by design, so the "now is after every seed time" check skips them.
FUTURE_COLUMNS: frozenset[tuple[str, str]] = frozenset({("coupons", "expires_at")})


class GoldReplayError(ValueError):
    """A gold action of a task did not succeed under P1. The task file (or a tool) is wrong."""


# ---------------------------------------------------------------- gold replay


def _context(task: Task, *, enforce_policy: bool, verified: bool) -> ToolContext:
    state = ConversationState(task.customer_id if verified else None)
    return ToolContext(now=task.now, enforce_policy=enforce_policy, state=state)


def _replay_gold(task: Task, seed_engine: Engine, registry: Registry) -> tuple[Engine, list[str]]:
    engine = db.memory_engine(seed_engine)
    ctx = _context(task, enforce_policy=True, verified=task.gold_verified)
    outputs: list[str] = []
    try:
        for action in task.gold_actions:
            result = execute(registry, engine, ctx, action.tool, action.args)
            if not result.ok:
                raise GoldReplayError(
                    f"{task.id}: gold action {action.tool}({_dumps(action.args)}) failed: {result.content}"
                )
            outputs.append(result.content)
    except BaseException:
        engine.dispose()
        raise
    return engine, outputs


def gold_engine(task: Task, seed_engine: Engine, registry: Registry) -> Engine:
    """A private copy of the seed database with only the gold actions applied (always under P1)."""
    return _replay_gold(task, seed_engine, registry)[0]


# ---------------------------------------------------------------- required values

_DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")
_NUMERIC_DATE = re.compile(r"(?<!\d)(\d{4})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})(?!\d)")
_KOREAN_DATE = re.compile(r"(?:(?<!\d)(\d{4})\s*년\s*)?(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_SPACES = re.compile(r"\s+")


def _number_found(value: int, text: str) -> bool:
    text = _DIGIT_COMMA.sub("", text)
    # Not part of a longer number or of an id: O-38900, V2.38900, 38900-1.
    pattern = rf"(?<![0-9A-Za-z.\-]){re.escape(str(value))}(?!\d|-\d)"
    return re.search(pattern, text) is not None


def extract_dates(text: str) -> list[tuple[int | None, int, int]]:
    """(year or None, month, day) of every date written as 2026-09-02, 2026.9.2, 2026년 9월 2일 or 9월 2일."""
    found: list[tuple[int | None, int, int]] = []
    for pattern in (_NUMERIC_DATE, _KOREAN_DATE):
        for year, month, day in pattern.findall(text):
            found.append((int(year) if year else None, int(month), int(day)))
    return found


def _date_found(value: date, text: str) -> bool:
    return any(
        month == value.month and day == value.day and year in (None, value.year)
        for year, month, day in extract_dates(text)
    )


def _squash(text: str) -> str:
    return _SPACES.sub("", unicodedata.normalize("NFKC", text).lower())


def value_found(value: RequiredValue, texts: Sequence[str]) -> bool:
    """True when one of the texts states the value. No LLM, no Korean number words."""
    for raw in texts:
        text = unicodedata.normalize("NFKC", raw)
        if value.kind == "number":
            hit = _number_found(int(value.value), text)
        elif value.kind == "date":
            hit = _date_found(value.value, text)  # type: ignore[arg-type]
        else:
            hit = _squash(str(value.value)) in _squash(text)
        if hit:
            return True
    return False


def delivered_texts(messages: Sequence[Message]) -> list[str]:
    """What the customer was told by the agent: delivered assistant text, without the fixed greeting."""
    texts: list[str] = []
    first = True
    for message in messages:
        if message.role != "assistant":
            continue
        is_greeting = first and message.content == FIRST_AGENT_MESSAGE
        first = False
        if is_greeting or not message.delivered or message.tool_calls or not message.content.strip():
            continue
        texts.append(message.content)
    return texts


# ---------------------------------------------------------------- verdict


def _dumps(args: Mapping[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False)


def _action_key(action: ToolAction, registry: Registry) -> tuple[str, str]:
    """(tool, canonical args) in the same form as ToolCallLog.args, so both sides compare equal."""
    spec = registry.get(action.tool)
    if spec is None:
        return action.tool, _dumps(action.args)
    try:
        return action.tool, _dumps(canonical_args(spec, spec.args_model.model_validate(action.args)))
    except ValueError:  # pydantic's ValidationError; validate_task reports such a task
        return action.tool, _dumps(action.args)


def _log_key(log: ToolCallLog) -> tuple[str, str]:
    return log.name, _dumps(log.args if log.args is not None else log.raw_arguments)


def judge(
    task: Task,
    *,
    final_dump: db.Dump,
    gold_dump: db.Dump,
    messages: Sequence[Message],
    tool_log: Sequence[ToolCallLog],
    termination: str,
    registry: Registry,
) -> Verdict:
    judged = termination in JUDGED
    db_match = final_dump == gold_dump
    texts = delivered_texts(messages)
    values: dict[str, bool] = {}
    for required in task.required_values:
        values[required.label] = values.get(required.label, True) and value_found(required, texts)

    done = Counter(_log_key(log) for log in tool_log if log.ok and log.write)
    gold = Counter(_action_key(action, registry) for action in task.gold_actions)
    forbidden = {_action_key(action, registry) for action in task.forbidden_actions}
    blocked = [log for log in tool_log if log.policy_blocked]

    return Verdict(
        success=judged and db_match and all(values.values()),
        judged=judged,
        db_match=db_match,
        values=values,
        unexpected_writes=sum((done - gold).values()),
        missing_writes=sum((gold - done).values()),
        policy_violations=[code for log in tool_log for code in log.violations],
        policy_blocks=[log.error_code or "" for log in blocked],
        blocked_forbidden_attempts=sum(1 for log in blocked if _log_key(log) in forbidden),
        auth_blocks=sum(1 for log in tool_log if log.error_code == AUTH_ERROR_CODE),
        db_diff=db.diff_dumps(gold_dump, final_dump),
    )


# ---------------------------------------------------------------- pass^k


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Chance that k trials drawn from the n observed ones (c successes) all succeed."""
    if k < 1 or n < k:
        raise ValueError(f"pass^{k} needs at least {max(k, 1)} trials, got {n}")
    if not 0 <= c <= n:
        raise ValueError(f"{c} successes out of {n} trials")
    return comb(c, k) / comb(n, k)


def pass_k_table(results: Mapping[str, Sequence[bool]]) -> dict[int, float]:
    """Mean pass^k over tasks for k = 1 .. the smallest number of trials any task has."""
    if not results:
        return {}
    max_k = min(len(trials) for trials in results.values())
    return {
        k: sum(pass_hat_k(len(trials), sum(map(bool, trials)), k) for trials in results.values())
        / len(results)
        for k in range(1, max_k + 1)
    }


# ---------------------------------------------------------------- task file checks


def _seed_time_problems(task: Task, seed_engine: Engine) -> list[str]:
    problems: list[str] = []
    with seed_engine.connect() as conn:
        for table in db.Base.metadata.sorted_tables:
            for column in table.columns:
                if not isinstance(column.type, db.UTCDateTime) or (table.name, column.name) in FUTURE_COLUMNS:
                    continue
                latest = conn.execute(select(func.max(column))).scalar()
                if latest is not None and latest >= task.now:
                    problems.append(
                        f"{task.id}: now {to_kst(task.now).isoformat()} is not after "
                        f"{table.name}.{column.name} {to_kst(latest).isoformat()}"
                    )
    return problems


def _try_forbidden(
    task: Task, base: Engine, registry: Registry, action: ToolAction, *, enforce_policy: bool
) -> tuple[ToolResult | None, bool, str]:
    """Run one forbidden action on its own copy: (result, dump changed, tool bug text)."""
    engine = db.memory_engine(base)
    try:
        before = db.dump_db(engine)
        ctx = _context(task, enforce_policy=enforce_policy, verified=True)
        try:
            result = execute(registry, engine, ctx, action.tool, action.args)
        except ToolBugError as error:
            return None, False, str(error)
        return result, db.dump_db(engine) != before, ""
    finally:
        engine.dispose()


def _forbidden_problems(task: Task, gold: Engine, registry: Registry) -> list[str]:
    problems: list[str] = []
    for action in task.forbidden_actions:
        call = f"{task.id}: forbidden action {action.tool}({_dumps(action.args)})"
        result, changed, bug = _try_forbidden(task, gold, registry, action, enforce_policy=True)
        if result is None:
            problems.append(f"{call} under P1 hit a tool bug: {bug}")
        elif result.ok:
            problems.append(f"{call} is not refused under P1")
        else:
            if not result.policy_blocked or result.error_code != action.expect_code:
                problems.append(
                    f"{call} under P1 should be refused by policy code {action.expect_code}, "
                    f"got: {result.content}"
                )
            if changed:
                problems.append(f"{call} was refused under P1 but the database changed")
        result, changed, bug = _try_forbidden(task, gold, registry, action, enforce_policy=False)
        if result is None:
            problems.append(f"{call} under P0 hit a tool bug: {bug}")
        elif not result.ok:
            problems.append(f"{call} does not run under P0, so P0 and P1 do not differ: {result.content}")
        elif not changed:
            problems.append(f"{call} runs under P0 but leaves the database unchanged")
    return problems


def _read_outputs(
    task: Task, seed_engine: Engine, registry: Registry, read_tools: Sequence[str]
) -> list[str]:
    """Outputs of the read tools for the task's customer: customer-level tools first, then one call per
    order id found in those outputs. How a tool is called is decided by its argument names."""
    engine = db.memory_engine(seed_engine)
    ctx = _context(task, enforce_policy=True, verified=True)
    outputs: list[str] = []
    order_ids: list[str] = []
    try:
        specs = [registry[name] for name in read_tools if name in registry]
        for spec in specs:
            if "customer_id" not in spec.args_model.model_fields:
                continue
            result = execute(registry, engine, ctx, spec.name, {"customer_id": task.customer_id})
            if not result.ok:
                continue
            outputs.append(result.content)
            orders = json.loads(result.content).get("orders", [])
            for order in orders if isinstance(orders, list) else []:
                if isinstance(order, dict) and order.get("order_id") not in (None, *order_ids):
                    order_ids.append(order["order_id"])
        for spec in specs:
            fields = spec.args_model.model_fields
            if "customer_id" in fields or "order_id" not in fields:
                continue
            for order_id in order_ids:
                result = execute(registry, engine, ctx, spec.name, {"order_id": order_id})
                if result.ok:
                    outputs.append(result.content)
    finally:
        engine.dispose()
    return outputs


def validate_task(
    task: Task,
    seed_engine: Engine,
    registry: Registry,
    read_tools: Sequence[str] = DEFAULT_READ_TOOLS,
) -> list[str]:
    """Problems of a task file against the seed data and the tools. An empty list means the task is fine."""
    problems: list[str] = []
    with seed_engine.connect() as conn:
        customers = db.Customer.__table__
        known = conn.execute(select(customers.c.id).where(customers.c.id == task.customer_id)).first()
    if known is None:
        problems.append(f"{task.id}: customer {task.customer_id} is not in the seed data")
    problems += _seed_time_problems(task, seed_engine)

    scenario = [task.user.all_text()]
    for required in task.required_values:
        if value_found(required, scenario):
            problems.append(
                f"{task.id}: required value {required.label!r} ({required.value}) "
                "is given away by the scenario"
            )

    try:
        gold, gold_outputs = _replay_gold(task, seed_engine, registry)
    except (GoldReplayError, ToolBugError) as error:
        problems.append(str(error))
        return problems  # the remaining checks build on the gold state
    try:
        problems += _forbidden_problems(task, gold, registry)
    finally:
        gold.dispose()

    try:
        outputs = _read_outputs(task, seed_engine, registry, read_tools) + gold_outputs
    except ToolBugError as error:
        problems.append(f"{task.id}: a read tool hit a bug: {error}")
        return problems
    for required in task.required_values:
        if not value_found(required, outputs):
            problems.append(
                f"{task.id}: required value {required.label!r} ({required.value}) is in no read tool output "
                "for this customer and in no gold action output"
            )
    return problems
