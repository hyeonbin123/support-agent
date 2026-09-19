"""Tool contracts: argument base class, context with the policy checks, the @tool decorator and execute().

execute() is the single path of every caller (agent loop, gold replay, later the service and the MCP server):
look up -> validate -> before_write hook -> handler in one DB session -> commit or roll back.
"""

from __future__ import annotations

import copy
import inspect
import json
import re
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from sqlalchemy import Engine
from sqlalchemy.orm import Session

ERROR_PREFIX = "Error"
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ToolArgs(BaseModel):
    """Base class of every tool argument model. Invented arguments are rejected."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _storable_text(cls, data: Any) -> Any:
        """Lone surrogates and NUL bytes would crash the database driver or the report writer."""

        def check(value: Any) -> None:
            if isinstance(value, str):
                if "\x00" in value:
                    raise ValueError("text must not contain NUL characters")
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError as error:
                    raise ValueError("text is not valid Unicode") from error
            elif isinstance(value, dict):
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)

        check(data)
        return data


class ToolError(Exception):
    """An expected failure that is reported back to the model as `Error: [code] message`."""

    def __init__(self, code: str, message: str, *, policy: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.policy = policy  # True when a policy rule (P axis) refused the call


class ToolBugError(RuntimeError):
    """A handler raised something unexpected. This is our bug, never the agent's failure."""


@dataclass
class ConversationState:
    """What tools remember across calls of one conversation. The service will store this per chat session."""

    verified_customer_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"verified_customer_id": self.verified_customer_id}


@dataclass
class ToolContext:
    """Per-conversation input of every tool call. Evaluation injects a fixed `now`."""

    now: datetime  # timezone-aware
    enforce_policy: bool = True  # P1 = True, P0 = False
    state: ConversationState = field(default_factory=ConversationState)
    violations: list[str] = field(default_factory=list)  # policy codes that were let through (P0 only)

    def require(self, ok: bool, code: str, message: str) -> None:
        """Integrity and identity checks. Refused in every mode."""
        if not ok:
            raise ToolError(code, message)

    def check_policy(self, ok: bool, code: str, message: str) -> None:
        """Policy rules that code can verify. P1 refuses; P0 records the violation and lets the call pass."""
        if ok:
            return
        if self.enforce_policy:
            raise ToolError(code, message, policy=True)
        self.violations.append(code)


Handler = Callable[[Session, ToolContext, Any], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[ToolArgs]
    handler: Handler
    write: bool  # True = changes the DB (confirmation, approval and the MCP write scope apply)
    unordered_args: tuple[str, ...] = ()  # list arguments whose order carries no meaning
    terminates: bool = False  # True = the conversation ends after a successful call (hand-off)
    uncompared_args: tuple[str, ...] = ()  # free text and the like: ignored when calls are compared

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": llm_schema(self.args_model)}

    def __call__(self, session: Session, ctx: ToolContext, args: Any) -> dict[str, Any]:
        return self.handler(session, ctx, args)


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str  # JSON text on success, "Error: [code] message" on failure
    error_code: str | None = None  # unknown_tool | invalid_arguments | a code raised by the handler or a hook
    policy_blocked: bool = False
    args: dict[str, Any] | None = None  # canonical arguments (None when validation failed)
    violations: tuple[str, ...] = ()  # policy codes this call violated while being let through (P0)

    @staticmethod
    def error(
        code: str, message: str, *, policy: bool = False, args: dict[str, Any] | None = None
    ) -> ToolResult:
        return ToolResult(False, f"{ERROR_PREFIX}: [{code}] {message}", code, policy, args)


Registry = dict[str, ToolSpec]
BeforeWrite = Callable[[ToolSpec, dict[str, Any]], ToolResult | None]


def tool(
    *,
    write: bool,
    unordered_args: tuple[str, ...] = (),
    terminates: bool = False,
    uncompared_args: tuple[str, ...] = (),
) -> Callable[[Handler], ToolSpec]:
    """Turn `def name(session, ctx, args: SomeArgs) -> dict` with a docstring into a ToolSpec.

    The ToolSpec is callable with the same arguments, so tests may call `cancel_order(session, ctx, args)`.
    """

    def wrap(fn: Handler) -> ToolSpec:
        args_model = typing.get_type_hints(fn).get("args")
        if not (isinstance(args_model, type) and issubclass(args_model, ToolArgs)):
            raise TypeError(f"{fn.__name__}: annotate the args parameter with a ToolArgs subclass")
        if not fn.__doc__:
            raise TypeError(f"{fn.__name__}: the docstring is the tool description and is required")
        return ToolSpec(
            fn.__name__,
            inspect.cleandoc(fn.__doc__),
            args_model,
            fn,
            write,
            unordered_args,
            terminates,
            uncompared_args,
        )

    return wrap


def make_registry(*specs: ToolSpec) -> Registry:
    """Build a registry and refuse tool definitions that small models or the MCP SDK handle badly."""
    registry: Registry = {}
    for spec in specs:
        if not _NAME_RE.match(spec.name):
            raise ValueError(f"bad tool name: {spec.name!r}")
        if spec.name in registry:
            raise ValueError(f"duplicate tool name: {spec.name}")
        if spec.args_model.model_config.get("extra") != "forbid":
            raise ValueError(f"{spec.name}: the args model must forbid extra fields")
        schema = llm_schema(spec.args_model)
        if not schema.get("required"):
            # Ollama's generic tool-call parser cannot handle a call whose arguments are all optional.
            raise ValueError(f"{spec.name}: at least one argument must be required")
        flat = json.dumps(schema)
        if "$ref" in flat or "anyOf" in flat:
            raise ValueError(f"{spec.name}: the argument schema must be flat (no $ref, no anyOf)")
        for arg in (*spec.unordered_args, *spec.uncompared_args):
            if arg not in spec.args_model.model_fields:
                raise ValueError(f"{spec.name}: unordered_args/uncompared_args name an unknown field {arg!r}")
        registry[spec.name] = spec
    return registry


def llm_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Flatten pydantic's JSON Schema for small models: inline $ref, fold Optional, drop titles."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(x) for x in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]])
            return walk({**target, **{k: v for k, v in node.items() if k != "$ref"}})
        if "anyOf" in node:
            non_null = [x for x in node["anyOf"] if x != {"type": "null"}]
            if len(non_null) == 1:
                return walk({**non_null[0], **{k: v for k, v in node.items() if k != "anyOf"}})
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "title" or (key == "default" and value is None):
                continue
            out[key] = {k: walk(v) for k, v in value.items()} if key == "properties" else walk(value)
        return out

    flat = walk(schema)
    flat.pop("description", None)  # the docstring of the args model is not part of the schema
    return flat


def short_errors(error: ValidationError, limit: int = 3) -> str:
    items = error.errors(include_url=False, include_input=False)
    parts = [(".".join(map(str, e["loc"])) or "(root)") + ": " + e["msg"] for e in items[:limit]]
    more = len(items) - limit
    return "; ".join(parts) + (f" (+{more} more)" if more > 0 else "")


def canonical_args(spec: ToolSpec, args: ToolArgs) -> dict[str, Any]:
    """Validated arguments in a comparable form: JSON types, no None, order-free lists sorted."""
    out = args.model_dump(mode="json", exclude_none=True)
    for name in spec.unordered_args:
        if isinstance(out.get(name), list):
            out[name] = sorted(out[name])
    return out


def execute(
    registry: Registry,
    engine: Engine,
    ctx: ToolContext,
    name: str,
    raw_args: dict[str, Any],
    *,
    before_write: BeforeWrite | None = None,
) -> ToolResult:
    """Run one tool call. Expected failures come back as ToolResult; a handler bug raises ToolBugError."""
    spec = registry.get(name)
    if spec is None:
        return ToolResult.error("unknown_tool", f"{name} 도구는 없습니다.")
    try:
        args = spec.args_model.model_validate(raw_args)
    except ValidationError as error:
        return ToolResult.error("invalid_arguments", f"인자가 잘못되었습니다. {short_errors(error)}")
    clean = canonical_args(spec, args)
    if spec.write and before_write is not None:
        refusal = before_write(spec, clean)
        if refusal is not None:
            return refusal
    seen = len(ctx.violations)
    saved_state = copy.copy(ctx.state)
    with Session(engine) as session:
        try:
            result = spec.handler(session, ctx, args)
            content = json.dumps(result, ensure_ascii=False)
            session.commit()
        except ToolError as error:
            session.rollback()
            ctx.state = saved_state
            del ctx.violations[seen:]  # a refused call changed nothing, so it violated nothing
            return ToolResult.error(error.code, error.message, policy=error.policy, args=clean)
        except Exception as error:
            session.rollback()
            ctx.state = saved_state
            raise ToolBugError(f"{name}({clean}) raised {type(error).__name__}: {error}") from error
    return ToolResult(True, content, args=clean, violations=tuple(ctx.violations[seen:]))
