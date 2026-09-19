"""What an episode leaves behind: call logs, the verdict and the JSONL record."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from support_agent.config import Termination


@dataclass
class ToolCallLog:
    agent_call: int  # index of the agent LLM call that produced this tool call
    name: str
    raw_arguments: dict[str, Any]  # as the model sent them
    args: dict[str, Any] | None  # canonical form, None if validation failed
    ok: bool
    error_code: str | None
    policy_blocked: bool
    violations: list[str]
    content: str
    write: bool
    ms: float


@dataclass
class LLMCallLog:
    who: Literal["agent", "user"]
    index: int  # call index within the episode, per role
    seed: int | None
    text: str
    tool_calls: list[dict[str, Any]]
    format_error: str | None  # empty | leaked_tool_call | cut_off, None when the reply was well formed
    dropped_text: str  # text that came with a tool call and was not delivered
    dropped_calls: int  # tool calls after the first one
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    wall_ms: float
    load_ms: float
    prompt_eval_ms: float
    eval_ms: float


@dataclass
class Verdict:
    success: bool
    judged: bool  # False when the episode did not end in a judged way (then success is False)
    db_match: bool
    values: dict[str, bool]  # required value label -> found in what the customer was told
    unexpected_writes: int  # successful write calls that are not in the gold actions (규정 위반 수)
    missing_writes: int
    policy_violations: list[str] = field(default_factory=list)  # codes let through under P0
    policy_blocks: list[str] = field(default_factory=list)  # codes refused under P1
    blocked_forbidden_attempts: int = 0  # the agent tried a forbidden call and the tool refused it
    auth_blocks: int = 0  # reads or writes refused because the customer was not verified
    db_diff: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    run_id: str
    task_id: str
    task_sha256: str
    trial: int
    status: Literal["completed", "truncated", "infra_error"]
    termination: Termination
    verdict: Verdict | None  # None for infra_error
    error: str  # exception text for infra_error, else ""
    messages: list[dict[str, Any]]  # agent-side history without the system prompt
    user_messages: list[dict[str, Any]]  # simulator-side history without the system prompt
    tool_calls: list[ToolCallLog]
    llm_calls: list[LLMCallLog]
    db_changes: dict[str, Any]  # diff_dumps(seed dump, final dump)
    seed_hash: str
    final_hash: str
    gold_hash: str
    wall_seconds: float

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
