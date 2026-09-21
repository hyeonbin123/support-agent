"""Settings of a run, the fixed strings of a conversation, and seed derivation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal

STOP_TOKEN = "###STOP###"  # the simulated customer is done
OUT_OF_SCOPE_TOKEN = "###OUT-OF-SCOPE###"  # the scenario does not tell the customer how to go on
FIRST_AGENT_MESSAGE = "안녕하세요, 고객센터입니다. 무엇을 도와드릴까요?"
HANDOFF_MESSAGE = "상담원에게 연결해 드리겠습니다. 잠시만 기다려 주세요."
THINK_TOOL = "think"

# How an episode ended. Only the first group is judged; the second group counts as failure;
# infra_error is nobody's failure and is left out of the denominator.
Termination = Literal[
    "user_stop",
    "out_of_scope",
    "handoff",
    "max_agent_calls",
    "max_user_turns",
    "too_many_tool_errors",
    "agent_format_error",
    "context_limit",
    "infra_error",
]
# max_user_turns is judged too: ending the conversation is the simulator's job, and an agent gains
# nothing by stalling (the database and what it said are judged as they are).
JUDGED: frozenset[str] = frozenset({"user_stop", "out_of_scope", "handoff", "max_user_turns"})


@dataclass(frozen=True)
class RunConfig:
    model: str = "qwen2.5:7b-instruct"
    user_model: str = "qwen2.5:7b-instruct"
    policy: Literal["P0", "P1"] = "P0"  # P0: rules only in the prompt, P1: tools also refuse
    reasoning: Literal["R0", "R1", "R2"] = "R0"  # R1: think tool, R2: confirmation enforced before writes
    # G1: a reply that only promises to act is sent back, not delivered. G2: and the retry is sampled.
    guard: Literal["G0", "G1", "G2"] = "G0"
    rescue: Literal["F0", "F1"] = "F0"  # F1: a tool call leaked into the text is parsed and run
    # V1: what the customer says reaches the agent through speech synthesis and recognition.
    # V2: and the recognised text goes through the rule-based normaliser first.
    voice: Literal["V0", "V1", "V2"] = "V0"
    # L1: a reply written in Chinese or Japanese script is not delivered but sent back, like a format
    # error. The service turns it on; it has not been measured, so the evaluation default is L0.
    language: Literal["L0", "L1"] = "L0"
    temperature: float = 0.0
    user_temperature: float = 0.3
    base_seed: int = 1000
    num_ctx: int = 16384
    max_tokens: int = 1024
    max_agent_calls: int = 30  # LLM calls by the agent per episode
    max_user_turns: int = 20
    max_tool_errors: int = 10
    stall_retry_temperature: float = 0.7  # G2 only
    max_stall_retries: int = 2  # per agent turn (G1); after that the reply is delivered as it is
    max_format_retries: int = 2  # per agent turn; 0 reproduces the strict "one malformed reply fails" rule

    @property
    def enforce_policy(self) -> bool:
        return self.policy == "P1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_seed(base_seed: int, task_id: str, trial: int, role: str, call_index: int) -> int:
    """A stable 31-bit seed per LLM call. Independent of process, platform and hash randomisation."""
    text = f"{base_seed}|{task_id}|{trial}|{role}|{call_index}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF
