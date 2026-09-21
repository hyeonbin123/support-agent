"""One episode: a simulated customer talks to the agent over a private DB copy, then it is judged."""

from __future__ import annotations

import re
import time
import traceback
import unicodedata

from sqlalchemy import Engine

from support_agent import db
from support_agent.agent import AgentState, agent_turn, build_system_prompt, new_state
from support_agent.chat import ChatProvider, ProviderError
from support_agent.config import (
    FIRST_AGENT_MESSAGE,
    JUDGED,
    RunConfig,
    Termination,
    derive_seed,
)
from support_agent.judge import gold_engine, judge
from support_agent.records import EpisodeResult
from support_agent.tasks import Task
from support_agent.toolkit import Registry, ToolBugError, ToolContext, execute
from support_agent.user_sim import SimulatorError, User
from support_agent.voice.channel import SpeechChannel

# What the agent is told when the recogniser heard nothing (a voice front end would say so too).
NOTHING_HEARD = "(고객의 말이 인식되지 않았습니다)"


def gold_dump_of(task: Task, seed_engine: Engine, registry: Registry) -> db.Dump:
    """The database a correct episode ends with. Computed once per task and reused for every trial."""
    engine = gold_engine(task, seed_engine, registry)
    try:
        return db.dump_db(engine)
    finally:
        engine.dispose()


_STOP = re.compile(r"#{2,}\s*STOP\s*#{2,}", re.IGNORECASE)
_OUT_OF_SCOPE = re.compile(r"#{2,}\s*OUT[-_ ]?OF[-_ ]?SCOPE\s*#{2,}", re.IGNORECASE)


def split_ending(user_text: str) -> tuple[str, Termination | None]:
    """The customer's words without the end token, and how the customer ended (tolerant of `### stop ###`)."""
    text = unicodedata.normalize("NFKC", user_text)
    for pattern, ending in ((_OUT_OF_SCOPE, "out_of_scope"), (_STOP, "user_stop")):
        if pattern.search(text):
            return pattern.sub("", text).strip(), ending
    return user_text, None


def _converse(
    state: AgentState,
    user: User,
    task: Task,
    trial: int,
    config: RunConfig,
    provider,
    registry,
    run_tool,
    channel: SpeechChannel | None = None,
    voice_log: list[dict] | None = None,
) -> Termination:
    agent_text = FIRST_AGENT_MESSAGE
    for _ in range(config.max_user_turns):
        user_text, ending = split_ending(user.reply(agent_text))
        if ending and not user_text:
            return ending
        if channel is not None:
            # Only the agent hears through the channel; the simulator remembers what it meant to say.
            index = len(voice_log) if voice_log is not None else 0
            heard = channel.hear(user_text, derive_seed(config.base_seed, task.id, trial, "voice", index))
            if voice_log is not None:
                voice_log.append(heard.to_dict())
            user_text = heard.text or NOTHING_HEARD
        # Text that came with the token ("네, 진행해 주세요 ###STOP###") still gets its answer;
        # otherwise a simulator habit would be counted as the agent's failure.
        turn = agent_turn(
            state,
            user_text,
            provider=provider,
            registry=registry,
            run_tool=run_tool,
            config=config,
            task_id=task.id,
            trial=trial,
        )
        if turn.stop:
            return turn.stop
        if ending:
            return ending
        agent_text = turn.reply or ""
    return "max_user_turns"


def run_episode(
    task: Task,
    trial: int,
    *,
    config: RunConfig,
    provider: ChatProvider,
    user: User,
    registry: Registry,
    seed_engine: Engine,
    policy_text: str,
    gold_dump: db.Dump | None = None,
    run_id: str = "",
    channel: SpeechChannel | None = None,
) -> EpisodeResult:
    started = time.perf_counter()
    if gold_dump is None:
        gold_dump = gold_dump_of(task, seed_engine, registry)
    seed_dump = db.dump_db(seed_engine)
    engine = db.memory_engine(seed_engine)  # private copy of this episode
    ctx = ToolContext(now=task.now, enforce_policy=config.enforce_policy)  # the task's fixed clock
    state = new_state(build_system_prompt(policy_text, task.now))

    def run_tool(name: str, arguments: dict):
        return execute(registry, engine, ctx, name, arguments)

    error = ""
    voice_log: list[dict] = []
    try:
        termination = _converse(
            state, user, task, trial, config, provider, registry, run_tool, channel, voice_log
        )
    except Exception as exc:  # noqa: BLE001
        # The model server, the simulator or our own code failed: nobody's task failure, and one broken
        # episode must not end a run that takes hours. KeyboardInterrupt still stops the run.
        termination, error = "infra_error", f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, ProviderError | SimulatorError | ToolBugError):
            error += "\n" + traceback.format_exc()

    final_dump = db.dump_db(engine)
    engine.dispose()
    verdict = None
    if termination != "infra_error":
        verdict = judge(
            task,
            final_dump=final_dump,
            gold_dump=gold_dump,
            messages=state.messages,
            tool_log=state.tool_log,
            termination=termination,
            registry=registry,
        )
    status = "infra_error" if verdict is None else ("completed" if termination in JUDGED else "truncated")
    return EpisodeResult(
        run_id=run_id,
        task_id=task.id,
        task_sha256=task.sha256(),
        trial=trial,
        status=status,
        termination=termination,
        verdict=verdict,
        error=error,
        messages=[m.to_dict() for m in state.messages if m.role != "system"],
        user_messages=[m.to_dict() for m in user.messages if m.role != "system"],
        tool_calls=state.tool_log,
        llm_calls=[*state.llm_log, *user.llm_log],
        db_changes=db.diff_dumps(seed_dump, final_dump),
        seed_hash=db.state_hash(seed_dump),
        final_hash=db.state_hash(final_dump),
        gold_hash=db.state_hash(gold_dump),
        wall_seconds=round(time.perf_counter() - started, 3),
        voice=voice_log,
    )
