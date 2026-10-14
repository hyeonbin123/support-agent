"""에피소드 하나: 시뮬레이터 고객이 개인 DB 복사본 위에서 에이전트와 대화하고, 끝나면 판정한다."""

# [파일 역할] 에피소드 하나를 처음부터 끝까지 돌린다. seed DB의 개인 복사본을 만들고, 사용자 시뮬레이터와
# 에이전트가 번갈아 말하게 한 뒤, 끝난 DB와 대화를 판정해 EpisodeResult 하나를 만든다.
# [설계 이유] τ-bench 방식이다. 고객은 시나리오를 받은 LLM이 맡고, 에피소드마다 DB 복사본을 따로 써서 서로의
# 쓰기를 보지 않으며, 판정은 끝난 DB와 전달한 값만으로 한다(README 평가 방식). 한 에피소드의 실패(모델 서버
# 오류, 시뮬레이터 오류, 도구 버그)는 infra_error로 기록하고 몇 시간짜리 실행 전체를 멈추지 않는다.
# [흐름] gold 덤프 준비 → 복사본·ToolContext(과제의 고정 시각, P0/P1, V4 발신 번호)·시스템 프롬프트 → (R2면
# 확인 관문) → _converse로 대화 → 최종 덤프 → judge → 기록.
# [연결] run.py가 과제 × 시도마다 run_episode를 부르고, gold_dump_of를 과제마다 한 번 계산해 넘긴다.
# sim_report.py는 split_ending으로 시뮬레이터 말에서 종료 토큰을 뗀다.

from __future__ import annotations

import re
import time
import traceback
import unicodedata

# Session: V4에서 과제 고객의 등록 전화번호를 seed DB에서 읽는다(caller_number).
from sqlalchemy import Engine
from sqlalchemy.orm import Session

# 내부 모듈: DB 복사·덤프(db), 에이전트 루프(agent), 설정·종료 사유·seed(config), R2 확인 관문(confirm), 정답
# 재실행과 판정(judge), 기록(records), 과제(tasks), 도구 실행(toolkit), 시뮬레이터(user_sim), 음성
# 채널(voice.channel).
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
from support_agent.confirm import ConfirmGate, customer_turn
from support_agent.judge import gold_engine, judge
from support_agent.records import EpisodeResult
from support_agent.tasks import Task
from support_agent.toolkit import Registry, ToolBugError, ToolContext, execute
from support_agent.user_sim import SimulatorError, User
from support_agent.voice.channel import SpeechChannel

# 음성 조건에서 인식 결과가 비었을 때 에이전트에게 고객 말 대신 주는 고정 문장. 실제 음성 프런트엔드도
# 이렇게 알려 줄 것이라 같은 처리를 한다.
NOTHING_HEARD = "(고객의 말이 인식되지 않았습니다)"


# 정답 동작만 실행한 DB의 덤프. 판정의 db_match는 최종 덤프를 이것과 비교한다. 과제마다 한 번 계산해 모든
# 시도에 다시 쓴다(run.py). 임시 엔진은 덤프 뒤 바로 닫는다.
def gold_dump_of(task: Task, seed_engine: Engine, registry: Registry) -> db.Dump:
    """올바른 에피소드가 끝났을 때의 DB. 과제마다 한 번 계산해 모든 시도에 다시 쓴다."""
    engine = gold_engine(task, seed_engine, registry)
    try:
        return db.dump_db(engine)
    finally:
        engine.dispose()


# 음성 V4(9단계): 전화가 걸려 온 번호. 시나리오가 이메일만 알려 주는 과제도 고객은 등록한 전화로 건다고 본다.
# 이 번호가 ToolContext.caller_phone이 되고, verify_caller는 이 번호의 고객과 이름만 맞춘다.
def caller_number(task: Task, seed_engine: Engine) -> str:
    """음성 V4: 전화가 걸려 온 번호. 모든 과제의 고객은 자기가 등록한 전화로 건다. 시나리오가 이메일
    주소만 알려 주는 과제도 마찬가지다."""
    with Session(seed_engine) as session:
        customer = session.get(db.Customer, task.customer_id)
        if customer is None:
            raise ValueError(f"{task.id}: customer {task.customer_id} is not in the seed data")
        return customer.phone


# 시뮬레이터의 종료 토큰을 너그럽게 찾는다. '###STOP###'뿐 아니라 '### stop ###', 'OUT_OF_SCOPE'처럼 조금
# 다르게 써도 인정한다(docs/experiments.md 공통 규칙 판정). 시뮬레이터의 표기 버릇이 에이전트의 실패로
# 세어지지 않게 하려는 것이다.
_STOP = re.compile(r"#{2,}\s*STOP\s*#{2,}", re.IGNORECASE)
_OUT_OF_SCOPE = re.compile(r"#{2,}\s*OUT[-_ ]?OF[-_ ]?SCOPE\s*#{2,}", re.IGNORECASE)


# 고객의 말에서 종료 토큰을 떼고, 어떻게 끝냈는지(out_of_scope 또는 user_stop)를 함께 돌려준다. 토큰이 없으면
# 원래 말 그대로와 None이다.
# OUT-OF-SCOPE를 먼저 본다. 토큰이 있을 때 돌려주는 말은 NFKC로 정규화된 문장이다.
def split_ending(user_text: str) -> tuple[str, Termination | None]:
    """종료 토큰을 뗀 고객의 말과, 고객이 대화를 끝낸 방식(`### stop ###` 같은 변형도 받아들인다)."""
    text = unicodedata.normalize("NFKC", user_text)
    for pattern, ending in ((_OUT_OF_SCOPE, "out_of_scope"), (_STOP, "user_stop")):
        if pattern.search(text):
            return pattern.sub("", text).strip(), ending
    return user_text, None


# 대화 루프. 시뮬레이터가 상담원의 마지막 말에 답하고, 에이전트가 그 말에 한 턴 답하기를 고객 턴 한도(20)까지
# 되풀이한다. 돌려주는 값은 종료 사유다.
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
    # 첫 턴에 시뮬레이터가 듣는 것은 고정 인사말이다(에이전트 상태에도 같은 인사말이 들어 있다).
    agent_text = FIRST_AGENT_MESSAGE
    for _ in range(config.max_user_turns):
        user_text, ending = split_ending(user.reply(agent_text))
        # 종료 토큰만 왔으면 에이전트를 부르지 않고 바로 끝낸다.
        if ending and not user_text:
            return ending
        # 음성 조건(V1·V2·V4): 고객의 말을 음성 합성 → 인식에 거쳐, 에이전트에게는 인식된 글만 준다. seed는
        # (과제, 시도, 'voice', 발화 순번)에서 만들어 같은 과제·시도·순번의 발화는 언제나 같은 seed로
        # 합성된다. 발화마다 원문·합성·인식 결과와 시간을 voice_log에 남긴다.
        if channel is not None:
            # 채널을 거쳐 듣는 쪽은 에이전트뿐이다. 시뮬레이터는 자기가 하려던 말(원문)을 그대로 기억한다.
            index = len(voice_log) if voice_log is not None else 0
            heard = channel.hear(user_text, derive_seed(config.base_seed, task.id, trial, "voice", index))
            if voice_log is not None:
                voice_log.append(heard.to_dict())
            user_text = heard.text or NOTHING_HEARD
        # 토큰과 함께 온 말("네, 진행해 주세요 ###STOP###")에도 에이전트가 답하게 한다. 그러지 않으면
        # 시뮬레이터의 버릇이 에이전트의 실패로 세어진다.
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
        # 에이전트 쪽 종료(이관, 한도, 형식 오류 등)가 고객의 종료보다 먼저다. 그다음 고객이 토큰과 함께 말을
        # 남겼으면 에이전트가 한 번 답한 뒤 여기서 끝난다.
        if turn.stop:
            return turn.stop
        if ending:
            return ending
        agent_text = turn.reply or ""
    return "max_user_turns"


# 과제 하나의 시도 하나를 돌려 EpisodeResult를 만든다. 입력: 과제, 시도 번호, 설정, 에이전트 제공자,
# 시뮬레이터, 도구 등록부, seed DB(읽기 전용으로만 씀), 규정 문서. gold_dump를 넘기지 않으면 여기서
# 계산한다(테스트용).
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
    # V4의 세 요소(프롬프트 규칙 한 줄, find_customer 대신 verify_caller인 도구 목록, 발신 번호)는 함께 쓰거나
    # 모두 빼야 한다. 하나만 바뀌면 다른 실험이 되므로 시작하기 전에 막는다.
    if ("verify_caller" in registry) != config.caller_id:
        raise ValueError("voice V4 needs build_registry(caller_id=True), and only V4 may use it")
    if gold_dump is None:
        gold_dump = gold_dump_of(task, seed_engine, registry)
    # seed 덤프는 기록의 db_changes(seed → 최종)와 seed_hash에 쓰인다. 에피소드는 seed DB를 바이트 복사한 개인
    # 메모리 DB에서만 쓴다.
    seed_dump = db.dump_db(seed_engine)
    engine = db.memory_engine(seed_engine)  # 이 에피소드만 쓰는 개인 복사본
    ctx = ToolContext(  # 시각은 과제에 고정된 시계(task.now)
        now=task.now,
        enforce_policy=config.enforce_policy,
        caller_phone=caller_number(task, seed_engine) if config.caller_id else None,
    )
    state = new_state(build_system_prompt(policy_text, task.now, caller_id=config.caller_id))
    # R2(10단계)면 쓰기가 고객의 동의를 기다리는 확인 관문을 붙인다. 프롬프트와 도구 목록은 R0와 같다.
    gate = ConfirmGate() if config.reasoning == "R2" else None  # R2: 쓰기는 고객의 "네"를 기다린다

    # 에이전트 루프에 넘기는 도구 실행 함수. 루프는 DB와 ToolContext를 모르고 이 클로저만 부른다. R2에서는
    # 지금 고객 턴 번호와 마지막 고객 말을 관문에 넘겨 동의 여부를 판단하게 한다.
    def run_tool(name: str, arguments: dict):
        if gate is None:
            return execute(registry, engine, ctx, name, arguments)
        turn, said = customer_turn(state.messages)
        return gate.run(registry, engine, ctx, name, arguments, turn=turn, said=said)

    error = ""
    voice_log: list[dict] = []
    try:
        termination = _converse(
            state, user, task, trial, config, provider, registry, run_tool, channel, voice_log
        )
    except Exception as exc:  # noqa: BLE001
        # 모델 서버, 시뮬레이터, 또는 우리 코드가 실패한 경우다. 과제를 누가 실패한 것도 아니고, 망가진
        # 에피소드 하나가 몇 시간짜리 실행을 끝내면 안 된다. KeyboardInterrupt(Exception이 아님)는 여전히
        # 실행을 멈춘다.
        termination, error = "infra_error", f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, ProviderError | SimulatorError | ToolBugError):
            error += "\n" + traceback.format_exc()

    # infra_error여도 최종 덤프는 남긴다(기록의 db_changes). 판정은 infra_error가 아닐 때만 한다.
    # infra_error는 분모에서 빠지므로 판정이 없다(verdict None).
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
    # status: 판정하는 종료면 completed, 한도·형식 오류 등이면 truncated(판정은 하되 judged=False라 실패),
    # 판정이 없으면 infra_error.
    status = "infra_error" if verdict is None else ("completed" if termination in JUDGED else "truncated")
    # 기록 한 줄. 시스템 프롬프트는 manifest에 해시로 따로 남기므로 메시지 목록에서 뺀다. LLM 로그는
    # 에이전트와 시뮬레이터의 것을 합친다.
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
