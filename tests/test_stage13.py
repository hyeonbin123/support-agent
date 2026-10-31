"""13단계(4B 에이전트의 남은 실패를 겨냥한 가드)의 하니스를 모델 없이 시험한다: E1(규정 거절 오류에 사실
붙이기), D1(같은 답 되풀이 가드), 실행기·서비스 설정, 선택·판정 분석, 등록한 규칙 문서."""

# [테스트 묶음] docs/experiments.md 13단계 규칙이 정한 것을 코드가 그대로 하는지 고정한다.
# - E1: tools.py의 고정 목록(기한·상태·쿠폰 거절)에만, P1에서 막힐 때만 오류 글 끝에 괄호 하나의 사실을
#   붙인다. 오류 코드와 이유 문장은 그대로이고, E0의 오류 글은 2026-10-15 전과 같다. P0에서는 아무것도 바뀌지
#   않는다.
# - D1: 도구 호출 없는 답이 바로 앞에 고객에게 전달한 답과 공백만 빼고 같으면 한 턴에 한 번 붙잡아
#   REPEAT_NOTICE로 다시 시키고, 다시도 같으면 그대로 전달한다. 붙잡은 답은 형식 오류로 세지 않는다.
# - 실행기·서비스: --errors·--repeat가 설정과 manifest·실행 이름에 남고, 서비스 설정 GUARD·ERRORS·REPEAT가
#   루프와 도구 문맥에 들어간다. 옛 manifest(새 축이 없음)와도 짝지어진다.
# - analyze guards·guards-verdict: 선택 조건 다섯과 판정표를 합성 기록으로 확인한다.
# - 문서: 등록한 안내문·사실 목록·문턱값이 docs/experiments.md 13단계에 그대로 있다.
# [방법] GPU·모델 없이 대본 제공자(ScriptedProvider), 작은 DB(conftest·factories), 임시 폴더의 합성 기록을
# 쓴다.

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from conftest import NOW
from factories import add_compensation_coupon, extend_tiny_rows, kst
from sqlalchemy.orm import Session
from test_agent import Harness
from test_service import engine, seeded_template  # noqa: F401 - fixture
from test_stage12 import start

from support_agent import analyze, db, guard_report, tools
from support_agent import episode as episode_module
from support_agent.agent import REPEAT_NOTICE, AgentState, is_repeat, last_delivered_reply
from support_agent.chat import Message, ScriptedProvider
from support_agent.config import FIRST_AGENT_MESSAGE, RunConfig
from support_agent.episode import run_episode
from support_agent.paths import ROOT
from support_agent.run import summarise
from support_agent.seed import build_seed_engine
from support_agent.service import core
from support_agent.service.core import ChatService
from support_agent.service.settings import Settings
from support_agent.tasks import load_tasks
from support_agent.toolkit import ConversationState, ToolContext, ToolResult, execute
from support_agent.tools import build_registry
from support_agent.user_sim import ScriptedUser

REGISTRY = build_registry()
EXPERIMENTS = (ROOT / "docs" / "experiments.md").read_text(encoding="utf-8")
STAGE13 = EXPERIMENTS[EXPERIMENTS.index("## 13단계") :]


# ==================================================================== E1: 규정 거절 오류에 사실 붙이기


# 작은 가게(tests/factories.py: O-1 결제완료, O-2 배송중(출고 기록 없음), O-3 9월 8일 수령, O-4 9월 2일 수령,
# O-5 배송준비중)를 새로 만든다.
@pytest.fixture
def shop(tiny_engine):
    store = db.memory_engine(tiny_engine)
    with Session(store) as session:
        extend_tiny_rows(session)
        session.commit()
    yield store
    store.dispose()


# 고객 C-1로 본인 확인된 대화에서 도구 하나를 부른다. facts는 E1, p1은 규정을 막는지.
def call(store, name: str, *, facts: bool = True, p1: bool = True, **args: Any) -> ToolResult:
    ctx = ToolContext(
        now=NOW, enforce_policy=p1, state=ConversationState(verified_customer_id="C-1"), error_facts=facts
    )
    return execute(REGISTRY, store, ctx, name, args)


# seed DB는 만드는 데 1초 넘게 걸려 이 파일에서 한 번만 만든다(에피소드는 그 복사본을 쓴다).
@cache
def seed_engine():
    return build_seed_engine()


# 코드마다 E1의 오류 글 = E0의 오류 글 + " " + 괄호 하나의 사실. 사실은 tools.py의 고정 목록 그대로다.
@pytest.mark.parametrize(
    ("name", "args", "code", "facts"),
    [
        (
            "request_return",
            {"order_id": "O-4", "line_nos": [1], "reason": "changed_mind"},
            "return_window_expired",
            "(수령: 2026-09-02 14:20, 접수 마감일: 2026-09-09)",
        ),
        (
            "request_exchange",
            {"order_id": "O-4", "line_no": 1, "new_variant_id": "V-1-02", "reason": "changed_mind"},
            "exchange_window_expired",
            "(수령: 2026-09-02 14:20, 접수 마감일: 2026-09-09)",
        ),
        (
            "request_return",
            {"order_id": "O-5", "line_nos": [1], "reason": "changed_mind"},
            "return_not_delivered",
            "(주문 상태: 배송준비중)",
        ),
        (
            "cancel_order",
            {"order_id": "O-3", "reason": "changed_mind"},
            "cancel_not_allowed_status",
            "(주문 상태: 배송완료, 출고: 2026-09-06 09:00, 수령: 2026-09-08 15:00)",
        ),
        (
            "change_shipping_address",
            {"order_id": "O-2", "address_id": "AD-C-1-2"},
            "address_change_not_allowed_status",
            "(주문 상태: 배송중)",
        ),
        (
            "issue_compensation_coupon",
            {"order_id": "O-3", "reason": "delivery_delay"},
            "coupon_not_eligible",
            "(도착 예정일: 2026-09-09, 수령: 2026-09-08 15:00)",
        ),
        (
            "issue_compensation_coupon",
            {"order_id": "O-5", "reason": "delivery_delay"},
            "coupon_not_eligible",
            "(도착 예정일: 2026-09-16, 수령: 아직 수령 전)",
        ),
    ],
)
def test_e1_appends_the_facts_the_check_used(shop, name, args, code, facts):
    plain = call(shop, name, facts=False, **args)
    with_facts = call(shop, name, **args)
    assert (plain.error_code, with_facts.error_code) == (code, code)
    assert plain.policy_blocked and with_facts.policy_blocked
    assert with_facts.content == f"{plain.content} {facts}"
    assert facts not in plain.content


# 이미 발급된 쿠폰(주문당 1장)과 최근 30일 2장 한도: 그 쿠폰들의 번호와 발급 시각을 붙인다.
def test_e1_lists_the_coupons_behind_a_coupon_refusal(shop):
    with Session(shop) as session:
        add_compensation_coupon(session, "CP-O-1-1", "C-1", "O-1", issued_at=kst(9, 10))
        add_compensation_coupon(session, "CP-O-4-1", "C-1", "O-4", issued_at=kst(9, 3, 11))
        session.commit()
    again = call(shop, "issue_compensation_coupon", order_id="O-1", reason="delivery_delay")
    assert again.error_code == "coupon_already_issued_for_order"
    assert again.content.endswith("(이 주문의 보상 쿠폰: CP-O-1-1 발급 2026-09-10 09:00)")
    third = call(shop, "issue_compensation_coupon", order_id="O-2", reason="delivery_delay")
    assert third.error_code == "coupon_limit_exceeded"
    assert third.content.endswith(
        "(최근 30일 보상 쿠폰: CP-O-1-1 발급 2026-09-10 09:00, CP-O-4-1 발급 2026-09-03 11:00)"
    )


# P0은 규정을 막지 않으므로 E1이어도 호출이 그대로 통과하고 위반만 기록된다.
def test_e1_changes_nothing_under_p0(shop):
    result = call(shop, "request_return", p1=False, order_id="O-4", line_nos=[1], reason="changed_mind")
    assert result.ok and result.violations == ("return_window_expired",)


# 목록 밖의 거절(다른 상품으로 교환)과 무결성 오류(이미 취소된 주문)는 E1이어도 글이 같다.
def test_e1_leaves_other_errors_alone(shop):
    for name, args in (
        (
            "request_exchange",
            {"order_id": "O-3", "line_no": 1, "new_variant_id": "V-2-01", "reason": "changed_mind"},
        ),
        ("cancel_order", {"order_id": "O-6", "reason": "changed_mind"}),
    ):
        assert call(shop, name, **args).content == call(shop, name, facts=False, **args).content


# 사실은 막힐 때만 만든다: E0이거나 검사를 통과하면 사실을 만드는 함수를 부르지 않는다(배송·쿠폰 행을 더 읽지
# 않는다).
def test_facts_are_built_only_for_an_e1_refusal(shop, monkeypatch):
    def boom(*_args):
        raise AssertionError("facts were built")

    monkeypatch.setattr(tools, "_status_facts", boom)
    monkeypatch.setattr(tools, "_window_facts", boom)
    assert call(
        shop, "request_return", facts=False, order_id="O-4", line_nos=[1], reason="changed_mind"
    ).error_code
    assert call(shop, "request_return", order_id="O-3", line_nos=[1], reason="changed_mind").ok


# 평가 에피소드는 config.errors로 도구 문맥의 error_facts를 정한다.
def test_the_episode_passes_e1_to_the_tool_context(monkeypatch):
    made: list[ToolContext] = []

    def recording(**fields):
        made.append(ToolContext(**fields))
        return made[-1]

    monkeypatch.setattr(episode_module, "ToolContext", recording)
    task = load_tasks("smoke")[0]
    for errors in ("E0", "E1"):
        run_episode(
            task,
            0,
            config=RunConfig(policy="P1", errors=errors),
            provider=ScriptedProvider(["무엇을 도와드릴까요?"]),
            user=ScriptedUser(["문의요"]),
            registry=REGISTRY,
            seed_engine=seed_engine(),
            policy_text="규정",
        )
    assert [ctx.error_facts for ctx in made] == [False, True]


# ==================================================================== D1: 같은 답 되풀이 가드


# 첫 턴에 "답 A"를 전달한 뒤, 다음 턴에 모델이 또 "답 A"를 내면 붙잡아 안내문을 넣고 다시 시킨다. 다시 낸 다른
# 답이 전달된다.
def test_d1_holds_a_repeated_reply_once_and_asks_again(tiny_engine):
    h = Harness(tiny_engine, ["답 A입니다.", "답  A입니다.\n", "다른 답입니다."], RunConfig(repeat="D1"))
    assert h.turn("첫 질문").reply == "답 A입니다."
    result = h.turn("같은 질문")
    assert result.reply == "다른 답입니다."
    held, notice, delivered = h.state.messages[-3:]
    assert (held.content, held.delivered) == ("답  A입니다.\n", False)
    assert (notice.content, notice.harness) == (REPEAT_NOTICE, True)
    assert delivered.content == "다른 답입니다."
    assert [c.format_error for c in h.state.llm_log] == [None, "repeat", None]
    assert h.state.repeats == 1
    # 다시 시킬 때 온도를 올리지 않는다
    assert {r["temperature"] for r in h.provider.requests} == {0.0}


# 다시 시켜도 같은 답이면(턴당 1번 예산을 다 씀) 그대로 전달한다.
def test_d1_delivers_the_same_reply_after_its_one_retry(tiny_engine):
    h = Harness(tiny_engine, ["답 A", "답 A", "답 A"], RunConfig(repeat="D1"))
    h.turn("첫 질문")
    assert h.turn("같은 질문").reply == "답 A"
    assert [c.format_error for c in h.state.llm_log] == [None, "repeat", None]


# D0(기본)은 같은 답을 그대로 전달한다.
def test_d0_delivers_repeats(tiny_engine):
    h = Harness(tiny_engine, ["답 A", "답 A"])
    h.turn("첫 질문")
    assert h.turn("같은 질문").reply == "답 A"
    assert h.state.repeats == 0


# 비교 대상은 고객에게 전달된 마지막 상담원 글이다: 붙잡혀 전달되지 않은 답과 도구 호출은 건너뛴다.
def test_the_repeat_check_compares_with_the_last_delivered_reply():
    messages = [
        Message("assistant", FIRST_AGENT_MESSAGE),
        Message("user", "질문"),
        Message("assistant", "전달된 답"),
        Message("user", "다시 질문"),
        Message("assistant", "붙잡힌 답", delivered=False),
        Message("user", "[시스템 안내] ...", harness=True),
    ]
    assert last_delivered_reply(messages) == "전달된 답"
    assert is_repeat("전달된   답", messages)
    assert not is_repeat("붙잡힌 답", messages)
    assert last_delivered_reply([]) is None and not is_repeat("답", [])


# 붙잡은 되풀이는 형식 오류가 아니다: summary.json의 format_errors에 들지 않고 held_repeats로 따로 센다.
def test_held_repeats_are_counted_apart_from_format_errors():
    task = load_tasks("smoke")[0]
    result = run_episode(
        task,
        0,
        config=RunConfig(repeat="D1"),
        provider=ScriptedProvider(["어떤 주문인가요?", "어떤 주문인가요?", "주문 번호를 알려 주세요."]),
        user=ScriptedUser(["주문 문의요", "주문 문의요"]),
        registry=build_registry(),
        seed_engine=seed_engine(),
        policy_text="규정",
    )
    summary = summarise([result])
    assert (summary["format_errors"], summary["held_repeats"]) == (0, 1)
    assert "repeat" in analyze._NOT_A_FORMAT_ERROR


# 서비스가 저장하는 상태: D1을 쓰지 않으면 repeats 키가 없어 2026-10-15 전과 같고, 있으면 되살린다.
def test_the_saved_state_keeps_repeats_only_when_used():
    state = AgentState([Message("system", "s")])
    assert "repeats" not in state.to_dict()
    state.repeats = 2
    assert core._state_from(state.to_dict()).repeats == 2


# ==================================================================== 실행기와 서비스 설정


# --errors·--repeat가 설정(manifest config)에 남고, 기본값이 아니면 실행 이름에 C 축처럼 붙는다.
def test_the_runner_records_the_new_axes(tmp_path, monkeypatch):
    _, manifest = start(
        tmp_path, monkeypatch, "--claims", "C1", "--guard", "G1", "--errors", "E1", "--repeat", "D1"
    )
    config = manifest["config"]
    assert (config["claims"], config["guard"], config["errors"], config["repeat"]) == ("C1", "G1", "E1", "D1")
    assert manifest["run_id"].endswith("-P0-R0-G1-F0-C1-E1-D1-unit")


def test_the_default_runner_name_is_unchanged(tmp_path, monkeypatch):
    _, manifest = start(tmp_path, monkeypatch)
    assert manifest["run_id"].endswith("-P0-R0-G0-F0-unit")
    assert (manifest["config"]["errors"], manifest["config"]["repeat"]) == ("E0", "D0")


# 서비스 설정 GUARD·ERRORS·REPEAT가 서비스의 RunConfig와 도구 문맥에 들어간다. 기본은 모두 끔이다.
def test_service_settings_carry_the_stage13_axes(engine):  # noqa: F811
    on = Settings.from_env(
        {"SUPPORT_AGENT_GUARD": "G1", "SUPPORT_AGENT_ERRORS": "E1", "SUPPORT_AGENT_REPEAT": "D1"}
    )
    service = ChatService(on, engine, ScriptedProvider([]))
    assert (service.config.guard, service.config.errors, service.config.repeat) == ("G1", "E1", "D1")
    assert service._context(None).error_facts is True
    default = ChatService(Settings(), engine, ScriptedProvider([]))
    assert (default.config.guard, default.config.errors, default.config.repeat) == ("G0", "E0", "D0")
    assert default._context(None).error_facts is False


# 새 축이 생기기 전의 manifest는 그 축을 기본값으로 보고 짝지어진다. 13단계 팔은 네 축을 무시하고 짝짓는다.
def test_old_manifests_pair_with_stage13_arms(tmp_path):
    old = synthetic_run(tmp_path, "old", {}, [], drop_new_axes=True)
    e1 = synthetic_run(tmp_path, "e1", {"errors": "E1"}, [])
    assert analyze.setup_differences(old, e1, ignore=tuple(guard_report.GUARD_AXES)) == []
    assert analyze.setup_differences(old, e1) == ["config.errors: 'E0' != 'E1'"]


# ==================================================================== analyze guards·guards-verdict


TYPES = analyze.load_task_types()
DEV_TASKS = [t.id for t in load_tasks("dev")]


# 합성 에피소드 하나: 판정 요소와 보고 지표에 필요한 칸만 채운다.
def episode(task_id: str, trial: int, success: bool, *, writes: int = 0, ending: str = "user_stop") -> dict:
    return {
        "task_id": task_id,
        "trial": trial,
        "status": "completed" if ending in ("user_stop", "handoff", "max_user_turns") else "truncated",
        "termination": ending,
        "verdict": {
            "db_match": success,
            "values": {},
            "unexpected_writes": writes,
            "policy_violations": [],
            "policy_blocks": [],
            "auth_blocks": 0,
        },
        "messages": [{"role": "assistant", "content": FIRST_AGENT_MESSAGE}],
        "llm_calls": [],
        "tool_calls": [],
        "wall_seconds": 10.0,
    }


# 합성 실행 폴더: 기준 구성(8단계 M2b의 설정)에 축만 바꾼 manifest와, 과제마다 4회의 에피소드.
def synthetic_run(
    tmp_path: Path, name: str, axes: dict, wins: list[tuple[str, int]], *, drop_new_axes: bool = False,
    writes: int = 0, endings: dict | None = None,
) -> Path:  # fmt: skip
    folder = tmp_path / name
    folder.mkdir()
    config = {
        **RunConfig(model="qwen3.5:4b", policy="P1", num_ctx=12288, user_num_ctx=12288).to_dict(),
        **axes,
    }
    if drop_new_axes:
        for key in ("errors", "repeat", "max_repeat_retries"):
            config.pop(key)
    manifest = {
        "config": config,
        "trials": 4,
        "task_sha256": {t: "h" for t in DEV_TASKS},
        "seed_hash": "s",
        "prompt_sha256": {"policy": "p"},
        "agent_provider": {"provider": "ollama", "model": "qwen3.5:4b", "think": False},
        "user_provider": {"provider": "ollama", "model": "qwen2.5:7b-instruct"},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    won = {(t, k) for t, k in wins}
    endings = endings or {}
    lines = [
        json.dumps(
            episode(t, k, (t, k) in won, writes=writes if (t, k) == (DEV_TASKS[0], 0) else 0,
                    ending=endings.get((t, k), "user_stop"))
        )
        for t in DEV_TASKS
        for k in range(4)
    ]  # fmt: skip
    (folder / "episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


def of_type(kind: str) -> list[str]:
    return [t for t in DEV_TASKS if TYPES[t] == kind]


# 기준: 과제마다 시도 0·1 성공(50%). 팔은 거기에 얹는다.
BASE_WINS = [(t, k) for t in DEV_TASKS for k in (0, 1)]


def plus(extra: list[tuple[str, int]], minus: list[tuple[str, int]] = ()) -> list[tuple[str, int]]:
    return [w for w in BASE_WINS if w not in minus] + extra


def test_guards_select_one_arm_that_meets_every_condition(tmp_path):
    base = synthetic_run(tmp_path, "base", {}, BASE_WINS)
    c1 = synthetic_run(tmp_path, "c1", {"claims": "C1"}, plus([(t, 2) for t in DEV_TASKS[:3]]))  # +3/96
    g1 = synthetic_run(tmp_path, "g1", {"guard": "G1"}, plus([(t, 2) for t in DEV_TASKS[:2]]))  # +2/96
    report, chosen = guard_report.select(base, [c1, g1])
    assert chosen == "C1"
    assert "서비스 CLAIMS=C1 기본값 조건" in report and "만족 → 켠다" in report


# 쓰기(기준 0 + 2 초과)나 형식 오류·완료 주장 종료(2 초과)를 어기면 고르지 않는다.
def test_guards_reject_extra_writes_and_endings(tmp_path):
    base = synthetic_run(tmp_path, "base", {}, BASE_WINS)
    gain = plus([(t, 2) for t in DEV_TASKS[:6]])
    writes = synthetic_run(tmp_path, "c1", {"claims": "C1"}, gain, writes=3)
    endings = {(DEV_TASKS[-1], 3): "unbacked_claim", (DEV_TASKS[-2], 3): "unbacked_claim",
               (DEV_TASKS[-3], 3): "agent_format_error"}  # fmt: skip
    ended = synthetic_run(tmp_path, "d1", {"repeat": "D1"}, gain, endings=endings)
    _, chosen = guard_report.select(base, [writes, ended])
    assert chosen is None


# E1은 거절 유형만 오르고 다른 유형이 한 에피소드라도 내려가면 고르지 않는다.
def test_guards_reject_e1_when_other_types_drop(tmp_path):
    base = synthetic_run(tmp_path, "base", {}, BASE_WINS)
    refusals = [(t, 2) for t in of_type("refusal")]
    dropped = plus(refusals, minus=[(of_type("lookup")[0], 0)])
    e1 = synthetic_run(tmp_path, "e1", {"errors": "E1"}, dropped)
    report, chosen = guard_report.select(base, [e1])
    assert chosen is None and "거절이 아닌 유형 pass^1 ≥ 기준 아니오" in report
    kept = synthetic_run(tmp_path, "e1b", {"errors": "E1"}, plus(refusals))
    assert guard_report.select(base, [kept])[1] == "E1"


# 단일 팔 둘 이상이 조건을 만족하면 합친 팔을 요구하고, 합친 팔이 조건을 만족하고 가장 좋은 단일 팔 이상이면
# 합친 팔을, 아니면 가장 좋은 단일 팔을 고른다.
def test_guards_ask_for_and_judge_the_combined_arm(tmp_path):
    base = synthetic_run(tmp_path, "base", {}, BASE_WINS)
    c1 = synthetic_run(tmp_path, "c1", {"claims": "C1"}, plus([(t, 2) for t in DEV_TASKS[:4]]))
    d1 = synthetic_run(tmp_path, "d1", {"repeat": "D1"}, plus([(t, 2) for t in DEV_TASKS[:5]]))
    report, chosen = guard_report.select(base, [c1, d1])
    assert chosen == "need:C1+D1" and "합친 팔(C1+D1)" in report
    good = synthetic_run(
        tmp_path, "both", {"claims": "C1", "repeat": "D1"}, plus([(t, 2) for t in DEV_TASKS[:6]])
    )
    assert guard_report.select(base, [c1, d1, good])[1] == "C1+D1"
    weak = synthetic_run(
        tmp_path, "weak", {"claims": "C1", "repeat": "D1"}, plus([(t, 2) for t in DEV_TASKS[:3]])
    )
    assert guard_report.select(base, [c1, d1, weak])[1] == "D1"


# 다른 설정(에이전트 모델 등)이 다르면 짝짓지 않는다.
def test_guards_refuse_runs_that_differ_elsewhere(tmp_path):
    base = synthetic_run(tmp_path, "base", {}, BASE_WINS)
    other = synthetic_run(tmp_path, "other", {"claims": "C1", "policy": "P0"}, BASE_WINS)
    report, chosen = guard_report.select(base, [other])
    assert chosen is None and report.startswith("짝지을 수 없음")


# 시험용 판정표: 구간 아래 끝 > 0이고 쓰기가 기준 + 3 이하면 개선, 쓰기가 넘으면 위험한 쓰기, 오르지 않으면
# 확인하지 못함.
def test_the_test_verdict_table(tmp_path):
    base = synthetic_run(tmp_path, "base", {}, BASE_WINS)
    better = plus([(t, k) for t in DEV_TASKS for k in (2, 3)][:60])
    assert guard_report.verdict(base, synthetic_run(tmp_path, "a", {"repeat": "D1"}, better))[1] == "개선"
    risky = synthetic_run(tmp_path, "b", {"repeat": "D1"}, better, writes=4)
    assert guard_report.verdict(base, risky)[1] == "성공률은 올랐지만 위험한 쓰기가 늘었음"
    flat = synthetic_run(tmp_path, "c", {"repeat": "D1"}, BASE_WINS)
    assert guard_report.verdict(base, flat)[1] == "개선을 확인하지 못함"


# 같은 답 되풀이로 끝난 에피소드: 전달된 상담원 답(첫 인사 제외)의 마지막 셋이 공백만 빼고 같다.
def test_a_loop_ending_is_three_equal_last_replies():
    def with_replies(*replies: str) -> dict:
        messages = [{"role": "assistant", "content": FIRST_AGENT_MESSAGE}]
        messages += [{"role": "assistant", "content": r} for r in replies]
        return {"messages": messages}

    assert guard_report.ends_in_a_loop(with_replies("가", "나", "나", " 나 "))
    assert not guard_report.ends_in_a_loop(with_replies("나", "나"))
    assert not guard_report.ends_in_a_loop(with_replies("나", "가", "나"))


# ==================================================================== 등록한 규칙 문서


# 안내문, E1 코드 목록, 문턱값이 docs/experiments.md 13단계에 측정 전에 적은 그대로 있다.
def test_the_registered_rules_match_the_code():
    assert REPEAT_NOTICE in STAGE13
    for code in guard_report.E1_CODES:
        assert f"`{code}`" in STAGE13
    for text in ("+3.1%p (96 에피소드 중 3)", "기준 + 2 (기준 3 → 5)", "합 ≤ 2", "기준 + 3 (13 → 16)",
                 "pass^1 차이 ≥ −5%p", "C1, G1, D1, E1 순"):  # fmt: skip
        assert text in STAGE13, text
    assert guard_report.MIN_GAIN == 3 / 96 and guard_report.DEV_WRITE_SLACK == 2
    assert guard_report.TEST_WRITE_SLACK == 3 and guard_report.C1_SERVICE_MIN_DIFF == -0.05
    # 13단계 네 팔의 순서는 등록한 그대로다. 15단계의 X1은 그 뒤에 붙는다(tests/test_stage15.py가 본다).
    assert guard_report.TIE_ORDER[:4] == ("C1", "G1", "D1", "E1")
    assert RunConfig().max_repeat_retries == 1
