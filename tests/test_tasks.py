"""Every task file is checked without a GPU: the task itself, a scripted agent that follows the gold path
(must pass) and agents that do nothing useful (must fail)."""

# [파일 역할] tasks/ 아래 모든 과제 파일(smoke·dev·test YAML)을 GPU 없이 검사하는 테스트 묶음이다. 과제 자체의
# 형식·정답 실행 가능성, 정답 경로를 따르는 대본 에이전트의 통과, 쓸모없는 행동을 하는 대본 에이전트의 실패를
# 과제마다 확인한다.
# [왜 중요한가] 판정은 LLM 없이 최종 DB와 전달 값으로만 하므로(README "평가 방식"), 과제 파일이 틀리면 측정
# 수치 전체가 틀린다. '아무것도 안 하고 규정만 읊는 에이전트', '상담원에게 떠넘기는 에이전트'가 통과하지
# 못한다는 것을 과제마다 실제로 돌려 보는 것이 이 프로젝트가 말하는 '과제 파일 자체도 테스트한다'의 구현이다.
# [방법] 모델 대신 ScriptedProvider(정해진 응답을 차례로 재생), 고객 대신 ScriptedUser(정해진 말 뒤 STOP)를
# 써서 실제 run_episode와 실제 도구 등록부·seed DB로 에피소드를 끝까지 돌린다.

from __future__ import annotations

# sqlalchemy: seed DB에서 고객 이름·전화번호를 읽어 대본 에이전트의 본인 확인 인자로 쓴다.
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

# 실제 평가와 같은 부품을 그대로 쓴다: 에이전트 프롬프트용 규정 문서(load_policy), 대본 재생 제공자, 실행
# 설정, 에피소드 실행기, 과제 검증(validate_task), 과제 폴더 경로, seed DB, 실제 도구 등록부, 대본 고객.
from support_agent import db
from support_agent.agent import load_policy
from support_agent.chat import ScriptedProvider, ToolCall
from support_agent.config import RunConfig
from support_agent.episode import run_episode
from support_agent.judge import validate_task
from support_agent.paths import TASKS
from support_agent.seed import build_seed_engine
from support_agent.tasks import Task, load_tasks
from support_agent.tools import build_registry
from support_agent.user_sim import ScriptedUser

# 실제 도구 14개 등록부와 tasks/*.yaml의 모든 과제. 모듈을 불러올 때 한 번만 만들어 parametrize의 과제
# 목록으로 쓴다.
REGISTRY = build_registry()
ALL_TASKS = [task for path in sorted(TASKS.glob("*.yaml")) for task in load_tasks(path)]


# 고객 id → (이름, 전화번호) 표. 과제 파일에는 고객 id만 있고 시나리오에는 id를 보여 주지 않으므로, 대본
# 에이전트가 find_customer로 본인 확인을 할 값을 seed DB에서 직접 가져온다.
def _contacts() -> dict[str, tuple[str, str]]:
    """How the scripted agent verifies each customer: name and phone, straight from the seed data."""
    with Session(build_seed_engine()) as session:
        return {c.id: (c.name, c.phone) for c in session.scalars(select(db.Customer))}


# HANDOFF는 정답에 이관이 있는 과제를 '불필요한 이관' 테스트에서 빼는 데 쓰는 도구 이름이다.
CONTACTS = _contacts()
HANDOFF = "transfer_to_human"


# 과제의 전달 값(required_values)을 상담원이 말하는 문장으로 만든다. 판정의 값 일치 규칙(docs/design.md
# "판정")에 맞춰 금액은 '38,900원'처럼 쉼표 숫자, 날짜는 'M월 D일', 글은 그대로 넣는다. 전달 값이 없으면
# '처리되었습니다.'를 말한다.
def say_values(task: Task) -> str:
    parts = []
    for required in task.required_values:
        if required.kind == "number":
            parts.append(f"{required.label}은 {required.value:,}원입니다.")
        elif required.kind == "date":
            parts.append(f"{required.label}은 {required.value.month}월 {required.value.day}일입니다.")
        else:
            parts.append(f"{required.label}: {required.value}")
    return " ".join(parts) or "처리되었습니다."


# 대본(도구 호출과 답의 목록)으로 과제 하나를 첫 시도(trial 0)로 돌린다. 고객은 '문의드립니다.' 한 마디 뒤
# STOP하므로, 상담원의 첫 글 답이 나오면 대화가 끝나고 판정이 나온다. policy로 P0/P1을 바꿔 규정 검사 방식만
# 달리 돌릴 수 있다.
def episode(task: Task, script, policy: str = "P1"):
    return run_episode(
        task,
        0,
        config=RunConfig(policy=policy),
        provider=ScriptedProvider(script),
        user=ScriptedUser(["문의드립니다."]),
        registry=REGISTRY,
        seed_engine=build_seed_engine(),
        policy_text=load_policy(),
    )


# 과제 파일 전체에서 과제가 5개 이상 읽히고 과제 id가 겹치지 않는지 본다. id는 기록·seed 유도·짝지은 비교의
# 열쇠라 겹치면 결과가 섞인다.
def test_there_are_tasks():
    assert len(ALL_TASKS) >= 5 and len({task.id for task in ALL_TASKS}) == len(ALL_TASKS)


# validate_task가 문제를 하나도 찾지 않는지 과제마다 본다: 고객이 seed에 있음, now가 seed 시각보다 뒤, 전달
# 값이 시나리오에 새지 않음, 할인 있는 주문이 아님, 정답 동작이 P1에서 모두 성공, 금지 동작이 P1에서 막히고
# P0에서 DB를 바꿈, 전달 값이 읽기 도구나 정답 쓰기 도구의 출력에 있음.
# prompt_texts를 넘기지 않으므로 '전달 값이 에이전트 시스템 프롬프트에 새지 않음'은 여기서 보지 않는다.
# 그 검사는 같은 과제 전체에 대해 test_review_regressions.py(글 프롬프트)와 test_stage9.py(V4 프롬프트)가
# 돌린다.
@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda task: task.id)
def test_the_task_is_well_formed(task):
    assert validate_task(task, build_seed_engine(), REGISTRY) == []


# 본인 확인 → 정답 쓰기 동작 → 전달 값 말하기를 그대로 따르는 대본 에이전트가 과제마다 성공으로 판정되는지,
# 정답에 없는 쓰기·빠진 쓰기가 0인지 본다. 완벽한 에이전트가 통과하지 못하는 과제는 측정에서 누구도 풀 수
# 없으므로 반드시 막아야 한다.
@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda task: task.id)
def test_an_agent_that_follows_the_gold_path_passes(task):
    name, contact = CONTACTS[task.customer_id]
    script = [ToolCall("find_customer", {"name": name, "contact": contact})]
    script += [ToolCall(action.tool, action.args) for action in task.gold_actions]
    script.append(say_values(task))
    result = episode(task, script)
    assert result.status == "completed" and result.verdict.success, result.verdict
    assert result.verdict.unexpected_writes == 0 and result.verdict.missing_writes == 0


# 도구를 하나도 부르지 않고 규정만 읊는 에이전트가 모든 과제에서 실패하는지 본다. 거절 과제도 DB가 그대로인
# 것만으로는 통과하지 못하고, 조회해야만 알 수 있는 값을 말해야 한다는 설계를 확인한다.
@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda task: task.id)
def test_an_agent_that_only_talks_fails(task):
    result = episode(task, ["규정상 수령 후 7일이 지나면 반품이 어렵습니다. 도와드릴 수 없습니다."])
    assert result.status == "completed" and not result.verdict.success


# 정답에 이관이 없는 과제에서, 전달 값을 요약에 적어 바로 상담원에게 넘기는 에이전트가 실패하는지 본다. 이관
# 요약은 고객에게 전달된 말이 아니고 이관 행이 DB를 바꾸므로 성공할 수 없어야 한다. 떠넘기기로 점수를 얻는
# 길을 막는다.
@pytest.mark.parametrize(
    "task", [t for t in ALL_TASKS if all(a.tool != HANDOFF for a in t.gold_actions)], ids=lambda task: task.id
)
def test_a_needless_handoff_fails(task):
    script = [ToolCall("transfer_to_human", {"reason": "customer_request", "summary": say_values(task)})]
    result = episode(task, script)
    assert result.termination == "handoff" and not result.verdict.success


# 금지 동작이 있는 과제마다 같은 대본(금지 호출 → 정답 동작 → 값 말하기)을 P0와 P1로 돌려 본다.
# P0: 금지 호출이 통과해 DB가 어긋나 실패하고, 통과된 위반 코드가 과제의 expect_code 하나, 정답에 없는 쓰기가
# 1이어야 한다.
# P1: 같은 호출이 도구에서 막혀(blocked_forbidden_attempts 1) 나머지가 맞으면 성공해야 한다.
# 이 과제들이 P0와 P1을 실제로 가른다는 것, 즉 2단계 후보 P1(규정을 도구에서도 막음)의 효과를 잴 수 있다는
# 것을 확인한다.
@pytest.mark.parametrize("task", [t for t in ALL_TASKS if t.forbidden_actions], ids=lambda task: task.id)
def test_the_forbidden_call_passes_under_p0_and_fails_the_task(task):
    name, contact = CONTACTS[task.customer_id]
    forbidden = task.forbidden_actions[0]
    script = [
        ToolCall("find_customer", {"name": name, "contact": contact}),
        ToolCall(forbidden.tool, forbidden.args),
        *[
            ToolCall(action.tool, action.args) for action in task.gold_actions
        ],  # mixed tasks also have work to do
        say_values(task),
    ]
    p0 = episode(task, script, policy="P0")
    assert not p0.verdict.success and not p0.verdict.db_match
    assert p0.verdict.policy_violations == [forbidden.expect_code] and p0.verdict.unexpected_writes == 1
    p1 = episode(task, script, policy="P1")
    assert (
        p1.verdict.success and p1.verdict.blocked_forbidden_attempts == 1
    )  # blocked, then refused correctly


# 정답 경로 에피소드 하나로 기록과 격리를 함께 본다: 기록이 JSON 한 줄로 직렬화되고 DB 변경분이 남는지, 최종
# 해시 == 정답 해시 != seed 해시인지, 에피소드가 끝난 뒤에도 seed DB의 해시가 처음과 같은지(에피소드는
# 복사본에서만 쓴다). reports/의 기록으로 수치를 다시 계산하는 방식의 전제다.
def test_the_record_is_json_and_episodes_do_not_touch_the_seed():
    before = db.state_hash(db.dump_db(build_seed_engine()))
    task = next(t for t in ALL_TASKS if t.gold_actions)
    name, contact = CONTACTS[task.customer_id]
    script = [ToolCall("find_customer", {"name": name, "contact": contact})]
    script += [ToolCall(a.tool, a.args) for a in task.gold_actions] + [say_values(task)]
    result = episode(task, script)
    assert '"task_id"' in result.to_json_line() and result.db_changes
    assert result.final_hash == result.gold_hash != result.seed_hash == before
    assert db.state_hash(db.dump_db(build_seed_engine())) == before
