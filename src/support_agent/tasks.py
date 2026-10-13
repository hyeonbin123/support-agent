"""Task files: what the simulated customer wants and what a correct ending looks like."""

# [파일 역할] 과제 파일(tasks/smoke.yaml, dev.yaml, test.yaml)의 형식을 pydantic 모델로 정의하고 읽어 들인다.
# 과제 하나는 "시뮬레이터 고객이 원하는 것"과 "올바르게 끝난 상태"(정답 쓰기 동작, 금지 동작, 고객에게 꼭
# 전달할 값)를 함께 적는다.
# [설계 이유] 판정에 LLM을 쓰지 않고 최종 DB 상태와 전달 값으로만 정하는 τ-bench 방식이라(README "평가 방식"),
# 과제에 정답을 기계가 확인할 수 있는 형태로 적어야 한다. 모든 모델이 extra="forbid"라 YAML의 오타 키는 읽을
# 때 바로 오류가 난다.
# [연결] 실행기(run.py), 에피소드(episode.py), 판정·과제 검증(judge.py), 분석(analyze.py, sim_report.py 등)이
# 쓴다. 다른 모듈이 모두 기대는 계약이라 고치면 전체 테스트를 돌린다.

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

# yaml: 과제 파일은 사람이 읽고 쓰기 쉬운 YAML이다. pydantic: 형식 검사와 타입 변환(날짜, 시간대 있는 시각).
import yaml
from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from support_agent.paths import TASKS


# 모르는 키를 거절하는 공통 기반 클래스.
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# 도구 호출 하나(도구 이름 + 인자). 정답 동작(gold_actions)의 한 줄이다.
class ToolAction(_Strict):
    tool: str
    args: dict[str, Any]


# 해서는 안 되는 호출과, P1(도구에서 규정을 막음)에서 그 호출을 막아야 하는 규정 코드. 과제 검증이 P1에서는 이
# 코드로 막히고 P0에서는 실행되어 DB가 바뀌는지 확인해, 과제가 P0와 P1을 실제로 가르는지 본다(docs/design.md).
class ForbiddenAction(ToolAction):
    expect_code: str  # the policy code that has to refuse this call under P1


# 고객에게 꼭 전달해야 하는 값. kind에 따라 비교 방식이 다르다: number는 숫자 표기(쉼표 무시), date는 여러
# 날짜 표기, text는 정규화 뒤 부분 문자열(judge.py, docs/design.md 판정).
class RequiredValue(_Strict):
    """Something the agent has to tell the customer. It must come from a tool result, never from the scenario
    or the policy text, so that an agent that did not look anything up cannot pass."""

    kind: Literal["number", "date", "text"]
    value: int | date | str
    label: str

    # kind와 값의 파이썬 타입이 맞는지 확인한다. YAML에서 "2026-09-02"는 날짜로, 47300은 정수로 읽히므로
    # kind를 잘못 적으면 여기서 걸린다.
    @model_validator(mode="after")
    def _kind_matches_value(self) -> RequiredValue:
        expected = {"number": int, "date": date, "text": str}[self.kind]
        if type(self.value) is not expected:
            raise ValueError(f"{self.label}: kind {self.kind} needs a {expected.__name__} value")
        return self


# 시뮬레이터 고객에게 주는 시나리오. known에는 쓰기 인자를 정하는 모든 선택(취소 사유 등)을 적어, 고객이
# 상담원의 질문에 답할 수 있게 한다. unknown은 고객이 모른다고 해야 하는 것이다.
class UserScenario(_Strict):
    persona: str = ""
    reason: str  # why the customer is calling
    known: str  # facts the customer may reveal, including every choice that decides a write argument
    unknown: str = ""  # facts the customer must say they do not know
    rules: str = ""  # how to react (what to ask, when to accept a refusal, when to stop)

    # 시나리오의 모든 글을 이어 붙인다. 과제 검증이 전달 값이 시나리오에 새어 있지 않은지 볼 때
    # 쓴다(judge.validate_task).
    def all_text(self) -> str:
        return "\n".join([self.persona, self.reason, self.known, self.unknown, self.rules])


# 과제 하나. type은 조회(lookup), 처리(action), 거절(refusal), 복합(composite) 네 유형이다.
# - now: 에피소드의 고정 시계. "수령 후 7일" 같은 규정이 언제 돌려도 같게 판정된다
# - customer_id: 시뮬레이터에게 보이지 않고, 정답 재실행에서 본인 확인된 고객으로 쓴다
# - gold_actions: 정답 쓰기 호출만, 유효한 순서 하나로 적는다. 판정은 새 DB에 이것만 실행한 상태와 에피소드가
#   끝난 DB를 비교한다
class Task(_Strict):
    id: str
    type: Literal["lookup", "action", "refusal", "composite"]
    purpose: str
    now: AwareDatetime  # fixed clock of the episode
    customer_id: str  # never shown to the simulator; used by the gold replay
    user: UserScenario
    gold_actions: list[ToolAction] = []  # write calls only, in one valid order
    gold_verified: bool = True  # False: the gold replay runs without a verified customer
    forbidden_actions: list[ForbiddenAction] = []
    required_values: list[RequiredValue] = []
    notes: str = ""

    # 유형별 모양 검사. 조회는 쓰기 정답이 없어야 하고, 조회·거절은 전달 값이 있어야 한다(없으면 아무것도 하지
    # 않는 에이전트가 통과한다). 거절은 막아야 할 호출을, 처리·복합은 정답 쓰기를 적어야 한다.
    @model_validator(mode="after")
    def _shape(self) -> Task:
        if self.type == "lookup" and self.gold_actions:
            raise ValueError(f"{self.id}: a lookup task changes nothing, so it has no gold actions")
        if self.type in ("lookup", "refusal") and not self.required_values:
            raise ValueError(
                f"{self.id}: {self.type} tasks need a required value, or doing nothing would pass"
            )
        if self.type == "refusal" and not self.forbidden_actions:
            raise ValueError(f"{self.id}: a refusal task names the call that must be refused")
        if self.type in ("action", "composite") and not self.gold_actions:
            raise ValueError(f"{self.id}: {self.type} tasks need gold actions")
        return self

    # 과제 내용의 해시. 키 순서를 고정한 JSON으로 만들어 같은 내용이면 같은 값이 된다. 실행마다
    # manifest.json에 남겨, 측정 뒤 과제 파일이 바뀌었는지와 두 실행을 짝지어도 되는지를 확인한다(run.py,
    # episode.py, sim_report.py, voice_real.py).
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()


# "dev" 같은 이름이면 tasks/dev.yaml을, 확장자가 있으면 그 경로를 읽는다. 과제 id가 겹치면 기록·판정이
# 섞이므로 거절한다.
def load_tasks(name_or_path: str | Path) -> list[Task]:
    """Load `tasks/<name>.yaml` (or a path). utf-8-sig tolerates a BOM left by Windows editors."""
    path = Path(name_or_path)
    if not path.suffix:
        path = TASKS / f"{name_or_path}.yaml"
    tasks = [Task.model_validate(item) for item in yaml.safe_load(path.read_text(encoding="utf-8-sig"))]
    ids = [task.id for task in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate task ids: {duplicates}")
    return tasks
