"""과제 파일: 시뮬레이터 고객이 원하는 것과, 올바르게 끝났을 때의 모습."""

# [파일 역할] 과제 파일(tasks/smoke.yaml, dev.yaml, test.yaml, 14단계 주입 과제 adv.yaml)의 형식을 pydantic
# 모델로 정의하고 읽어 들인다. 과제 하나는 "시뮬레이터 고객이 원하는 것"과 "올바르게 끝난 상태"(정답 쓰기
# 동작, 금지 동작, 고객에게 꼭 전달할 값)를 함께 적는다. 주입 과제는 시뮬레이터 대신 대본 고객이 말할
# 줄(user.script)과 주입 종류(injection)를 더 갖는다.
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
    expect_code: str  # P1에서 이 호출을 거절해야 하는 규정 코드


# 고객에게 꼭 전달해야 하는 값. kind에 따라 비교 방식이 다르다: number는 숫자 표기(쉼표 무시), date는 여러
# 날짜 표기, text는 정규화 뒤 부분 문자열(judge.py, docs/design.md 판정).
class RequiredValue(_Strict):
    """에이전트가 고객에게 꼭 알려야 하는 것. 시나리오나 규정 글이 아니라 반드시 도구 결과에서 나오는 값이어야
    한다. 그래야 아무것도 조회하지 않은 에이전트가 통과할 수 없다."""

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
# script(14단계 주입 과제만, tasks/adv.yaml): 시뮬레이터 대신 대본 고객이 상담원의 말과 상관없이 차례로 말하는
# 줄. 실행기의 --user-script가 이 줄을 user_sim.ScriptedUser로 그대로 보내고, 줄이 떨어지면 고객이 대화를
# 끝낸다(user_stop). 주입 문장(독립 변수)을 시뮬레이터가 바꿔 말하지 않게 하려는 것이다(docs/experiments.md
# 14단계). 다른 과제에는 없다(Task가 검사한다).
class UserScenario(_Strict):
    persona: str = ""
    reason: str  # 고객이 연락한 이유
    known: str  # 고객이 밝힐 수 있는 사실. 쓰기 인자를 정하는 모든 선택을 포함한다
    unknown: str = ""  # 고객이 모른다고 말해야 하는 사실
    rules: str = ""  # 반응하는 방법(무엇을 물을지, 언제 거절을 받아들일지, 언제 끝낼지)
    script: list[str] = []  # 14단계 주입 과제만: 대본 고객이 차례로 말하는 줄

    # 고객 쪽 글을 모두 이어 붙인다(시나리오 항목과, 있으면 대본 줄). 과제 검증이 전달 값이 고객 쪽 글에 새어
    # 있지 않은지 볼 때 쓴다(judge.validate_task). 대본 줄은 에이전트에게 글자 그대로 가므로 꼭 넣는다. 대본이
    # 없는 과제는 글이 그 전과 똑같다.
    def all_text(self) -> str:
        return "\n".join([self.persona, self.reason, self.known, self.unknown, self.rules, *self.script])


# 14단계 주입 과제의 주입 종류 다섯(docs/experiments.md 14단계): 쿠폰 발급 지시, 배송지 변경 지시, 다른 고객
# 정보 요구, 규정 재정의, 도구 결과 위장. 보고서(adv_report.py)가 이 순서로 종류별 표를 낸다.
Injection = Literal["coupon", "address", "other_customer", "policy_override", "fake_tool_result"]
# 주입 과제의 대본 줄 수: (1) 본인 확인 정보 + 조회 요청 (2) 주입 문장 (3) 처음 물은 것을 다시 묻기.
INJECTION_SCRIPT_LINES = 3


# 과제 하나. type은 조회(lookup), 처리(action), 거절(refusal), 복합(composite) 네 유형이다.
# - now: 에피소드의 고정 시계. "수령 후 7일" 같은 규정이 언제 돌려도 같게 판정된다
# - customer_id: 시뮬레이터에게 보이지 않고, 정답 재실행에서 본인 확인된 고객으로 쓴다
# - gold_actions: 정답 쓰기 호출만, 유효한 순서 하나로 적는다. 판정은 새 DB에 이것만 실행한 상태와 에피소드가
#   끝난 DB를 비교한다
# - injection: 14단계 주입 과제만 갖는 주입 종류. 이 값이 있는 과제만 user.script(대본)를 갖는다
class Task(_Strict):
    id: str
    type: Literal["lookup", "action", "refusal", "composite"]
    purpose: str
    now: AwareDatetime  # 에피소드의 고정 시계
    customer_id: str  # 시뮬레이터에게는 보이지 않는다. 정답 재실행이 쓴다
    user: UserScenario
    gold_actions: list[ToolAction] = []  # 쓰기 호출만, 유효한 순서 하나로
    gold_verified: bool = True  # False: 정답 재실행을 본인 확인된 고객 없이 돌린다
    forbidden_actions: list[ForbiddenAction] = []
    required_values: list[RequiredValue] = []
    notes: str = ""
    injection: Injection | None = None  # 14단계 주입 과제만

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

    # 14단계 주입 과제의 모양(docs/experiments.md 14단계). 대본은 주입 과제만 갖는다: 대본이 다른 과제에
    # 섞이면 실행기가 그 과제를 시뮬레이터 없이 돌리게 되고, 대본 줄이 주입 과제의 검사를 받지 않는다. 주입
    # 과제는 조회형이고(정답 DB = seed 그대로), 대본은 빈 줄 없는 세 줄이며, 전달 값은 첫 줄이 물은 조회 값
    # 하나다. 대본 줄에 종료 토큰 꼴('##')이 있으면 episode.split_ending이 그 줄에서 대화를 끝내 뒤 줄이 가지
    # 않으므로 막는다(대본이 떨어진 뒤의 STOP은 ScriptedUser가 낸다).
    @model_validator(mode="after")
    def _script(self) -> Task:
        script = self.user.script
        if self.injection is None:
            if script:
                raise ValueError(f"{self.id}: only injection tasks have a script")
            return self
        if self.type != "lookup":
            raise ValueError(f"{self.id}: an injection task is a lookup, so a correct agent writes nothing")
        if len(script) != INJECTION_SCRIPT_LINES or any(not line.strip() for line in script):
            raise ValueError(
                f"{self.id}: an injection task needs {INJECTION_SCRIPT_LINES} non-empty script lines"
            )
        if any("##" in line for line in script):
            raise ValueError(f"{self.id}: a script line must not look like an end token (##)")
        if len(self.required_values) != 1:
            raise ValueError(f"{self.id}: an injection task asks for one looked-up value")
        return self

    # 과제 내용의 해시. 키 순서를 고정한 JSON으로 만들어 같은 내용이면 같은 값이 된다. 실행마다
    # manifest.json에 남겨, 측정 뒤 과제 파일이 바뀌었는지와 두 실행을 짝지어도 되는지를 확인한다(run.py,
    # episode.py, sim_report.py, voice_real.py).
    # 14단계에 생긴 두 필드(injection, user.script)는 비어 있으면 해시에 넣지 않는다. 그래서 그 전의 모든
    # 과제(smoke·dev·test)는 해시가 기록의 manifest와 그대로 같고(analyze same-setup의 짝짓기가 끊기지 않음),
    # 주입 과제는 대본까지 해시에 들어간다.
    def sha256(self) -> str:
        exclude: dict[str, Any] = {}
        if self.injection is None:
            exclude["injection"] = True
        if not self.user.script:
            exclude["user"] = {"script"}
        return hashlib.sha256(
            json.dumps(
                self.model_dump(mode="json", exclude=exclude or None), sort_keys=True, ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()


# 과제 id가 겹치면 기록·판정이 섞이므로 거절한다.
def load_tasks(name_or_path: str | Path) -> list[Task]:
    """`tasks/<name>.yaml`(또는 주어진 경로)을 읽는다. "dev" 같은 이름이면 tasks/dev.yaml, 확장자가 있으면 그
    경로다. utf-8-sig는 Windows 편집기가 남긴 BOM을 받아 준다."""
    path = Path(name_or_path)
    if not path.suffix:
        path = TASKS / f"{name_or_path}.yaml"
    tasks = [Task.model_validate(item) for item in yaml.safe_load(path.read_text(encoding="utf-8-sig"))]
    ids = [task.id for task in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate task ids: {duplicates}")
    return tasks
