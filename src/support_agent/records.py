"""에피소드가 남기는 것: 호출 로그, 판정, JSONL 기록."""

# [파일 역할] 에피소드 하나가 남기는 기록의 형태를 정의한다. 도구 호출 로그(ToolCallLog), LLM 호출
# 로그(LLMCallLog), 판정(Verdict), 그리고 이것들을 묶어 episodes.jsonl의 한 줄이 되는 EpisodeResult다.
# [왜 필요한가] 이 프로젝트는 수치를 나중에 기록에서 다시 계산할 수 있어야 한다(README 다시 재는 방법: analyze
# table·compare 등이 모델 없이 reports/의 기록만으로 돈다). 그래서 대화 전체, 도구 호출의 원래 인자와 정규화한
# 인자, 판정 요소, DB 변경분과 해시를 모두 한 줄에 남긴다.
# [연결] agent.py가 ToolCallLog·LLMCallLog를, user_sim.py가 시뮬레이터의 LLMCallLog를, judge.py가 Verdict를,
# episode.py가 EpisodeResult를 만들고, run.py가 to_json_line()으로 파일에 쓴다. analyze.py는 이 JSON을 다시
# 읽는다. 다른 모듈이 모두 기대는 계약 파일이다.

from __future__ import annotations

# dataclasses.asdict로 중첩 dataclass(Verdict, 로그 목록)를 통째로 dict로 바꾼 뒤 json으로 한 줄에 쓴다.
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# 종료 사유 타입은 config.py에 하나만 정의하고 여기서 가져와, 기록과 루프가 같은 값 목록을 쓰게 한다.
from support_agent.config import Termination


# 도구 호출 한 번의 기록. 판정(judge.py)이 정답 동작과 비교하고 보조 지표를 세는 재료다.
# - raw_arguments: 모델이 보낸 그대로. args: 검증·정규화한 형태(검증 실패면 None). 정답과는 args로 비교하고,
#   None이면 raw_arguments로 대신한다.
# - ok·write: 성공한 쓰기만 세어 정답에 없는 쓰기(unexpected_writes)와 빠진 쓰기를 계산한다.
# - policy_blocked·error_code: P1에서 규정 검사가 막은 호출과 그 코드. violations: P0에서 위반인데 통과시킨
#   코드 목록.
# - content: 모델에 돌려준 도구 결과(JSON 또는 'Error: [code] 메시지'). ms: 실행 시간.
@dataclass
class ToolCallLog:
    agent_call: int  # 이 도구 호출을 낸 에이전트 LLM 호출의 순번
    name: str
    raw_arguments: dict[str, Any]  # 모델이 보낸 그대로
    args: dict[str, Any] | None  # 정규화한 형태. 검증에 실패하면 None
    ok: bool
    error_code: str | None
    policy_blocked: bool
    violations: list[str]
    content: str
    write: bool
    ms: float


# LLM 호출 한 번의 기록. who는 에이전트(agent)인지 시뮬레이터(user)인지, index는 역할별 호출 순번이고 seed는
# 그 호출에 쓴 derive_seed 값이다.
# format_error는 그 답을 받은 그대로 쓰지 않은 이유(빈 응답, 본문에 샌 도구 호출, 잘림, 언어, 멈춤, 근거 없는
# 완료 주장 등)라 형식 오류 수와 가드 발동 수를 여기서 센다. 토큰 수와 시간 값은 호출당 시간, 컨텍스트 사용량
# 같은 보조 지표에 쓴다.
@dataclass
class LLMCallLog:
    who: Literal["agent", "user"]
    index: int  # 에피소드 안에서 역할별로 센 호출 순번
    seed: int | None
    text: str
    tool_calls: list[dict[str, Any]]
    # 답을 받은 그대로 쓰지 않은 이유: empty | leaked_tool_call | cut_off | rescued_tool_call |
    # wrong_language | stall(G1) | unbacked_claim(C1) | repeat(D1, 13단계). 그대로 썼으면 None.
    format_error: str | None
    dropped_text: str  # 도구 호출과 함께 왔지만 전달하지 않은 본문
    dropped_calls: int  # 첫 번째 뒤의 도구 호출 수(실행하지 않음)
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    wall_ms: float
    load_ms: float
    prompt_eval_ms: float
    eval_ms: float


# 에피소드 판정. 성공 = judged(판정하는 종료) 그리고 db_match(최종 DB == 정답만 실행한 DB) 그리고 values(꼭
# 전달해야 할 값이 모두 전달됨)다. LLM 판정은 쓰지 않는다(docs/experiments.md 공통 규칙 판정).
# 나머지는 보조 지표다. unexpected_writes는 성공 조건에 넣지 않고 따로 보고한다(τ-bench와 같이, 바꿨다가
# 되돌린 쓰기도 DB가 같으면 성공). db_diff는 정답 DB와 최종 DB의 차이로, 실패 분석용이다.
@dataclass
class Verdict:
    success: bool
    judged: bool  # 판정하는 방식으로 끝나지 않았으면 False(그때 success도 False)
    db_match: bool
    values: dict[str, bool]  # 꼭 전달할 값의 이름 -> 고객에게 한 말에서 찾았는지
    unexpected_writes: int  # 정답 동작에 없는 성공한 쓰기 호출 수 (규정 위반 수)
    missing_writes: int
    policy_violations: list[str] = field(default_factory=list)  # P0에서 통과시킨 위반 코드
    policy_blocks: list[str] = field(default_factory=list)  # P1에서 거절한 위반 코드
    blocked_forbidden_attempts: int = 0  # 에이전트가 금지 호출을 시도했고 도구가 거절한 횟수
    auth_blocks: int = 0  # 고객 본인 확인이 안 되어 거절된 읽기·쓰기 수
    db_diff: dict[str, Any] = field(default_factory=dict)


# episodes.jsonl의 한 줄 = 에피소드 하나의 전부.
# - status: completed(판정하는 종료), truncated(한도·형식 오류 등으로 끝나 실패로 셈), infra_error(모델
#   서버·시뮬레이터·도구 버그. 분모에서 뺌).
# - messages는 에이전트 쪽 대화, user_messages는 시뮬레이터 쪽 대화다. 음성 조건에서는 두 쪽이 들은 말이
#   달라서 둘 다 남긴다.
# - db_changes는 seed DB와 최종 DB의 차이, seed_hash·final_hash·gold_hash는 각 DB 덤프의 해시라 같은 seed에서
#   시작했는지, 정답과 같은 상태로 끝났는지를 기록만으로 확인할 수 있다.
# - task_sha256은 과제 내용의 해시라, 측정 뒤 과제 파일이 바뀌었는지 드러난다.
@dataclass
class EpisodeResult:
    run_id: str
    task_id: str
    task_sha256: str
    trial: int
    status: Literal["completed", "truncated", "infra_error"]
    termination: Termination
    verdict: Verdict | None  # infra_error면 None
    error: str  # infra_error면 예외 글, 아니면 ""
    messages: list[dict[str, Any]]  # 시스템 프롬프트를 뺀 에이전트 쪽 대화
    user_messages: list[dict[str, Any]]  # 시스템 프롬프트를 뺀 시뮬레이터 쪽 대화
    tool_calls: list[ToolCallLog]
    llm_calls: list[LLMCallLog]
    db_changes: dict[str, Any]  # diff_dumps(seed 덤프, 최종 덤프)
    seed_hash: str
    final_hash: str
    gold_hash: str
    wall_seconds: float
    # 음성 조건(V1·V2·V4)에서만: 고객 발화마다 한 항목(said, spoken, heard, text, 시간 값)
    voice: list[dict[str, Any]] = field(default_factory=list)

    # JSONL 한 줄로 직렬화한다. ensure_ascii=False로 한국어 대화를 \u 이스케이프 없이 그대로 남겨 사람이
    # 기록을 바로 읽을 수 있게 한다.
    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
