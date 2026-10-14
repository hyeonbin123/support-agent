"""실행 한 번의 설정, 대화에 쓰는 고정 문자열, seed 계산."""

# [파일 역할] 실행 한 번의 설정(RunConfig), 대화에 쓰는 고정 문자열(종료 토큰, 첫 인사, 이관 안내), 에피소드
# 종료 사유의 목록, LLM 호출마다의 seed 계산을 모은 계약 모듈이다.
# [왜 필요한가] 실행기(run.py), 에피소드(episode.py), 에이전트 루프(agent.py), 판정(judge.py),
# 분석(analyze.py), 서비스(service/core.py)가 같은 설정 값과 같은 문자열을 봐야 기록끼리 비교가 된다. 그래서
# 다른 모듈이 모두 기대는 계약 파일로 두고, 바꾸면 전체 테스트를 돌린다.
# [설계 이유] RunConfig는 frozen dataclass라 실행 중에 바뀌지 않고, to_dict()로 그대로 manifest.json에 남는다.
# analyze same-setup이 두 실행의 설정을 키별로 비교해 짝지어도 되는지 확인할 때 이 값을 쓴다.
# [실험 축] 각 필드의 P/R/G/F/V/L/C 코드는 docs/experiments.md의 단계별 후보 이름과 같다(기준 구성
# P0·R0·G0·F0·V0·L0·C0에서 한 축씩 바꿔 잰다).

from __future__ import annotations

# hashlib: seed를 프로세스·플랫폼과 무관하게 만들려고 sha256을 쓴다(파이썬 내장 hash()는 실행마다
# 무작위화된다).
# dataclasses·typing: 설정을 불변 객체로 두고, 축 코드를 Literal로 제한해 오타를 타입 수준에서 드러낸다.
import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal

# 사용자 시뮬레이터가 대화를 끝낼 때 내는 토큰(prompts/user_sim.md 8번 지침). episode.py는 '### stop ###'처럼
# 조금 달라도 정규식으로 인정하고, ScriptedUser(테스트용 대본 고객)는 대사가 떨어지면 이 토큰을 낸다.
STOP_TOKEN = "###STOP###"  # 시뮬레이터 고객이 볼일을 마쳤다
# 시나리오만으로는 대화를 이어 갈 수 없을 때 시뮬레이터가 내는 토큰(user_sim.md 9번 지침: 같은 막힘이 세 번
# 넘게 되풀이될 때만). episode.py가 정규식으로 찾아 out_of_scope 종료로 바꾼다.
OUT_OF_SCOPE_TOKEN = "###OUT-OF-SCOPE###"  # 시나리오가 고객에게 어떻게 이어 갈지 알려 주지 않는다
# 모든 대화를 여는 상담원의 고정 인사말. 모델이 만든 말이 아니므로 판정(judge.py)에서 전달 값을 찾을 때는 이
# 인사말을 뺀다.
FIRST_AGENT_MESSAGE = "안녕하세요, 고객센터입니다. 무엇을 도와드릴까요?"
# 상담원 이관(transfer_to_human)이 성공하면 고객에게 전달한 것으로 치는 고정 안내문. tools.py가 도구 결과에
# 넣고, agent.py가 이 문장으로 턴을 끝낸다.
HANDOFF_MESSAGE = "상담원에게 연결해 드리겠습니다. 잠시만 기다려 주세요."
# 생각을 적기만 하는 도구 이름. R1 구성에서만 모델에게 보여 주고(agent.py), MCP 서버는 외부 클라이언트에 이
# 도구를 노출하지 않는다(mcp_server.py).
THINK_TOOL = "think"

# 에피소드 종료 사유 목록. agent.py(TurnResult.stop), episode.py(고객 쪽 종료),
# records.py(EpisodeResult.termination)가 이 타입을 쓰고, 기록(episodes.jsonl)에 문자열 그대로 남는다.
# 판정하는 종료는 아래 JUDGED에 든 것뿐이고, 나머지(한도·형식 오류·완료 주장·컨텍스트)는 실패로 센다.
# infra_error는 누구의 실패도 아니라 분모에서 뺀다.
Termination = Literal[
    "user_stop",
    "out_of_scope",
    "handoff",
    "max_agent_calls",
    "max_user_turns",
    "too_many_tool_errors",
    "agent_format_error",
    "unbacked_claim",  # C1/C2: 도구 결과가 보여 주지 않는 일을 끝냈다고 에이전트가 계속 말했다
    "context_limit",
    "infra_error",
]
# JUDGED: 판정하는 종료. 고객 턴 한도(max_user_turns)는 2026-09-19에 판정 대상으로 옮겼다. 기준선에서 일이
# 끝났는데도 시뮬레이터가 종료 토큰을 내지 못해 20턴을 채운 에피소드가 3개 있었기 때문이다(docs/experiments.md
# 공통 규칙 판정). judge.py·episode.py·analyze.py가 성공 계산과 분모를 가를 때 이 집합을 본다.
# max_user_turns도 판정하는 이유: 대화를 끝내는 것은 시뮬레이터의 일이고, 에이전트는 시간을 끌어도 얻는 것이
# 없다(DB와 에이전트가 한 말을 있는 그대로 판정한다).
JUDGED: frozenset[str] = frozenset({"user_stop", "out_of_scope", "handoff", "max_user_turns"})


# 실행 한 번의 모든 설정. 기본값이 곧 평가의 기준 구성이다(qwen2.5 7B, P0·R0, 규정은 프롬프트에만).
# run.py가 CLI 옵션으로 만들고, service/core.py는 서비스용 값(서비스 모델, 정책, L1 언어 가드 등)으로 만든다.
# 같은 객체가 에이전트 루프와 판정에 그대로 넘어간다.
@dataclass(frozen=True)
class RunConfig:
    # model은 상담원 역할 모델, user_model은 고객 역할(사용자 시뮬레이터) 모델이다. 12단계 전까지는 둘 다
    # qwen2.5 7B였고, 시뮬레이터 모델은 한 단계 안에서 바꾸지 않는다(바꾸면 그 단계 후보를 모두 다시 잼).
    model: str = "qwen2.5:7b-instruct"
    user_model: str = "qwen2.5:7b-instruct"
    # 아래 필드가 실험 축이다. P(규정을 도구에서도 막는가), R(think 도구·쓰기 전 고객 확인), G(멈춤 가드),
    # F(본문에 샌 도구 호출 구제), V(음성 채널), L(중국어·일본어 문자 답 되돌리기), C(근거 없는 완료 주장
    # 되돌리기). 기본값 0번이 기준 구성이다.
    policy: Literal["P0", "P1"] = "P0"  # P0: 규정은 프롬프트에만, P1: 도구도 거절한다
    # R1: think 도구. R2(10단계): 고객이 미리보기에 동의했을 때만 쓰기가 실행된다(confirm.py). 프롬프트와
    # 도구 목록은 R0와 같다.
    reasoning: Literal["R0", "R1", "R2"] = "R0"
    # G1: 하겠다는 약속만 하는 답은 전달하지 않고 다시 시킨다. G2: 거기에 더해 재시도는 표본 추출로 한다.
    guard: Literal["G0", "G1", "G2"] = "G0"
    rescue: Literal["F0", "F1"] = "F0"  # F1: 본문에 샌 도구 호출을 해석해 실행한다
    # V1: 고객의 말이 음성 합성과 인식을 거쳐 에이전트에게 간다.
    # V2: 거기에 더해 인식된 글이 먼저 규칙 기반 정규화기를 거친다.
    # V4(9단계): V2이고, 전화가 고객의 등록 번호에서 온다. verify_caller가 그 고객 한 명의 이름과만 맞춰
    # 본다(find_customer는 주지 않는다). V3는 쓰지 않는다.
    voice: Literal["V0", "V1", "V2", "V4"] = "V0"
    # L1: 한자나 가나로 쓴 답은 전달하지 않고 형식 오류처럼 다시 시킨다. 서비스는 이것을 켠다. 측정한 적이
    # 없어서 평가의 기본값은 L0다.
    language: Literal["L0", "L1"] = "L0"
    # C1: 이 대화의 어떤 도구 결과도 보여 주지 않는데 일이 끝났다고 말하는 답("취소가 완료되었습니다")은
    # 전달하지 않고 다시 시킨다. C2: 거기에 더해 재시도는 표본 추출로 한다.
    claims: Literal["C0", "C1", "C2"] = "C0"
    # 에이전트는 0.0(결정적)으로 두고, 시뮬레이터는 0.3으로 둬 시도마다 고객의 말이 조금씩 달라지게 한다.
    # 시뮬레이터 값은 기준선 첫 실행 뒤 0.7에서 0.3으로 낮췄다(docs/experiments.md 공통 규칙 무작위성).
    # base_seed는 derive_seed의 첫 입력이라, 같은 명령은 같은 seed로 돈다.
    temperature: float = 0.0
    user_temperature: float = 0.3
    base_seed: int = 1000
    # 에이전트의 Ollama 컨텍스트 크기(토큰). 0단계 스모크에서 시스템 프롬프트가 약 3,400토큰, 대화 끝이 최대
    # 5,160토큰이라 16384에 여유가 있음을 확인했다(docs/experiments.md 0단계 결과). 프롬프트가 이 값의 95%를
    # 넘으면 agent.py가 context_limit로 끝낸다.
    num_ctx: int = 16384
    # 시뮬레이터의 num_ctx. None(8단계 이전 기록)은 num_ctx와 같다는 뜻이고, 실행기는 언제나 정해진 숫자를
    # 적는다. 한 모델이 두 역할을 맡으면 값도 하나여야 한다(아니면 Ollama가 모델을 다시 읽는다).
    user_num_ctx: int | None = None
    # 한 응답의 최대 생성 토큰 수. 이 한도에 닿아 잘린 응답은 형식 오류로 처리된다(docs/design.md 에이전트
    # 루프 규칙).
    max_tokens: int = 1024
    # 에피소드 한도(docs/design.md 에이전트 루프 규칙 표): 에이전트 LLM 호출 30, 고객 턴 20, 누적 도구 오류
    # 10. 넘으면 각각 max_agent_calls·max_user_turns·too_many_tool_errors로 끝난다. 서비스에서는
    # max_agent_calls를 턴당 한도로 바꿔 넣는다.
    max_agent_calls: int = 30  # 에피소드당 에이전트의 LLM 호출 수
    max_user_turns: int = 20
    max_tool_errors: int = 10
    # 재시도 관련 값. 멈춤 가드(G1·G2, 2단계)와 완료 주장 가드(C1·C2, 7단계)는 한 턴에 2번까지 다시 시키고,
    # G2·C2는 다시 시킬 때 temperature 0.7로 표본을 뽑는다. 두 단계 모두 판정은 '개선 없음'이었다(README 측정
    # 결과).
    stall_retry_temperature: float = 0.7  # G2 전용
    max_stall_retries: int = 2  # 에이전트 턴당(G1). 넘으면 답을 그대로 전달한다
    claim_retry_temperature: float = 0.7  # C2 전용
    max_claim_retries: int = 2  # 에이전트 턴당(C1). 넘으면 아무것도 전달하지 않고 에피소드가 끝난다
    # 형식 오류(빈 응답, 본문에 샌 도구 호출, 길이 제한으로 잘림)는 고객에게 보내지 않고 한 턴에 2번까지 다시
    # 시킨다(docs/experiments.md 공통 규칙 에이전트 루프).
    max_format_retries: int = 2  # 에이전트 턴당. 0이면 "형식이 깨진 답 하나로 실패"인 엄격한 규칙이 된다

    # P1이면 도구 안의 규정 검사(ctx.check_policy)가 위반을 실제로 막는다. P0이면 위반을 기록만 하고
    # 통과시킨다(docs/design.md 실행 경로 하나).
    @property
    def enforce_policy(self) -> bool:
        return self.policy == "P1"

    # 음성 V4(9단계)인지. True면 tools.py가 find_customer 대신 verify_caller를 등록하고, agent.py가 시스템
    # 프롬프트에 발신 번호 본인 확인 규칙 한 줄을 더한다.
    @property
    def caller_id(self) -> bool:
        """V4: 고객을 전화가 걸려 온 번호와 고객이 말한 이름으로 확인한다."""
        return self.voice == "V4"

    # manifest.json에 남기고, analyze same-setup에서 두 실행의 설정을 비교하는 데 쓰는 dict 형태.
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# LLM 호출마다 고정 seed를 만든다. 입력은 (기준 seed, 과제, 시도, 역할, 호출 순번)이고 역할은 agent·user·voice
# 등이다.
# 왜 이 방식인가: 난수 생성기 하나를 실행 내내 돌려 쓰면 앞 에피소드의 호출 수가 바뀔 때 뒤 에피소드의 seed가
# 모두 밀린다. 입력 값에서 바로 해시로 만들면 같은 과제·시도·순번은 언제나 같은 seed를 받아 두 실행을 과제
# 단위로 짝지을 수 있다.
# sha256의 앞 4바이트를 정수로 바꾸고 0x7FFFFFFF로 31비트에 맞춰 부호 있는 32비트 정수 범위 안의 양수로
# 만든다.
# 호출 위치: agent.py(에이전트), user_sim.py(시뮬레이터), episode.py·voice_real.py(음성 합성).
def derive_seed(base_seed: int, task_id: str, trial: int, role: str, call_index: int) -> int:
    """LLM 호출마다 고정된 31비트 seed. 프로세스, 플랫폼, 해시 무작위화와 무관하다."""
    text = f"{base_seed}|{task_id}|{trial}|{role}|{call_index}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF
