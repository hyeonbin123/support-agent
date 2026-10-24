"""서비스 설정. 환경 변수(접두사 SUPPORT_AGENT_)에서 읽는다."""

# [파일 역할] 채팅 서비스(3단계)의 설정 전부를 한 클래스(Settings)로 모은다. 값은 SUPPORT_AGENT_ 접두사가 붙은
# 환경 변수에서 읽는다(docs/service.md의 설정 목록).
# [왜 필요한가] 같은 코드를 Docker Compose(PostgreSQL), 호스트의 SQLite 실행, 모델 없는 오프라인 앱,
# 테스트에서 돌리므로 DB 주소·모델·한도 같은 값을 코드 밖에서 바꿀 수 있어야 한다. pydantic 모델로 두어 잘못된
# 값(예: THINK=false)은 앱이 뜰 때 바로 거절된다.
# [연결] service/app.py의 create_app, service/offline.py, mcp_server.py가 Settings.from_env()로 만든다.
# service/core.py는 clock()으로 쇼핑몰의 현재 시각을 얻는다.

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any, Literal

# pydantic: 환경 변수 문자열을 타입(int, bool, Literal, 시간대 있는 datetime)으로 바꾸고 범위(ge/le)를
# 검사한다. AwareDatetime은 시간대 없는 시각을 거절해, 고정 시계가 KST/UTC를 헷갈리지 않게 한다.
from pydantic import AwareDatetime, BaseModel, Field, model_validator

# 환경 변수 접두사. 필드 이름을 대문자로 바꿔 붙인다(model → SUPPORT_AGENT_MODEL).
PREFIX = "SUPPORT_AGENT_"
# 생성한 쇼핑몰 데이터는 이 시각을 기준으로 쓰였다(과제도 이 시각을 쓴다). 벽시계를 쓰면 모든 주문이 곧
# 반품 기한을 넘기므로, 시연은 SUPPORT_AGENT_NOW=real이 아니면 고정 시계로 돈다.
DEMO_NOW = "2026-09-14T10:00:00+09:00"
# 2026-10-04부터 기본 모델(docs/experiments.md 8단계): 사고 모드를 끄고 잰 시험용 pass^1이 qwen2.5 7B의
# 18.1%에 비해 55.6%였다. 다만 과제가 요구하지 않은 쓰기도 6건에 비해 13건이었으므로 승인 관문은 그대로 둔다.
# MODEL=qwen2.5:7b-instruct로 이전 모델로 돌아갈 수 있다.
DEFAULT_MODEL = "qwen3.5:4b"
# THINK 설정 문자열 → Ollama 요청의 think 값. None은 요청에 think 필드를 아예 넣지 않는다는 뜻이다(사고 모드가
# 없는 모델은 이 필드를 거부할 수 있어서, docs/service.md).
_THINK = {"on": True, "off": False, "omit": None}


# 서비스 설정 한 묶음. 필드 기본값이 곧 시연용 기본 구성이다.
# - database_url: 기본은 SQLite 파일(outputs/service.db). Compose는 PostgreSQL 주소를 넣는다
# - num_ctx: 서비스는 16384. 8단계 측정은 12288로 쟀으므로 잰 구성과 다른 점이다(docs/service.md)
# - approval_refund_won: 이 금액 이상 환불되는 취소·반품은 관리 화면에서 사람이 승인해야 실행된다
# - admin_token: 빈 문자열이면 관리자 API가 꺼진다
# - max_*: 메시지 길이, 턴당 LLM 호출, 세션당 턴, 음성 파일 크기·길이, 동시에 도는 턴 수, 동시에 도는 답
#   읽어 주기(합성) 수의 한도(docs/service.md "한도")
# 정수 설정에는 모두 범위(Field의 ge/le)를 둔다. 범위 밖 값(0, 음수, 너무 작은 num_ctx, 너무 큰 동시 턴 수)은
# 서비스가 시작할 때 ValidationError로 멈춘다: 전에는 글자 한도 둘만 검사해, 예컨대
# MAX_TURNS_PER_SESSION=0으로 띄우면 /healthz는 정상인데 모든 상담이 첫 메시지 뒤 닫히는 상태가
# 됐다(2026-10-24 고침, tests/test_review_t7.py).
class Settings(BaseModel):
    database_url: str = "sqlite:///outputs/service.db"
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = DEFAULT_MODEL
    # 모든 에이전트 요청에 넣는 Ollama의 `think` 필드. 비워 두면 기본 모델은 "off"(그렇게 쟀다), 다른 모델은
    # "omit"(보내지 않음)이 된다: 사고 모드가 없는 모델은 이 필드를 거절한다.
    think: Literal["on", "off", "omit"] = "off"
    # 에이전트 호출의 컨텍스트 길이(Ollama의 num_ctx). 서비스는 16384(8단계 측정은 12288, docs/service.md).
    # 하한 4096: 시스템 프롬프트(규정·지침)와 도구 설명만으로 첫 에이전트 호출의 프롬프트가 이미 약
    # 3,500토큰이라(실제 모델로 확인한 상담의 첫 호출이 3,526토큰, docs/service.md) 그보다 작으면 모든 상담이
    # 첫 턴에 컨텍스트 한도로 닫힌다. 0·음수도 전에는 그대로 받아들여졌다(2026-10-24 고침).
    num_ctx: int = Field(default=16384, ge=4096)
    # 서비스는 규정이 금지하는 것을 거절한다(P1). 그것이 성공률을 올리는지는 docs/experiments.md의 질문이고,
    # 여기서는 잘못된 환불이 돈을 잃게 하므로 코드가 검사할 수 있는 것은 도구가 모두 검사한다.
    policy: Literal["P0", "P1"] = "P1"
    # C1: 이 대화의 어떤 도구 결과도 보여 주지 않는데 일이 끝났다고 말하는 답은 절대 전달하지 않는다. 기본은
    # 꺼짐: 개발용 과제에서 C1은 −2.1%p와 정답에 없는 쓰기 1건 증가, C2는 −5.2%p라 둘 다 측정 전에 적은 규칙을
    # 만족하지 못했다(docs/experiments.md 7단계). CLAIMS=C1로 켠다.
    claims: Literal["C0", "C1", "C2"] = "C0"
    # 13단계 후보(docs/experiments.md 13단계). 모두 기본은 꺼짐이고, 켜는 조건은 13단계 규칙에 측정 전에
    # 적었다. GUARD=G1: "확인해 보겠습니다"만 하는 답을 붙잡아 다시 시킨다(2단계 멈춤 가드). ERRORS=E1: 규정이
    # 막은 쓰기의 오류 글에 그 검사가 쓴 사실(수령·출고 시각, 접수 마감일, 이미 발급된 쿠폰, 도착 예정일)을
    # 붙인다. REPEAT=D1: 바로 앞에 고객에게 전달한 답과 같은 답을 붙잡아 한 번 다시 시킨다.
    guard: Literal["G0", "G1"] = "G0"
    errors: Literal["E0", "E1"] = "E0"
    repeat: Literal["D0", "D1"] = "D0"
    now: AwareDatetime | Literal["real"] = Field(default_factory=lambda: datetime.fromisoformat(DEMO_NOW))
    # 이 금액 이상 환불되는 취소·반품은 사람이 승인해야 실행된다(core.py의 승인 관문, docs/service.md "사람
    # 승인"). 0이면 환불액과 상관없이 취소·반품이 모두 승인 대기에 오른다('전부 승인' 구성이라 받는다). 음수는
    # 뜻이 없어 거절한다.
    approval_refund_won: int = Field(default=100_000, ge=0)
    admin_token: str = ""  # ""이면 관리자 API가 꺼진다
    max_message_chars: int = Field(default=1000, ge=1, le=4000)  # 요청 모델(MessageIn)의 상한이 4000이다
    # 턴당 LLM 호출 수와 세션당 턴 수는 1 이상이어야 한다. 0이면 모든 턴이 호출 한도의 고정 안내로 끝나거나
    # 모든 세션이 첫 메시지 뒤 닫히는데, 서비스는 /healthz가 정상인 채 떠서 설정이 원인이라는 신호가
    # 없었다(2026-10-24 고침). 세션당 턴 수에 상한은 두지 않는다: 보안 스캔은 세션이 닫히지 않게 100000으로
    # 띄운다(docs/service.md "모델 없이 띄우기, 보안 스캔").
    max_agent_calls_per_turn: int = Field(default=12, ge=1)
    max_turns_per_session: int = Field(default=40, ge=1)
    load_seed: bool = True  # 빈 쇼핑몰 DB를 생성 데이터로 채운다
    expose_openapi: bool = False  # /openapi.json을 내놓는다(API 스캐너와 클라이언트용, 기본은 꺼짐)
    voice: bool = False  # 음성 입출력(`voice` 의존성 그룹이 필요하다)
    tts_device: Literal["cuda", "cpu"] = "cuda"
    stt_device: Literal["cuda", "cpu"] = "cuda"
    # 녹음 파일 크기와 풀었을 때 길이의 한도. 기본 5 MB는 브라우저로 녹음한 말 약 5분 분량이고, 길이 한도가
    # 따로 있는 것은 낮은 비트레이트면 작은 파일에도 몇 시간 분량의 말이 들어갈 수 있어서다. 둘 다 0이면 모든
    # 녹음이 거절되므로 1 이상만 받는다.
    max_audio_bytes: int = Field(default=5_000_000, ge=1)
    max_tts_chars: int = Field(default=600, ge=1, le=4000)
    max_audio_seconds: int = Field(default=60, ge=1)
    # 모든 세션을 통틀어 동시에 도는 턴(과 인식) 수. 0이면 모든 메시지가 503이고 음수는 create_app의
    # BoundedSemaphore가 ValueError를 내므로 1 이상만 받는다. 상한 16을 두는 이유: 턴의 SSE 응답 생성기는 동기
    # 함수라 Starlette가 이벤트를 하나 꺼낼 때마다 anyio 스레드 풀(기본 40자리)에서 돌리는데, 다음 이벤트를
    # 기다리는 동안(모델이 답할 때까지) 그 자리를 쥔 채 멈춰 있다(app.py turn_stream의 stream). 같은 풀을
    # /healthz·글 메시지 수신·대화 조회·관리자 API의 동기 처리기가 쓰므로, 이 값을 40 가까이 올리면 턴이
    # 자리를 모두 차지해 /healthz가 답하지 못하고 Docker의 HEALTHCHECK가 실패해 재시작될 수 있다. 16이면 턴이
    # 모두 차도 자리 24개가 남는다(docs/service.md "한도").
    max_concurrent_turns: int = Field(default=4, ge=1, le=16)
    # 모든 세션을 통틀어 동시에 도는 답 읽어 주기(/speech 합성) 수. 턴 자리와는 따로 센다: 합성은 턴을 쓰지
    # 않으므로 위 한도가 /speech를 막지 못했고, 음성 모델은 한 번에 하나만 돌리므로(voice_frontend의 잠금)
    # 겹친 합성 요청은 모두 작업 스레드를 쥔 채 그 잠금을 기다렸다. 넘으면 기다리지 않고 429와 Retry-After로
    # 돌려보낸다(app.py speak). 기본 2 = 하나는 합성 중, 하나는 그 뒤에 줄 서기(화면은 조각을 한 번에 하나씩만
    # 청한다). 0이면 모든 요청을 거절하게 되므로 받지 않는다(ge=1).
    max_concurrent_speech: int = Field(default=2, ge=1)

    # THINK를 비워 두면 모델에 맞춰 채운다: 기본 모델(qwen3.5:4b)은 잰 그대로 "off", 다른 모델은 "omit"(보내지
    # 않음).
    # mode="before"라서 필드 검증 전에 원래 입력 dict를 고친다. Compose는 THINK가 없으면 빈 문자열을
    # 넘기므로(compose.yaml) None과 ""를 같이 본다.
    # 이전 모델(qwen2.5:7b-instruct)로 돌아갈 때 MODEL만 바꿔도 think가 거부되지 않게 하려는
    # 것이다(tests/test_service.py가 두 경우를 확인한다).
    @model_validator(mode="before")
    @classmethod
    def _think_follows_the_model(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("think") in (None, ""):
            model = data.get("model") or DEFAULT_MODEL
            data = {**data, "think": "off" if model == DEFAULT_MODEL else "omit"}
        return data

    # OllamaProvider에 넘길 think 값(True/False/None). app.py의 create_app이 제공자를 만들 때 쓴다.
    def agent_think(self) -> bool | None:
        """OllamaProvider가 `think`로 보낼 값. None이면 아무것도 보내지 않는다."""
        return _THINK[self.think]

    # 도구와 규정이 보는 "쇼핑몰의 지금". 기본은 고정 시각(DEMO_NOW)이고, NOW=real일 때만 실제 UTC 시각이다.
    # 세션·감사 로그·승인 기록의 시각은 이것이 아니라 항상 실제 시각을 쓴다(core.py의 wall_clock,
    # docs/service.md "고정 시계").
    def clock(self) -> datetime:
        return datetime.now(UTC) if self.now == "real" else self.now

    # 환경 변수(또는 테스트가 넘긴 dict)에서 Settings를 만든다.
    # 값이 빈 문자열인 변수는 없는 것으로 친다. Compose의 `${THINK:-}`처럼 비어 있는 변수가 기본값을 덮어쓰지
    # 않게 하려는 것이다.
    # 문자열 → 타입 변환과 범위 검사는 model_validate가 맡으므로 잘못된 값은 여기서 ValidationError로 멈춘다.
    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> Settings:
        env = dict(os.environ) if env is None else env
        values = {
            name: env[PREFIX + name.upper()]
            for name in Settings.model_fields
            if env.get(PREFIX + name.upper(), "") != ""
        }
        return Settings.model_validate(values)
