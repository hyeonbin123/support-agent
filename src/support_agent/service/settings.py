"""Settings of the service, read from environment variables (prefix SUPPORT_AGENT_)."""

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
# The generated shop data is written around this moment (the tasks use it too). With the wall clock every
# order would soon be past its return window, so the demo runs on a fixed clock unless SUPPORT_AGENT_NOW=real.
DEMO_NOW = "2026-09-14T10:00:00+09:00"
# Since 2026-10-04 (docs/experiments.md, stage 8): test pass^1 55.6% against 18.1% for qwen2.5 7B, measured
# with thinking off. It also made 13 writes the tasks did not ask for against 6, so the approval gate stays.
# MODEL=qwen2.5:7b-instruct brings the earlier model back.
DEFAULT_MODEL = "qwen3.5:4b"
# THINK 설정 문자열 → Ollama 요청의 think 값. None은 요청에 think 필드를 아예 넣지 않는다는 뜻이다(사고 모드가
# 없는 모델은 이 필드를 거부할 수 있어서, docs/service.md).
_THINK = {"on": True, "off": False, "omit": None}


# 서비스 설정 한 묶음. 필드 기본값이 곧 시연용 기본 구성이다.
# - database_url: 기본은 SQLite 파일(outputs/service.db). Compose는 PostgreSQL 주소를 넣는다
# - num_ctx: 서비스는 16384. 8단계 측정은 12288로 쟀으므로 잰 구성과 다른 점이다(docs/service.md)
# - approval_refund_won: 이 금액 이상 환불되는 취소·반품은 관리 화면에서 사람이 승인해야 실행된다
# - admin_token: 빈 문자열이면 관리자 API가 꺼진다
# - max_*: 메시지 길이, 턴당 LLM 호출, 세션당 턴, 음성 파일 크기·길이, 동시에 도는 턴 수의
#   한도(docs/service.md "한도")
class Settings(BaseModel):
    database_url: str = "sqlite:///outputs/service.db"
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = DEFAULT_MODEL
    # Ollama's `think` field on every agent request. Left unset it is "off" for the default model (measured
    # so) and "omit" (not sent) for any other: models without a thinking mode refuse the field.
    think: Literal["on", "off", "omit"] = "off"
    num_ctx: int = 16384
    # The service refuses what the rules forbid (P1). Whether that raises the success rate is a question of
    # docs/experiments.md; here a wrong refund costs money, so the tools check whatever code can check.
    policy: Literal["P0", "P1"] = "P1"
    # C1: a reply that says work is done which no tool result of the conversation shows is never delivered.
    # Off by default: on the development tasks C1 cost −2.1%p and one more unexpected write, C2 −5.2%p, so
    # neither met the rule written before measuring (docs/experiments.md, stage 7). CLAIMS=C1 turns it on.
    claims: Literal["C0", "C1", "C2"] = "C0"
    now: AwareDatetime | Literal["real"] = Field(default_factory=lambda: datetime.fromisoformat(DEMO_NOW))
    approval_refund_won: int = 100_000  # refunds from this amount wait for a person
    admin_token: str = ""  # "" disables the admin API
    max_message_chars: int = Field(default=1000, ge=1, le=4000)  # the request model stops at 4000
    max_agent_calls_per_turn: int = 12
    max_turns_per_session: int = 40
    load_seed: bool = True  # fill an empty shop database with the generated data
    expose_openapi: bool = False  # serve /openapi.json (for API scanners and clients; off by default)
    voice: bool = False  # speech in and out (needs the `voice` dependency group)
    tts_device: Literal["cuda", "cpu"] = "cuda"
    stt_device: Literal["cuda", "cpu"] = "cuda"
    max_audio_bytes: int = 5_000_000  # about five minutes of browser-recorded speech
    max_tts_chars: int = Field(default=600, ge=1, le=4000)
    max_audio_seconds: int = 60  # a small file can hold hours of low-bitrate speech
    max_concurrent_turns: int = 4  # turns (and recognitions) running at once, over all sessions

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
        """The value OllamaProvider sends as `think`; None sends nothing."""
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
