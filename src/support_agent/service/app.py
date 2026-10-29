"""웹 계층: 서버 전송 이벤트(SSE)로 하는 채팅, 관리자 API, 정적 화면 두 개.

실행: uv run uvicorn support_agent.service.app:create_app --factory --port 8062
"""

# [파일 역할] 채팅 서비스의 웹 계층(FastAPI). 고객 채팅(SSE), 음성 끝점, 관리자 API, 정적 화면 두
# 개(채팅·관리)를 내놓는다.
# [설계 이유] 상담 한 턴의 처리, 감사 로그, 승인 대기열은 모두 core.ChatService에 있고 이 파일은 그 위의 얇은
# 층이다. 그래서 테스트는 웹 없이 core를 직접 부를 수 있고, 같은 core를 MCP 서버(mcp_server.py)도
# 쓴다(docs/service.md 구성).
# [흐름] 브라우저 → POST /api/sessions/{id}/messages → 작업 스레드에서 ChatService.handle() → emit된 진행
# 이벤트를 SSE로 흘림 → 마지막에 reply·end 이벤트.
# [연결] Dockerfile과 README의 uvicorn 명령이 create_app을 팩토리로 띄우고, service/offline.py가 대역을 넣어
# 같은 함수를 부른다.

from __future__ import annotations

# queue·threading: 한 턴을 작업 스레드에서 돌리고, 그 스레드가 넣는 이벤트를 큐로 받아 SSE 응답으로 흘린다.
# 시작 때의 모델 예열도 작업 스레드에서 한다. secrets: 관리자 토큰을 상수 시간으로 비교한다.
# importlib.resources.files: 패키지 안의 static/ 폴더를 설치 위치와 상관없이 찾는다. logging: 예열 결과를 서버
# 로그에 한 줄 남긴다.
import json
import logging
import queue
import secrets
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any

# FastAPI: 라우팅, 의존성 주입(Depends), 요청 본문 검증(pydantic). StreamingResponse로 SSE를, StaticFiles로
# 화면의 js·css를 내보낸다. RequestValidationError·jsonable_encoder·JSONResponse는 본문 검증 오류(422)의
# 응답을 직접 만들 때 쓴다(create_app의 validation_error).
from fastapi import Depends, FastAPI, Header, HTTPException, Path, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import Engine

# run_in_threadpool: async 처리기(post_voice) 안의 동기 DB 조회를, FastAPI가 동기 처리기를 돌리는 것과 같은
# 작업 스레드 풀로 보낸다. 이벤트 루프 스레드에서 DB를 읽으면 그동안 모든 세션의 SSE 전달과 본문 수신이
# 멈춘다.
from starlette.concurrency import run_in_threadpool

# 프로젝트 내부: 모델 제공자(기본 OllamaProvider), DB 준비(bootstrap), 대화 처리와 그 예외(core), 설정, 음성
# 앞단.
from support_agent.chat import ChatProvider
from support_agent.ollama import OllamaProvider
from support_agent.service.bootstrap import prepare_database
from support_agent.service.core import (
    ApprovalStateError,
    BusyError,
    ChatService,
    SessionClosedError,
    SessionNotFoundError,
)
from support_agent.service.settings import Settings
from support_agent.service.voice_frontend import VoiceFrontEnd, load_voice_front_end
from support_agent.voice.speech import AudioTooLongError

# index.html·admin.html·chat.js·admin.js·style.css가 있는 정적 파일 폴더.
STATIC = files("support_agent.service") / "static"
# 모든 응답에 붙이는 보안 헤더.
# - nosniff: 브라우저가 응답 형식을 추측하지 않게 한다(JSON을 HTML로 읽지 않음)
# - X-Frame-Options·frame-ancestors: 다른 사이트가 이 화면을 iframe에 넣지 못하게 한다(클릭재킹 방지)
# - CSP default-src 'self': 같은 출처의 스크립트·스타일만 허용하고 인라인 스크립트는 막는다. 그래서 화면의
#   스크립트는 모두 별도 js 파일이다. ZAP 스캔의 저장형 XSS 항목을 고칠 것으로 보지 않은 근거 가운데
#   하나다(docs/service.md 스캔 8~10)
# - Permissions-Policy: 카메라·마이크·위치를 기본으로 끈다. 음성 모드에서만 마이크를 같은 출처에
#   연다(create_app)
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


# 고객 메시지 본문. 4000은 설정으로 올릴 수 있는 상한이고, 실제 한도(기본 1,000자)는 post_message가 설정값으로
# 다시 검사한다.
class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


# 소리로 읽어 줄 답 글. 실제 상한(max_tts_chars, 기본 600자)은 speak가 검사한다.
class SpeakIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


# 관리 화면의 승인 결정. by는 담당자 이름으로 입력한 그대로 기록되고(인증은 공유 토큰 하나뿐), note는 거절
# 사유로 고객 안내에 붙는다(core.py).
class DecisionIn(BaseModel):
    approve: bool
    by: str = Field(default="admin", min_length=1, max_length=64)
    note: str = Field(default="", max_length=500)


# SSE 한 덩어리를 만든다: "event: 이름" 줄, "data: JSON" 줄, 빈 줄. ensure_ascii=False로 한국어를
# 이스케이프하지 않고 그대로 보낸다.
def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# 서버 로그. uvicorn의 기본 로그 설정(uvicorn/config.py의 LOGGING_CONFIG)은 uvicorn·uvicorn.error·
# uvicorn.access 로거에만 처리기와 INFO 수준을 준다. 그래서 이 이름으로 쓰면 "INFO:     Application startup
# complete." 같은 uvicorn의 시작 줄과 같은 꼴로 서버 출력(stderr, docker compose logs)에 나오고, 다른 이름의
# 로거로 쓴 INFO는 처리기가 없어 버려진다. 지금은 시작 때 모델 예열의 결과 한 줄에만 쓴다.
log = logging.getLogger("uvicorn.error")


# 시작 때 모델 예열(SUPPORT_AGENT_PRELOAD=1)을 작업 스레드 하나에서 한다. provider.preload()는 빈 메시지로
# /api/chat을 불러 모델만 올리고(에이전트 호출과 같은 num_ctx·keep_alive) 걸린 ms를 돌려준다(ollama.py).
# - 스레드로 하는 이유: 모델 적재는 이 PC의 D: HDD에서 55초가 걸렸다(docs/service.md). lifespan 안에서
#   기다리면 그동안 서버가 요청을 받지 않아 /healthz도 답하지 못한다. 예열이 끝나기 전에 온 메시지가
#   Ollama에서 그 적재를 기다리는지는 재지 않았다
# - daemon: 종료할 때 예열이 끝나기를 기다리지 않는다. 예열 중에 종료하면 lifespan이 제공자의 연결을 닫으므로
#   예열 요청은 실패해 경고 한 줄로 끝나거나, 그 전에 프로세스와 함께 끝난다
# - 무엇이 실패해도(Ollama가 꺼져 있음, 모델 이름이 틀림, 연결이 닫힘) 잡아서 경고만 남긴다. 예열은 덤이라
#   서비스를 멈출 이유가 없고, 그러면 첫 메시지가 모델을 올린다(예열이 없던 때의 동작)
# 돌려준 스레드는 lifespan이 app.state.preload_thread에 두어, 테스트가 끝나기를 기다릴 수 있게 한다.
def start_preload(provider: OllamaProvider) -> threading.Thread:
    def preload() -> None:
        try:
            ms = provider.preload()
        except Exception as e:
            log.warning(
                "model preload failed, the first message will load the model: %s: %s", type(e).__name__, e
            )
            return
        log.info("model %s preloaded in %.1f s", provider.model, ms / 1000)

    thread = threading.Thread(target=preload, name="model-preload", daemon=True)
    thread.start()
    return thread


# 앱을 만드는 팩토리. 인자를 주지 않으면 환경 변수 설정으로 실제 DB·Ollama·음성 모델을 쓴다.
# provider·engine·voice를 넘기면 그것을 쓴다: 테스트는 대본대로 답하는 가짜 모델과 임시 DB를, offline.py는
# 규칙 대역을 넣는다.
def create_app(
    settings: Settings | None = None,
    *,
    provider: ChatProvider | None = None,
    engine: Engine | None = None,
    voice: VoiceFrontEnd | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    headers = dict(SECURITY_HEADERS)
    # 음성 모드에서만 헤더 두 곳을 연다: 마이크를 같은 출처에 허용하고, 합성한 답(WAV)을 blob: 주소로 재생할
    # 수 있게 CSP에 media-src를 더한다(docs/voice.md).
    if settings.voice or voice:
        headers["Permissions-Policy"] = "camera=(), microphone=(self), geolocation=()"
        # 소리로 읽어 주는 답은 WAV로 받아 blob: 주소로 재생한다.
        headers["Content-Security-Policy"] += "; media-src 'self' blob:"

    # 앱 시작 시 DB(마이그레이션·생성 데이터 포함), 모델 제공자, ChatService, 음성 앞단을 만들고(PRELOAD=1이면
    # 모델 예열도 시작하고) 끝날 때 정리한다.
    # 밖에서 넘겨받은 provider·engine은 호출한 쪽 것이므로 닫지 않고, 여기서 만든 것만 닫는다.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        own_engine = engine or prepare_database(settings.database_url, load_seed=settings.load_seed)
        own_provider = provider or OllamaProvider(
            settings.model,
            num_ctx=settings.num_ctx,
            keep_alive=settings.agent_keep_alive(),
            think=settings.agent_think(),
            base_url=settings.ollama_url,
        )
        app.state.service = ChatService(settings, own_engine, own_provider)
        app.state.voice = voice
        if settings.voice and voice is None:  # 몇 초에서 몇 분: 여기서 음성 모델을 올린다
            app.state.voice = load_voice_front_end(
                settings.tts_device, settings.stt_device, settings.max_audio_seconds
            )
        # 예열은 여기서 만든 OllamaProvider에만 한다. 밖에서 넘겨받은 제공자(테스트의 대본 모델, offline.py의
        # 규칙 대역)는 올릴 모델이 없거나 호출한 쪽 것이다. 음성 모델을 올린 뒤에 시작해서, GPU에 올라가는
        # 순서가 예열이 없을 때(음성 모델이 먼저, 에이전트 모델은 첫 메시지 때)와 같다.
        app.state.preload_thread = None
        if settings.preload and provider is None:
            app.state.preload_thread = start_preload(own_provider)
        yield
        # yield 뒤는 서버 종료 시점이다. 직접 만든 Ollama 연결과 DB 연결 풀만 닫는다.
        if provider is None:
            own_provider.close()
        if engine is None:
            own_engine.dispose()

    # 문서 화면(/docs, /redoc)은 끈다. API 설명(/openapi.json)은 EXPOSE_OPENAPI=1일 때만 내놓는다. 보안
    # 스캐너가 경로와 본문 형태를 정확히 알게 할 때만 켜고 평소에는 꺼 둔다(docs/service.md).
    app = FastAPI(
        title="support-agent",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.expose_openapi else None,
    )

    # 모든 응답에 보안 헤더를 붙인다. setdefault라서 처리기가 직접 정한 헤더는 덮어쓰지 않는다.
    # /api/ 응답은 대화·고객 정보가 들어 있으므로 Cache-Control: no-store로 브라우저·프록시가 저장하지 않게
    # 한다.
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in headers.items():
            response.headers.setdefault(name, value)
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    # 요청 본문·경로·질의가 모델(MessageIn, SpeakIn, DecisionIn, 경로의 승인 번호 등)에 맞지 않을 때의 422
    # 응답. FastAPI 기본 처리기와 같은 꼴({"detail": [{loc, msg, type, ...}]})이지만 오류마다 들어 있던 입력
    # 원문(`input`)을 뺀다. 이유 둘: (1) 본문에 짝 없는 서로게이트(JSON 이스케이프 "\ud800")가 있으면
    # pydantic이 거절한 뒤 기본 처리기가 그 원문을 담은 JSON을 UTF-8로 인코딩하다 UnicodeEncodeError를 내서,
    # 클라이언트 오류가 500과 트레이스백으로 기록됐다(2026-10-24 고침, tests/test_service.py). (2) 고객이 보낸
    # 글을 오류 응답에 그대로 되돌려 보낼 이유가 없다. 어느 칸이 왜 틀렸는지는 그대로 알려 준다.
    # jsonable_encoder는 기본 처리기와 같이 loc의 튜플이나 ctx의 예외 객체를 JSON으로 바꾼다.
    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{key: value for key, value in error.items() if key != "input"} for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})

    # 의존성: lifespan에서 만든 ChatService를 꺼낸다.
    def service(request: Request) -> ChatService:
        return request.app.state.service

    # 관리자 API의 인증 의존성. X-Admin-Token 헤더를 설정의 토큰과 비교한다.
    # - 토큰이 설정되지 않았으면 관리자 API 전체를 503으로 끈다(빈 토큰으로 통과되는 일이 없게)
    # - compare_digest는 비교 시간이 내용과 무관해, 응답 시간으로 토큰을 한 글자씩 알아내는 공격을 막는다
    # 쿠키가 아니라 사용자 정의 헤더라서 다른 사이트의 페이지가 이 헤더를 붙여 요청을 보낼 수 없다(CORS 허용
    # 없음). HawkScan의 CSRF 항목을 오탐으로 분류한 근거다(docs/service.md 스캔 1).
    def admin(x_admin_token: str = Header(default="")) -> None:
        if not settings.admin_token:
            raise HTTPException(503, "the admin API is disabled (SUPPORT_AGENT_ADMIN_TOKEN is not set)")
        if not secrets.compare_digest(x_admin_token.encode(), settings.admin_token.encode()):
            raise HTTPException(401, "wrong admin token")

    # ------------------------------------------------------------ 화면 (pages)

    # --- 화면: 채팅(index.html)과 관리(admin.html). API 설명에는 넣지 않는다 ---
    @app.get("/", include_in_schema=False)
    def chat_page() -> FileResponse:
        return FileResponse(str(STATIC / "index.html"))

    @app.get("/admin", include_in_schema=False)
    def admin_page() -> FileResponse:
        return FileResponse(str(STATIC / "admin.html"))

    # 상태 확인. Dockerfile의 HEALTHCHECK가 부르고, 어떤 모델·think·규정 모드·음성으로 떠 있는지 보여
    # 준다(모델 교체 뒤 실제 설정 확인용).
    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "model": settings.model,
            "think": settings.think,
            "policy": settings.policy,
            "voice": getattr(app.state, "voice", None) is not None,
        }

    # ------------------------------------------------------------ 채팅 (chat)

    # 새 상담 세션. 응답의 session_id(추측할 수 없는 난수)가 이후 모든 고객 요청의 열쇠다. 화면은 이것을
    # sessionStorage에 둔다.
    @app.post("/api/sessions", status_code=201)
    def new_session(svc: ChatService = Depends(service)) -> dict[str, Any]:
        return svc.new_session()

    # 고객이 본 대화(고객 말과 전달된 답만)와 대기 중인 승인. 고객 화면이 이것을 주기적으로 다시 읽어,
    # 담당자가 승인한 결과 문장을 보여 준다(docs/service.md).
    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str, svc: ChatService = Depends(service)) -> dict[str, Any]:
        try:
            return svc.transcript(session_id)
        except SessionNotFoundError:
            raise HTTPException(404, "no such session") from None

    # 스트림을 시작하기 전에 세션이 있고(404) 열려 있는지(409) 확인한다. 이 오류들을 SSE 이벤트가 아니라 HTTP
    # 상태 코드로 돌려주려는 것이다. 상태 열 하나만 읽는다(core.ChatService.session_status): 전에는 대화
    # 전체와 승인 대기열까지 읽는 transcript로 상태를 봤다. 동기 함수라 글 메시지(post_message)에서는
    # FastAPI가 작업 스레드에서 돌리고, async인 post_voice는 run_in_threadpool로 같은 풀에 보낸다.
    def open_session(svc: ChatService, session_id: str) -> None:
        try:
            status = svc.session_status(session_id)
        except SessionNotFoundError:
            raise HTTPException(404, "no such session") from None
        if status != "open":
            raise HTTPException(409, f"the session is {status}")

    # 턴마다 모델이 답할 때까지 작업 스레드 하나를 잡고 있다. 동시에 도는 턴이 이 수를 넘으면 기다리지 않고
    # 바로 503과 Retry-After로 거절한다(tests/test_review_t7.py).
    slots = threading.BoundedSemaphore(settings.max_concurrent_turns)

    # 한 턴을 작업 스레드에서 돌리고 그 진행 이벤트를 SSE로 흘리는 공통 함수. 글 메시지와 음성 메시지가 함께
    # 쓴다.
    # 응답을 글자 단위가 아니라 이벤트 단위로 흘리는 이유: 에이전트의 답은 고객에게 가기 전에 검사(본문에 샌
    # 도구 호출, 잘린 응답, 언어 가드)를 거쳐야 하므로 다 만들어진 뒤에만 보낼 수 있다. 대신 도구를 부를
    # 때마다 진행 이벤트를 보낸다(docs/service.md).
    def turn_stream(
        svc: ChatService, session_id: str, read_message, first_stage: str = "thinking"
    ) -> StreamingResponse:
        """작업 스레드에서 한 턴을 돌리고 그 이벤트를 흘려보낸다. `read_message(emit)`는 고객의 글과 그 글이
        들어온 방식을 주고, 답할 것이 없으면 None을 준다."""
        # 동시에 도는 턴이 한도(기본 4)에 닿았으면 기다리지 않고 바로 503과 Retry-After(5초)로 돌려보낸다.
        # 기다리게 하면 작업 스레드가 계속 쌓이기 때문이다.
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "too many conversations at once, try again", {"Retry-After": "5"})
        # 작업 스레드 → 응답 생성기로 이벤트를 넘기는 큐. None은 "더 없음" 신호다.
        events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()

        def emit(event: str, data: dict[str, Any]) -> None:
            events.put((event, data))

        # 작업 스레드에서 도는 본체. 어떤 예외든 고객용 error 이벤트와 end로 바꾸고, finally에서 동시 실행
        # 자리를 반드시 돌려준다.
        def work() -> None:
            # 브라우저가 떠나도 턴은 끝까지 간다: 도구가 쓴 것은 그것을 낳은 대화와 함께 기록되어야 한다.
            try:
                message = read_message(emit)
                if message is None:
                    events.put(("end", {"status": "open"}))
                    return
                text, via = message
                svc.handle(session_id, text, emit, via=via)
            except BusyError:
                # 같은 세션의 앞 메시지가 아직 처리 중이다(세션별 잠금, core.ChatService.handle).
                events.put(("error", {"message": "앞선 메시지에 답하는 중입니다. 잠시 후 다시 보내 주세요."}))
                events.put(("end", {"status": "open"}))
            except SessionClosedError as exc:
                # 그 사이 세션이 닫혔다(handoff·closed). 화면이 상태를 보고 입력을 막는다.
                events.put(("end", {"status": str(exc)}))
            except Exception:  # noqa: BLE001
                # 예상 못 한 예외는 내용을 고객에게 보이지 않고 고정 문장만 보낸다.
                events.put(("error", {"message": "처리 중 문제가 생겼습니다."}))
                events.put(("end", {"status": "open"}))
            finally:
                slots.release()
                events.put(None)

        # 스레드를 응답을 읽기 시작할 때가 아니라 여기서 바로 띄운다. 클라이언트가 응답을 읽지 않고 끊어도
        # work의 finally가 자리를 돌려주고, 턴은 끝까지 가서 대화 기록이 저장된다(도구가 DB를 바꿨는데 기록이
        # 없는 상태를 막음, docs/service.md).
        threading.Thread(target=work, daemon=True).start()

        # 응답 본문 생성기. 첫 이벤트로 단계(thinking 또는 음성이면 listening)를 바로 보내 화면이 진행 표시를
        # 띄우게 하고, 이후 큐의 이벤트를 None이 올 때까지 SSE로 바꿔 흘린다.
        def stream() -> Iterator[str]:
            yield sse("status", {"stage": first_stage})
            while (item := events.get()) is not None:
                yield sse(*item)

        # X-Accel-Buffering: no는 nginx 같은 앞단 프록시가 SSE를 모아 두었다 한꺼번에 보내지 않게 하는 헤더다.
        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    # 고객이 입력한 글 메시지 한 개를 처리한다. 앞뒤 공백을 뺀 길이로 설정 한도를 검사해 422로 거절하고,
    # 세션을 확인한 뒤 턴을 시작한다.
    @app.post("/api/sessions/{session_id}/messages")
    def post_message(session_id: str, body: MessageIn, svc: ChatService = Depends(service)):
        text = body.text.strip()
        if not text or len(text) > settings.max_message_chars:
            raise HTTPException(422, f"the message must have 1 to {settings.max_message_chars} characters")
        open_session(svc, session_id)
        return turn_stream(svc, session_id, lambda _emit: (text, None))

    # ------------------------------------------------------------ 음성 (voice)

    # --- 음성(선택 기능, SUPPORT_AGENT_VOICE=1) ---
    # 음성 앞단이 없으면 음성 끝점은 503으로 답한다.
    def voice_front_end(request: Request) -> VoiceFrontEnd:
        front_end = getattr(request.app.state, "voice", None)
        if front_end is None:
            raise HTTPException(503, "voice is not enabled (SUPPORT_AGENT_VOICE)")
        return front_end

    # 녹음 한 개를 받아 인식한 뒤 글 메시지와 같은 턴으로 처리한다(인식된 문장은 heard 이벤트로 먼저 보냄).
    # async로 두고 본문을 조각마다 읽어, 크기 한도(기본 5 MB)를 넘는 순간 413으로 멈춘다. 다 받은 뒤에 재면 큰
    # 파일이 메모리에 다 올라온다.
    # 턴 자리(slots)는 본문을 다 받은 뒤 turn_stream에서 잡는다. 자리를 먼저 잡으면 느린 업로드가 턴 자리를
    # 쥐어 글 채팅까지 막는 새 경로가 생기므로 순서를 바꾸지 않는다. 그래서 동시 연결 수 × 5 MB까지 메모리에
    # 오를 수 있고, 이것은 다른 끝점의 본문과 함께 앞단 프록시가 막을 일이다(docs/service.md "하지 않은 것").
    @app.post("/api/sessions/{session_id}/voice")
    async def post_voice(session_id: str, request: Request, svc: ChatService = Depends(service)):
        """본문은 녹음한 발화 하나다(브라우저가 녹음하는 형식 무엇이든: webm, ogg, wav, mp4)."""
        front_end = voice_front_end(request)
        # 세션 확인은 DB 조회라 이벤트 루프 스레드에서 하지 않는다: 루프가 DB 왕복 동안 멈추면 다른 세션의 SSE
        # 이벤트 전달과 음성 본문 수신까지 함께 멈춘다(2026-10-24 고침, tests/test_service.py).
        await run_in_threadpool(open_session, svc, session_id)
        audio = bytearray()
        async for chunk in request.stream():  # 한도를 넘는 순간 읽기를 멈춘다
            audio += chunk
            if len(audio) > settings.max_audio_bytes:
                raise HTTPException(413, f"at most {settings.max_audio_bytes} bytes of audio")
        if not audio:
            raise HTTPException(422, "the body is empty")

        # 작업 스레드 안에서 인식을 돌리고 (에이전트에게 줄 글, 감사 로그용 정보)를 돌려준다. None이면 턴을
        # 쓰지 않고 끝난다.
        # 인식에 실패하거나 너무 길거나 비어 있으면 고객에게 다시 말해 달라고 하고 턴 수를 늘리지 않는다.
        def read_message(emit):
            try:
                heard, text = front_end.listen(bytes(audio))
            except AudioTooLongError:
                emit("error", {"message": "녹음이 너무 깁니다. 짧게 나누어 다시 말씀해 주세요."})
                return None
            except Exception:  # noqa: BLE001 - 디코딩할 수 없는 소리이거나 모델 실패
                emit("error", {"message": "음성을 처리하지 못했습니다. 다시 말씀해 주세요."})
                return None
            text = text.strip()
            if len(text) > settings.max_message_chars:
                # 절대 잘라 쓰지 않는다: 문장 끝이 뜻을 뒤집을 수 있다("... 취소하지 마세요").
                emit("error", {"message": "말씀이 너무 깁니다. 짧게 나누어 다시 말씀해 주세요."})
                return None
            # 에이전트에게 가는 글(고친 글)을 화면에 먼저 보여 준다. 고객이 잘못 들린 것을 바로 알 수 있다.
            emit("heard", {"text": text})
            if not text:
                emit("error", {"message": "잘 들리지 않았습니다. 다시 말씀해 주세요."})
                return None
            # via: 음성으로 들어왔다는 표시와 인식 원문, 녹음 크기. 감사 로그의 customer_message에 함께
            # 남는다.
            return text, {"voice": True, "heard": heard, "audio_bytes": len(audio)}

        return turn_stream(svc, session_id, read_message, first_stage="listening")

    # 동시에 도는 답 읽어 주기(합성) 수의 한도. 턴 자리(slots)와 따로 둔다: /speech는 턴을 쓰지 않아 그 한도에
    # 걸리지 않았고, 음성 모델은 한 번에 하나만 돌리므로(VoiceFrontEnd의 잠금) 겹친 합성 요청이 모두 작업
    # 스레드를 쥔 채 그 잠금을 기다렸다. 세션 하나로 /speech를 수십 개 겹쳐 보내면 같은 스레드 풀(기본 40)을
    # 쓰는 채팅·대화 조회·관리자 API·SSE 생성기까지 멈출 수 있었다(2026-10-24 고침, tests/test_review_t7.py).
    speech_slots = threading.BoundedSemaphore(settings.max_concurrent_speech)

    # 이 세션에서 에이전트가 한 답(또는 그 일부)만 소리로 만들어 준다. 공백을 모두 뺀 뒤 어느 답의 부분
    # 문자열인지 확인해, 아니면 403이다. 세션 번호만 있으면 아무 글이나 합성하는 음성 서비스가 되지 않게
    # 하려는 것이다(docs/voice.md).
    # 소리로 낼 것이 없으면 204를 돌려준다.
    @app.post("/api/sessions/{session_id}/speech")
    def speak(session_id: str, body: SpeakIn, request: Request, svc: ChatService = Depends(service)):
        """이 세션의 답을 소리로 읽어 준다(WAV). 이 세션에서 에이전트가 한 말(전체 또는 일부)만 합성하므로,
        이 끝점은 아무 글이나 읽어 주는 음성 서비스가 아니다."""
        front_end = voice_front_end(request)
        try:
            said = [m["text"] for m in svc.transcript(session_id)["messages"] if m["role"] == "agent"]
        except SessionNotFoundError:
            raise HTTPException(404, "no such session") from None
        if len(body.text) > settings.max_tts_chars:
            raise HTTPException(422, f"at most {settings.max_tts_chars} characters")
        wanted = "".join(body.text.split())
        if not wanted or not any(wanted in "".join(text.split()) for text in said):
            raise HTTPException(403, "only replies of this session are read aloud")
        # 합성 자리가 없으면 기다리지 않고 바로 429와 Retry-After(2초)로 돌려보낸다. 503이 아닌 이유:
        # 화면(chat.js)은 503을 '음성 기능이 꺼졌다'로 읽어 음성 막대를 숨기므로, 잠깐 붐빈 것으로 고객의 음성
        # 모드가 꺼진다. 화면은 그 시간 뒤 한 번만 다시 청한다. 자리는 합성이 예외로 끝나도 finally에서 반드시
        # 돌려준다.
        if not speech_slots.acquire(blocking=False):
            raise HTTPException(
                429, "too many replies are being read aloud at once, try again", {"Retry-After": "2"}
            )
        try:
            wav = front_end.speak(body.text)
        finally:
            speech_slots.release()
        if not wav:
            return Response(status_code=204)
        return Response(wav, media_type="audio/wav")

    # ------------------------------------------------------------ 관리자 (admin)

    # --- 관리자 API: 모두 admin 의존성(X-Admin-Token)을 거친다 ---
    # 승인 대기열 목록. status로 거를 수 있다(관리 화면은 기본으로 pending만 읽고, 전체 보기를 고르면 전부
    # 읽는다).
    @app.get("/api/admin/approvals", dependencies=[Depends(admin)])
    def list_approvals(
        status: str | None = None, svc: ChatService = Depends(service)
    ) -> list[dict[str, Any]]:
        return svc.approvals(status)

    # 승인 또는 거절. 승인이면 그 자리에서 도구를 실행한다. 없는·이미 결정된 승인이거나, 같은 세션의 턴이 30초
    # 안에 끝나지 않아 잠금을 얻지 못하면 409다.
    @app.post("/api/admin/approvals/{approval_id}/decision", dependencies=[Depends(admin)])
    def decide(
        body: DecisionIn,
        # 범위를 둔다: DB 정수 범위를 넘는 값은 서버 오류가 아니라 클라이언트 오류(422)다.
        approval_id: int = Path(ge=1, le=2_147_483_647),
        svc: ChatService = Depends(service),
    ) -> dict[str, Any]:
        try:
            return svc.decide(approval_id, approve=body.approve, by=body.by, note=body.note)
        except ApprovalStateError as exc:
            raise HTTPException(409, str(exc)) from None
        except BusyError:
            raise HTTPException(409, "the session is busy, try again") from None

    # 최근 상담 세션 목록(최근에 바뀐 순).
    @app.get("/api/admin/sessions", dependencies=[Depends(admin)])
    def list_sessions(svc: ChatService = Depends(service)) -> list[dict[str, Any]]:
        return svc.sessions()

    # 한 세션의 고객이 본 대화와 감사 로그 전체.
    @app.get("/api/admin/sessions/{session_id}", dependencies=[Depends(admin)])
    def session_detail(session_id: str, svc: ChatService = Depends(service)) -> dict[str, Any]:
        try:
            return {"transcript": svc.transcript(session_id), "audit": svc.audit(session_id)}
        except SessionNotFoundError:
            raise HTTPException(404, "no such session") from None

    # 화면이 불러오는 js·css를 /static 아래로 내보낸다.
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    return app
