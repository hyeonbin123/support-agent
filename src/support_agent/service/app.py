"""The web layer: chat over server-sent events, the admin API and two static pages.

Run: uv run uvicorn support_agent.service.app:create_app --factory --port 8062
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
# secrets: 관리자 토큰을 상수 시간으로 비교한다. importlib.resources.files: 패키지 안의 static/ 폴더를 설치
# 위치와 상관없이 찾는다.
import json
import queue
import secrets
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any

# FastAPI: 라우팅, 의존성 주입(Depends), 요청 본문 검증(pydantic). StreamingResponse로 SSE를, StaticFiles로
# 화면의 js·css를 내보낸다.
from fastapi import Depends, FastAPI, Header, HTTPException, Path, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import Engine

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
        # Replies read aloud are fetched as WAV and played from a blob: URL.
        headers["Content-Security-Policy"] += "; media-src 'self' blob:"

    # 앱 시작 시 DB(마이그레이션·생성 데이터 포함), 모델 제공자, ChatService, 음성 앞단을 만들고 끝날 때
    # 정리한다.
    # 밖에서 넘겨받은 provider·engine은 호출한 쪽 것이므로 닫지 않고, 여기서 만든 것만 닫는다.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        own_engine = engine or prepare_database(settings.database_url, load_seed=settings.load_seed)
        own_provider = provider or OllamaProvider(
            settings.model,
            num_ctx=settings.num_ctx,
            think=settings.agent_think(),
            base_url=settings.ollama_url,
        )
        app.state.service = ChatService(settings, own_engine, own_provider)
        app.state.voice = voice
        if settings.voice and voice is None:  # several seconds to minutes: the models load here
            app.state.voice = load_voice_front_end(
                settings.tts_device, settings.stt_device, settings.max_audio_seconds
            )
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

    # ------------------------------------------------------------ pages

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

    # ------------------------------------------------------------ chat

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
    # 상태 코드로 돌려주려는 것이다.
    def open_session(svc: ChatService, session_id: str) -> None:
        try:
            status = svc.transcript(session_id)["status"]
        except SessionNotFoundError:
            raise HTTPException(404, "no such session") from None
        if status != "open":
            raise HTTPException(409, f"the session is {status}")

    # Every turn holds a worker thread until the model has answered; more than this many wait outside.
    slots = threading.BoundedSemaphore(settings.max_concurrent_turns)

    # 한 턴을 작업 스레드에서 돌리고 그 진행 이벤트를 SSE로 흘리는 공통 함수. 글 메시지와 음성 메시지가 함께
    # 쓴다.
    # 응답을 글자 단위가 아니라 이벤트 단위로 흘리는 이유: 에이전트의 답은 고객에게 가기 전에 검사(본문에 샌
    # 도구 호출, 잘린 응답, 언어 가드)를 거쳐야 하므로 다 만들어진 뒤에만 보낼 수 있다. 대신 도구를 부를
    # 때마다 진행 이벤트를 보낸다(docs/service.md).
    def turn_stream(
        svc: ChatService, session_id: str, read_message, first_stage: str = "thinking"
    ) -> StreamingResponse:
        """Run one turn in a worker thread and stream its events. `read_message(emit)` gives the
        customer's text and how it arrived, or None when there is nothing to answer."""
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
            # The turn runs to its end even when the browser goes away: what the tools wrote must be
            # recorded together with the conversation that led to it.
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

        # Started here, not when the response is first read: the slot is given back in any case.
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

    # ------------------------------------------------------------ voice

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
    @app.post("/api/sessions/{session_id}/voice")
    async def post_voice(session_id: str, request: Request, svc: ChatService = Depends(service)):
        """The body is one recorded utterance (whatever the browser records: webm, ogg, wav, mp4)."""
        front_end = voice_front_end(request)
        open_session(svc, session_id)
        audio = bytearray()
        async for chunk in request.stream():  # stop reading as soon as the limit is passed
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
            except Exception:  # noqa: BLE001 - undecodable audio or a model failure
                emit("error", {"message": "음성을 처리하지 못했습니다. 다시 말씀해 주세요."})
                return None
            text = text.strip()
            if len(text) > settings.max_message_chars:
                # Never cut it: the end of a sentence can turn its meaning around ("... 취소하지 마세요").
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

    # 이 세션에서 에이전트가 한 답(또는 그 일부)만 소리로 만들어 준다. 공백을 모두 뺀 뒤 어느 답의 부분
    # 문자열인지 확인해, 아니면 403이다. 세션 번호만 있으면 아무 글이나 합성하는 음성 서비스가 되지 않게
    # 하려는 것이다(docs/voice.md).
    # 소리로 낼 것이 없으면 204를 돌려준다.
    @app.post("/api/sessions/{session_id}/speech")
    def speak(session_id: str, body: SpeakIn, request: Request, svc: ChatService = Depends(service)):
        """A reply of this session read aloud (WAV). Only what the agent said here is synthesised, in
        whole or in part, so the endpoint is not a speech service for arbitrary text."""
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
        wav = front_end.speak(body.text)
        if not wav:
            return Response(status_code=204)
        return Response(wav, media_type="audio/wav")

    # ------------------------------------------------------------ admin

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
        # Bounded: an integer beyond the database's range is a client error (422), not a server error.
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
