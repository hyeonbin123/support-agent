"""The web layer: chat over server-sent events, the admin API and two static pages.

Run: uv run uvicorn support_agent.service.app:create_app --factory --port 8062
"""

from __future__ import annotations

import json
import queue
import secrets
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import Engine

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

STATIC = files("support_agent.service") / "static"
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


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class SpeakIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class DecisionIn(BaseModel):
    approve: bool
    by: str = Field(default="admin", min_length=1, max_length=64)
    note: str = Field(default="", max_length=500)


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(
    settings: Settings | None = None,
    *,
    provider: ChatProvider | None = None,
    engine: Engine | None = None,
    voice: VoiceFrontEnd | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    headers = dict(SECURITY_HEADERS)
    if settings.voice or voice:
        headers["Permissions-Policy"] = "camera=(), microphone=(self), geolocation=()"
        # Replies read aloud are fetched as WAV and played from a blob: URL.
        headers["Content-Security-Policy"] += "; media-src 'self' blob:"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        own_engine = engine or prepare_database(settings.database_url, load_seed=settings.load_seed)
        own_provider = provider or OllamaProvider(
            settings.model, num_ctx=settings.num_ctx, base_url=settings.ollama_url
        )
        app.state.service = ChatService(settings, own_engine, own_provider)
        app.state.voice = voice
        if settings.voice and voice is None:  # several seconds to minutes: the models load here
            app.state.voice = load_voice_front_end(settings.tts_device, settings.stt_device)
        yield
        if provider is None:
            own_provider.close()
        if engine is None:
            own_engine.dispose()

    app = FastAPI(title="support-agent", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in headers.items():
            response.headers.setdefault(name, value)
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    def service(request: Request) -> ChatService:
        return request.app.state.service

    def admin(x_admin_token: str = Header(default="")) -> None:
        if not settings.admin_token:
            raise HTTPException(503, "the admin API is disabled (SUPPORT_AGENT_ADMIN_TOKEN is not set)")
        if not secrets.compare_digest(x_admin_token.encode(), settings.admin_token.encode()):
            raise HTTPException(401, "wrong admin token")

    # ------------------------------------------------------------ pages

    @app.get("/", include_in_schema=False)
    def chat_page() -> FileResponse:
        return FileResponse(str(STATIC / "index.html"))

    @app.get("/admin", include_in_schema=False)
    def admin_page() -> FileResponse:
        return FileResponse(str(STATIC / "admin.html"))

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "model": settings.model,
            "policy": settings.policy,
            "voice": getattr(app.state, "voice", None) is not None,
        }

    # ------------------------------------------------------------ chat

    @app.post("/api/sessions", status_code=201)
    def new_session(svc: ChatService = Depends(service)) -> dict[str, Any]:
        return svc.new_session()

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str, svc: ChatService = Depends(service)) -> dict[str, Any]:
        try:
            return svc.transcript(session_id)
        except SessionNotFoundError:
            raise HTTPException(404, "no such session") from None

    def open_session(svc: ChatService, session_id: str) -> None:
        try:
            status = svc.transcript(session_id)["status"]
        except SessionNotFoundError:
            raise HTTPException(404, "no such session") from None
        if status != "open":
            raise HTTPException(409, f"the session is {status}")

    def turn_stream(
        svc: ChatService, session_id: str, read_message, first_stage: str = "thinking"
    ) -> StreamingResponse:
        """Run one turn in a worker thread and stream its events. `read_message(emit)` gives the
        customer's text and how it arrived, or None when there is nothing to answer."""
        events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()

        def emit(event: str, data: dict[str, Any]) -> None:
            events.put((event, data))

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
                events.put(("error", {"message": "앞선 메시지에 답하는 중입니다. 잠시 후 다시 보내 주세요."}))
                events.put(("end", {"status": "open"}))
            except SessionClosedError as exc:
                events.put(("end", {"status": str(exc)}))
            except Exception:  # noqa: BLE001
                events.put(("error", {"message": "처리 중 문제가 생겼습니다."}))
                events.put(("end", {"status": "open"}))
            finally:
                events.put(None)

        def stream() -> Iterator[str]:
            threading.Thread(target=work, daemon=True).start()
            yield sse("status", {"stage": first_stage})
            while (item := events.get()) is not None:
                yield sse(*item)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/sessions/{session_id}/messages")
    def post_message(session_id: str, body: MessageIn, svc: ChatService = Depends(service)):
        text = body.text.strip()
        if not text or len(text) > settings.max_message_chars:
            raise HTTPException(422, f"the message must have 1 to {settings.max_message_chars} characters")
        open_session(svc, session_id)
        return turn_stream(svc, session_id, lambda _emit: (text, None))

    # ------------------------------------------------------------ voice

    def voice_front_end(request: Request) -> VoiceFrontEnd:
        front_end = getattr(request.app.state, "voice", None)
        if front_end is None:
            raise HTTPException(503, "voice is not enabled (SUPPORT_AGENT_VOICE)")
        return front_end

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

        def read_message(emit):
            try:
                heard, text = front_end.listen(bytes(audio))
            except Exception:  # noqa: BLE001 - undecodable audio or a model failure
                emit("error", {"message": "음성을 처리하지 못했습니다. 다시 말씀해 주세요."})
                return None
            text = text.strip()[: settings.max_message_chars]
            emit("heard", {"text": text})
            if not text:
                emit("error", {"message": "잘 들리지 않았습니다. 다시 말씀해 주세요."})
                return None
            return text, {"voice": True, "heard": heard, "audio_bytes": len(audio)}

        return turn_stream(svc, session_id, read_message, first_stage="listening")

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

    @app.get("/api/admin/approvals", dependencies=[Depends(admin)])
    def list_approvals(
        status: str | None = None, svc: ChatService = Depends(service)
    ) -> list[dict[str, Any]]:
        return svc.approvals(status)

    @app.post("/api/admin/approvals/{approval_id}/decision", dependencies=[Depends(admin)])
    def decide(approval_id: int, body: DecisionIn, svc: ChatService = Depends(service)) -> dict[str, Any]:
        try:
            return svc.decide(approval_id, approve=body.approve, by=body.by, note=body.note)
        except ApprovalStateError as exc:
            raise HTTPException(409, str(exc)) from None
        except BusyError:
            raise HTTPException(409, "the session is busy, try again") from None

    @app.get("/api/admin/sessions", dependencies=[Depends(admin)])
    def list_sessions(svc: ChatService = Depends(service)) -> list[dict[str, Any]]:
        return svc.sessions()

    @app.get("/api/admin/sessions/{session_id}", dependencies=[Depends(admin)])
    def session_detail(session_id: str, svc: ChatService = Depends(service)) -> dict[str, Any]:
        try:
            return {"transcript": svc.transcript(session_id), "audit": svc.audit(session_id)}
        except SessionNotFoundError:
            raise HTTPException(404, "no such session") from None

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    return app
