"""The service without a model: a rule-based stand-in answers, everything else is the real code.

For work on the pages, for security scans of the web layer and for smoke tests, where a language model
would only cost GPU time:

    uv run uvicorn support_agent.service.offline:create_offline_app --factory --port 8062

The stand-in verifies a customer when it reads a name with a phone number, looks an order up when it reads
an order number, cancels it when the message also says "취소", and otherwise asks for what is missing.
With SUPPORT_AGENT_VOICE=1 the voice endpoints answer too, through stand-ins that need no speech model: a
"recording" is read as UTF-8 text, and every reply sounds like a short silence.
"""

from __future__ import annotations

import io
import re
import wave

from fastapi import FastAPI

from support_agent.chat import ChatProvider, ChatResponse, ToolCall
from support_agent.service.app import create_app
from support_agent.service.settings import Settings
from support_agent.service.voice_frontend import VoiceFrontEnd
from support_agent.voice.speech import Audio

_NAME_AND_PHONE = re.compile(r"([가-힣]{2,4})(?:이고|입니다|이며|이에요|,|\s).*?(01\d[- ]?\d{3,4}[- ]?\d{4})")
_ORDER = re.compile(r"\bO-\d{5}\b")


class OfflineProvider(ChatProvider):
    def describe(self):
        return {"provider": "offline", "model": "rules"}

    def chat(self, messages, tools=(), *, temperature=0.0, seed=None, max_tokens=1024):
        last = messages[-1]
        if last.role == "tool":
            ok = not last.content.startswith("Error")
            return ChatResponse(("처리했습니다. " if ok else "처리하지 못했습니다. ") + last.content[:300])
        verified = any(
            m.role == "tool" and m.tool_name == "find_customer" and not m.content.startswith("Error")
            for m in messages
        )
        text = last.content
        if not verified:
            found = _NAME_AND_PHONE.search(text)
            if found:
                return ChatResponse("", (ToolCall("find_customer", {"name": found[1], "contact": found[2]}),))
            return ChatResponse("본인 확인을 위해 성함과 가입하신 전화번호를 알려 주세요.")
        order = _ORDER.search(text)
        if order and "취소" in text:
            return ChatResponse(
                "", (ToolCall("cancel_order", {"order_id": order[0], "reason": "changed_mind"}),)
            )
        if order:
            return ChatResponse("", (ToolCall("get_order", {"order_id": order[0]}),))
        return ChatResponse("확인할 주문 번호를 알려 주세요. (모델 없이 도는 오프라인 모드입니다)")


class TextListener:
    """Takes the bytes of a recording for UTF-8 text; anything else cannot be "decoded"."""

    def describe(self):
        return {"stt": "offline"}

    def transcribe(self, audio: bytes) -> str:
        return audio.decode("utf-8").strip()


class SilentSpeaker:
    def describe(self):
        return {"tts": "offline"}

    def synthesize(self, spoken_text: str, seed: int = 0) -> Audio:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16_000)
            out.writeframes(bytes(2 * 3_200))  # 0.2 s of silence
        return Audio(buffer.getvalue(), seconds=0.2)


def create_offline_app() -> FastAPI:
    settings = Settings.from_env()
    voice = VoiceFrontEnd(SilentSpeaker(), TextListener()) if settings.voice else None
    # The stand-ins are passed in, so create_app does not load the speech models although `voice` is set.
    return create_app(settings, provider=OfflineProvider(), voice=voice)
