"""모델 없는 서비스: 규칙 기반 대역이 답하고, 나머지는 모두 실제 코드다.

화면 작업, 웹 계층의 보안 스캔, 스모크 테스트처럼 언어 모델이 GPU 시간만 쓰는 경우를 위한 것이다:

    uv run uvicorn support_agent.service.offline:create_offline_app --factory --port 8062

대역은 이름과 전화번호를 읽으면 고객 본인 확인을 하고, 주문 번호를 읽으면 주문을 조회하며, 메시지에
"취소"도 있으면 그 주문을 취소하고, 그 밖에는 빠진 것을 묻는다. SUPPORT_AGENT_VOICE=1이면 음성 끝점도
음성 모델이 필요 없는 대역으로 답한다: "녹음"은 UTF-8 글로 읽고, 모든 답은 짧은 무음으로 들린다.
"""

# [파일 역할] LLM과 음성 모델 자리에 규칙 몇 줄짜리 대역을 끼운 서비스 앱을 만든다. 웹 계층, 세션·감사
# 로그·승인 대기열, 도구 실행은 실제 코드(app.py, core.py, tools.py) 그대로다.
# [왜 필요한가] 화면 작업, 웹 계층의 보안 스캔(HawkScan, 2026-10-05부터 OWASP ZAP), 스모크 테스트에는 모델의
# 답이 필요 없고 GPU만 쓴다(docs/service.md "모델 없이 띄우기, 보안 스캔"). 이 저장소의 DAST는 모두 이 앱을
# 대상으로 돌렸다.
# [연결] uvicorn이 create_offline_app을 팩토리로 부른다. tests/test_service.py가 OfflineProvider로 실제 웹
# 계층을 거치는 테스트와, 음성 모델 없이 음성 끝점이 응답하는지 테스트한다.

from __future__ import annotations

# io·wave: 음성 합성 대역이 돌려줄 짧은 무음 WAV 파일을 메모리에서 만든다.
import io
import re
import wave

from fastapi import FastAPI

# 실제 서비스와 같은 계약(ChatProvider, ChatResponse, ToolCall)과 같은 create_app·Settings·VoiceFrontEnd를
# 쓴다. 대역만 바꾸고 나머지 경로가 모두 실제 코드를 지나게 하려는 것이다.
from support_agent.chat import ChatProvider, ChatResponse, ToolCall
from support_agent.service.app import create_app
from support_agent.service.settings import Settings
from support_agent.service.voice_frontend import VoiceFrontEnd
from support_agent.voice.speech import Audio

# 본인 확인용 패턴: 2~4자 한글 이름 + 연결 어미(이고·입니다·이며·이에요)나 쉼표·공백 + (중간 글자는 짧게
# 건너뛰고) 01로 시작하는 휴대전화 번호. 하이픈·공백 구분은 있어도 없어도 된다.
_NAME_AND_PHONE = re.compile(r"([가-힣]{2,4})(?:이고|입니다|이며|이에요|,|\s).*?(01\d[- ]?\d{3,4}[- ]?\d{4})")
# 주문 번호 패턴(O-와 숫자 다섯 자리, 예: O-90005). 단어 경계로 앞뒤에 다른 글자가 붙은 것은 받지 않는다.
_ORDER = re.compile(r"\bO-\d{5}\b")


# 모델 대신 규칙으로 답하는 제공자. 실제 에이전트 루프(agent.agent_turn)가 이것을 Ollama 제공자처럼 부른다.
# 규칙(docs/service.md): 이름과 전화번호를 보면 find_customer, 본인 확인 뒤 주문 번호를 보면 get_order, 주문
# 번호와 "취소"가 함께 있으면 cancel_order. 그 밖에는 필요한 것을 묻는 고정 문장으로 답한다.
class OfflineProvider(ChatProvider):
    def describe(self):
        return {"provider": "offline", "model": "rules"}

    # 실제 제공자와 같은 서명을 받지만 temperature·seed·max_tokens는 쓰지 않는다(규칙이라 항상 같은 답).
    def chat(self, messages, tools=(), *, temperature=0.0, seed=None, max_tokens=1024):
        # 직전 메시지가 도구 결과이면 그 결과 글(앞 300자)을 그대로 답으로 옮긴다. 그래서 도구가
        # 에이전트에게만 주는 문구(승인 대기 안내 등)가 답에 섞이고, 서비스가 그것을 고객에게서 걸러 내는지
        # HTTP로 확인할 수 있다(docs/service.md, 8차 스캔 준비).
        last = messages[-1]
        if last.role == "tool":
            ok = not last.content.startswith("Error")
            return ChatResponse(("처리했습니다. " if ok else "처리하지 못했습니다. ") + last.content[:300])
        # 대화 안에 성공한 find_customer 결과가 하나라도 있으면 본인 확인이 끝난 것으로 본다. 실패한 도구
        # 결과는 "Error"로 시작한다(toolkit의 오류 문자열 형식).
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


# 음성 인식 대역. 업로드된 "녹음" 바이트를 UTF-8 글로 읽는다. 스캐너나 테스트가 글을 녹음처럼 보내면 그 글이
# 인식 결과가 된다.
# UTF-8이 아닌 바이트는 디코딩 예외가 나고, app.py의 post_voice가 이것을 "음성을 처리하지 못했습니다" 안내로
# 바꾼다.
class TextListener:
    """녹음 바이트를 UTF-8 글로 받아들인다. 그 밖의 것은 "디코딩"할 수 없다."""

    def describe(self):
        return {"stt": "offline"}

    def transcribe(self, audio: bytes) -> str:
        return audio.decode("utf-8").strip()


# 음성 합성 대역. 어떤 글이든 0.2초 무음 WAV(16 kHz, 모노, 16비트)를 돌려준다.
class SilentSpeaker:
    def describe(self):
        return {"tts": "offline"}

    def synthesize(self, spoken_text: str, seed: int = 0) -> Audio:
        # 16 kHz × 0.2초 = 3,200 표본, 표본당 2바이트이므로 0바이트 6,400개가 무음 데이터다.
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16_000)
            out.writeframes(bytes(2 * 3_200))  # 0.2초 무음
        return Audio(buffer.getvalue(), seconds=0.2)


# uvicorn --factory가 부르는 앱 생성 함수. 설정은 실제 서비스와 같이 환경 변수에서 읽고,
# SUPPORT_AGENT_VOICE=1이면 음성 대역을 붙인다.
def create_offline_app() -> FastAPI:
    settings = Settings.from_env()
    voice = VoiceFrontEnd(SilentSpeaker(), TextListener()) if settings.voice else None
    # 대역을 넘겨주므로, `voice` 설정이 켜져 있어도 create_app은 음성 모델을 올리지 않는다.
    return create_app(settings, provider=OfflineProvider(), voice=voice)
