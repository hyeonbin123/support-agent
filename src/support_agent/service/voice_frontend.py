"""Speech in front of the chat: recognise what the customer said, read the agent's reply aloud.

Optional (SUPPORT_AGENT_VOICE=1, needs the `voice` dependency group). The chat itself does not change: a
recognised utterance takes the same path as a typed message.
"""

# [파일 역할] 채팅 서비스의 음성 앞단. 고객의 녹음을 글로 바꾸고(인식 → 표기 고침), 에이전트의 답을 소리로
# 바꾼다(말로 풀어 쓰기 → 합성).
# [설계 이유] 대화 처리(core.py)는 건드리지 않고 앞뒤에만 붙인다. 인식된 문장은 글로 보낸 메시지와 같은 길을
# 가므로 규정 검사, 사람 승인, 감사 로그가 그대로 적용된다(docs/service.md "음성").
# [연결] service/app.py의 /voice·/speech 끝점이 listen·speak를 부르고, create_app이 SUPPORT_AGENT_VOICE=1일 때
# load_voice_front_end로 실제 모델을 올린다. service/offline.py는 같은 클래스에 모델 없는 대역을 끼운다.

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

# Listener·Speaker: 인식기·합성기가 지켜야 할 인터페이스(실제 모델과 오프라인 대역이 모두 따른다).
# normalize_heard: 5단계 측정에서 효과가 확인된 표기 규칙(V2). spoken_to_written: 사람이 한글로 부른
# 전화번호·주문 번호를 숫자로 되돌린다(실제 마이크 시험에서 나온 문제, 측정과 무관). verbalize: 합성 전에
# 숫자·기호를 사람이 말하는 한글로 풀어 쓴다(docs/voice.md).
from support_agent.voice.channel import Listener, Speaker
from support_agent.voice.normalize import normalize_heard
from support_agent.voice.spoken import spoken_to_written
from support_agent.voice.verbalize import verbalize


# 합성기·인식기·정규화 함수를 묶은 음성 앞단. normalizer가 None이면 인식 결과를 그대로 에이전트에
# 준다(오프라인 대역).
class VoiceFrontEnd:
    def __init__(self, speaker: Speaker, listener: Listener, normalizer: Callable[[str], str] | None = None):
        self.speaker, self.listener, self.normalizer = speaker, listener, normalizer
        # 요청은 여러 작업 스레드에서 동시에 올 수 있지만 음성 모델은 스레드 안전하게 만들어지지 않았으므로,
        # 인식과 합성을 합쳐 한 번에 하나만 돌린다(docs/voice.md 구성 표 "모델 호출은 한 번에 하나").
        self._lock = threading.Lock()  # one model call at a time: the models are not built for threads

    # 합성기·인식기의 describe() 결과(모델 이름, 장치 등)를 한 dict로 합친다.
    def describe(self) -> dict[str, Any]:
        return {**self.speaker.describe(), **self.listener.describe()}

    # 녹음 바이트 → (인식 원문, 에이전트가 받을 글). 인식 원문은 감사 로그의 customer_message.via에 남기고,
    # 에이전트에게는 고친 글을 준다.
    # 잠금은 모델을 부르는 인식 단계에만 건다. 정규화는 순수 함수라 잠금 밖에서 돈다. 너무 긴 녹음이면
    # 인식기가 AudioTooLongError를 내고 app.py가 안내 문장으로 바꾼다.
    def listen(self, audio: bytes) -> tuple[str, str]:
        """(what the recogniser wrote, what the agent receives)."""
        with self._lock:
            heard = self.listener.transcribe(audio)
        return heard, (self.normalizer(heard) if self.normalizer else heard)

    # 답 글을 말로 풀어 쓴 뒤 합성한 WAV 바이트를 돌려준다. 합성 seed는 0으로 고정해 같은 글은 같은 소리가
    # 된다.
    # 풀어 쓴 글에 한글 음절이 하나도 없으면(기호나 영문만 남은 경우) 합성하지 않고 b""를 돌려주며, app.py는
    # 이때 204(내용 없음)로 답한다.
    def speak(self, text: str) -> bytes:
        """A WAV file of `text` read aloud; b"" when nothing in it can be pronounced."""
        spoken = verbalize(text)
        if not any("가" <= ch <= "힣" for ch in spoken):
            return b""
        with self._lock:
            return self.speaker.synthesize(spoken, 0).wav


# 서비스에서 인식 결과에 적용하는 고침 순서: 먼저 V2 표기 규칙(주문 번호의 O, 이메일의 골뱅이와 도메인,
# 전화번호 구분, 만 단위 금액), 그다음 사람이 부른 번호를 숫자로. load_voice_front_end가 normalizer로 넘긴다.
def repair_heard(text: str) -> str:
    """What the agent receives instead of the raw transcript: the notation rules that the measurement
    adopted (V2, docs/experiments.md stage 5), then the numbers a person read aloud as words."""
    return spoken_to_written(normalize_heard(text))


# 실제 음성 모델(MeloTTS 합성, faster-whisper 인식)을 올려 앞단을 만든다. 모델을 읽는 데 수 초~수 분이 걸려 앱
# 시작(lifespan)에서 한 번만 부른다.
# voice.speech는 함수 안에서 가져온다. 음성 의존성 그룹(voice)이 없는 환경에서도 이 모듈과 서비스 전체를
# 불러올 수 있게 하려는 것이다.
def load_voice_front_end(tts_device: str, stt_device: str, max_audio_seconds: float) -> VoiceFrontEnd:
    from support_agent.voice.speech import MeloSpeaker, WhisperListener

    # max_seconds: 풀었을 때 이 길이(기본 60초)를 넘는 녹음은 거절한다. vad_filter=True: 말이 없는 구간을 먼저
    # 걸러 낸다. 실제 마이크로 해 보니 말없이 버튼만 눌렀다 뗀 녹음을 Whisper가 "다음 영상에서 만나요."로
    # 적었기 때문이다. 측정용 채널은 합성 음성이라 이 옵션을 끈 채로 둔다(docs/voice.md).
    listener = WhisperListener(device=stt_device, max_seconds=max_audio_seconds, vad_filter=True)
    return VoiceFrontEnd(MeloSpeaker(device=tts_device), listener, normalizer=repair_heard)
