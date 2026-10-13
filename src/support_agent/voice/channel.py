"""The speech channel: what the customer says goes through synthesis and recognition before the agent
reads it. The customer's side of the conversation keeps the words as they were meant."""

# [파일 역할] 5단계 음성 채널. 시뮬레이터 고객이 쓴 글을 말로 풀어 쓰고(verbalize) → 음성 합성 → 음성 인식 →
# (V2·V4면 규칙 정규화)를 거친 글만 에이전트에게 준다.
# [설계 이유] 고객 → 에이전트 방향만 바꾼다. 시뮬레이터는 자기가 쓴 글을 그대로 기억하므로, 텍스트 실행과의
# 차이는 "에이전트가 잘못 들었다"에서만 나온다(docs/voice.md "정한 것과 이유").
# 같은 입력이면 같은 소리·같은 받아쓰기가 나오도록 합성 seed를 발화마다 고정하므로, 왕복 결과를 JSONL에 캐시해
# 같은 실행을 음성 모델 없이 다시 돌리거나 분석할 수 있다.
# [연결] run.py의 build_channel이 실제 모델(MeloSpeaker, WhisperListener)과 캐시
# 경로(outputs/voice-cache/roundtrips.jsonl)로 만들고, episode.py가 고객 발화마다 hear()를 부른다.
# Speaker·Listener 인터페이스는 서비스의 음성 앞단(service/voice_frontend.py)도 쓴다.

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

# Audio: 합성 결과(WAV 바이트와 길이). timed: 함수를 실행하고 걸린 시간(ms)을 함께 돌려준다. verbalize:
# "O-91010" → "오 다시 구일공일공"처럼 사람이 말하는 한글로 풀어 쓴다.
from support_agent.voice.speech import Audio, timed
from support_agent.voice.verbalize import verbalize


# 발화 하나의 왕복 기록: 고객이 뜻한 글(said), 합성한 말(spoken), 인식된 글(heard), 에이전트가 받은 글(text),
# 음성 길이와 합성·인식 시간, 캐시에서 왔는지. episode.py가 to_dict()로 실행 기록의 voice 항목에 남긴다(음성
# 파일은 남기지 않음).
@dataclass(frozen=True)
class Heard:
    said: str  # what the customer meant, in writing
    spoken: str  # the words that were synthesised
    heard: str  # what the recogniser wrote
    text: str  # what the agent receives: `heard`, or `heard` after the normaliser
    audio_seconds: float = 0.0
    tts_ms: float = 0.0
    stt_ms: float = 0.0
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# 합성기와 인식기가 지킬 최소 인터페이스. Protocol이라 상속 없이 메서드만 맞으면 된다(실제 모델, 테스트의
# 가짜, 서비스 오프라인 대역). describe()는 모델·장치·버전 같은 정체를 돌려주고 캐시 키와 실행 기록에
# 들어간다.
class Speaker(Protocol):
    def synthesize(self, spoken_text: str, seed: int = 0) -> Audio: ...
    def describe(self) -> dict[str, Any]: ...


class Listener(Protocol):
    def transcribe(self, audio: bytes) -> str: ...
    def describe(self) -> dict[str, Any]: ...


# 합성기·인식기·정규화기와 왕복 캐시를 묶은 채널.
class SpeechChannel:
    """Text -> spoken form -> audio -> recognised text (-> normaliser).

    Same spoken text and seed give the same audio and the same transcript, so results are kept in a JSONL
    cache: a run can be repeated, or analysed again, without the speech models.
    """

    def __init__(
        self,
        speaker: Speaker,
        listener: Listener,
        *,
        normalizer: Callable[[str], str] | None = None,
        cache_path: Path | None = None,
    ):
        self.speaker, self.listener, self.normalizer = speaker, listener, normalizer
        # 캐시 키에 넣을 채널의 정체: 합성기·인식기의 describe()(모델 이름, 장치, 옵션, 패키지 버전). 모델이나
        # 버전이 바뀌면 키가 달라져 예전 결과를 쓰지 않는다(docs/voice.md).
        self._identity = json.dumps([speaker.describe(), listener.describe()], sort_keys=True)
        self._cache_path = cache_path
        self._cache: dict[str, dict[str, Any]] = {}
        # 시작할 때 캐시 파일(JSONL, 한 줄에 한 왕복)을 모두 읽어 키 → 기록 dict로 둔다.
        if cache_path and cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                self._cache[entry["key"]] = entry

    # 실행 manifest에 남길 채널 설명. 정규화기는 함수 이름으로 적어 V1(없음)과 V2·V4(normalize_heard)를
    # 구분한다.
    def describe(self) -> dict[str, Any]:
        return {
            **self.speaker.describe(),
            **self.listener.describe(),
            "normalizer": getattr(self.normalizer, "__name__", None),
        }

    # 고객 글 하나를 채널에 통과시킨다. seed는 episode.py가 (기본 seed, 과제, 시도, "voice", 발화 순번)에서
    # 만들어 넘긴다.
    def hear(self, said: str, seed: int = 0) -> Heard:
        spoken = verbalize(said)
        # 풀어 쓴 글에 한글 음절이 없으면("..."나 이모지만) 합성할 소리가 없으므로 아무것도 들리지 않은 것으로
        # 돌려준다. 에이전트에게는 episode.py가 "들리지 않음" 표기를 대신 준다.
        if not re.search("[가-힣]", spoken):  # nothing pronounceable ("...", an emoji): nothing is heard
            return Heard(said, spoken, "", "")
        # 캐시 키 = sha256(채널 정체 | seed | 합성할 말). 정규화기는 키에 넣지 않고 캐시에는 인식 원문만 둔다.
        # 그래서 정규화 규칙을 바꾸거나 V1과 V2를 바꿔 돌려도 같은 왕복을 다시 쓴다.
        key = hashlib.sha256(f"{self._identity}|{seed}|{spoken}".encode()).hexdigest()
        entry = self._cache.get(key)
        cached = entry is not None
        # 캐시에 없으면 실제로 합성하고 인식한 뒤, 메모리 캐시에 넣고 파일 끝에 한 줄을 덧붙인다(덧붙이기만
        # 하므로 중간에 멈춰도 앞의 기록은 남는다).
        if entry is None:
            audio, tts_ms = timed(self.speaker.synthesize, spoken, seed)
            heard, stt_ms = timed(self.listener.transcribe, audio.wav)
            entry = {
                "key": key,
                "spoken": spoken,
                "heard": heard,
                "audio_seconds": round(audio.seconds, 3),
                "tts_ms": round(tts_ms, 1),
                "stt_ms": round(stt_ms, 1),
            }
            self._cache[key] = entry
            if self._cache_path:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self._cache_path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # 정규화는 캐시 밖에서 매번 적용한다. 에이전트가 받는 글은 정규화 결과(V2·V4)거나 인식 원문
        # 그대로(V1)다.
        heard = entry["heard"]
        text = self.normalizer(heard) if self.normalizer else heard
        return Heard(
            said, spoken, heard, text, entry["audio_seconds"], entry["tts_ms"], entry["stt_ms"], cached
        )
