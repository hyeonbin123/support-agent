"""The speech channel: what the customer says goes through synthesis and recognition before the agent
reads it. The customer's side of the conversation keeps the words as they were meant."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from support_agent.voice.speech import Audio, timed
from support_agent.voice.verbalize import verbalize


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


class Speaker(Protocol):
    def synthesize(self, spoken_text: str, seed: int = 0) -> Audio: ...
    def describe(self) -> dict[str, Any]: ...


class Listener(Protocol):
    def transcribe(self, audio: bytes) -> str: ...
    def describe(self) -> dict[str, Any]: ...


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
        self._identity = json.dumps([speaker.describe(), listener.describe()], sort_keys=True)
        self._cache_path = cache_path
        self._cache: dict[str, dict[str, Any]] = {}
        if cache_path and cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                self._cache[entry["key"]] = entry

    def describe(self) -> dict[str, Any]:
        return {
            **self.speaker.describe(),
            **self.listener.describe(),
            "normalizer": getattr(self.normalizer, "__name__", None),
        }

    def hear(self, said: str, seed: int = 0) -> Heard:
        spoken = verbalize(said)
        if not re.search("[가-힣]", spoken):  # nothing pronounceable ("...", an emoji): nothing is heard
            return Heard(said, spoken, "", "")
        key = hashlib.sha256(f"{self._identity}|{seed}|{spoken}".encode()).hexdigest()
        entry = self._cache.get(key)
        cached = entry is not None
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
        heard = entry["heard"]
        text = self.normalizer(heard) if self.normalizer else heard
        return Heard(
            said, spoken, heard, text, entry["audio_seconds"], entry["tts_ms"], entry["stt_ms"], cached
        )
