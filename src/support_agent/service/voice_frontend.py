"""Speech in front of the chat: recognise what the customer said, read the agent's reply aloud.

Optional (SUPPORT_AGENT_VOICE=1, needs the `voice` dependency group). The chat itself does not change: a
recognised utterance takes the same path as a typed message.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from support_agent.voice.channel import Listener, Speaker
from support_agent.voice.normalize import normalize_heard
from support_agent.voice.spoken import spoken_to_written
from support_agent.voice.verbalize import verbalize


class VoiceFrontEnd:
    def __init__(self, speaker: Speaker, listener: Listener, normalizer: Callable[[str], str] | None = None):
        self.speaker, self.listener, self.normalizer = speaker, listener, normalizer
        self._lock = threading.Lock()  # one model call at a time: the models are not built for threads

    def describe(self) -> dict[str, Any]:
        return {**self.speaker.describe(), **self.listener.describe()}

    def listen(self, audio: bytes) -> tuple[str, str]:
        """(what the recogniser wrote, what the agent receives)."""
        with self._lock:
            heard = self.listener.transcribe(audio)
        return heard, (self.normalizer(heard) if self.normalizer else heard)

    def speak(self, text: str) -> bytes:
        """A WAV file of `text` read aloud; b"" when nothing in it can be pronounced."""
        spoken = verbalize(text)
        if not any("가" <= ch <= "힣" for ch in spoken):
            return b""
        with self._lock:
            return self.speaker.synthesize(spoken, 0).wav


def repair_heard(text: str) -> str:
    """What the agent receives instead of the raw transcript: the notation rules that the measurement
    adopted (V2, docs/experiments.md stage 5), then the numbers a person read aloud as words."""
    return spoken_to_written(normalize_heard(text))


def load_voice_front_end(tts_device: str, stt_device: str, max_audio_seconds: float) -> VoiceFrontEnd:
    from support_agent.voice.speech import MeloSpeaker, WhisperListener

    listener = WhisperListener(device=stt_device, max_seconds=max_audio_seconds, vad_filter=True)
    return VoiceFrontEnd(MeloSpeaker(device=tts_device), listener, normalizer=repair_heard)
