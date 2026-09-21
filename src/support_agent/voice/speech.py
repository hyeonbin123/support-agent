"""Speech synthesis (MeloTTS) and recognition (faster-whisper). Needs the `voice` dependency group.

The libraries are imported inside the constructors, so importing this module needs none of them.
MeloTTS was picked for Korean in the sibling voice-translator project by re-recognition error (CER 5.5%,
human recordings 5.4%); the three Windows workarounds below come from there.
"""

from __future__ import annotations

import functools
import io
import os
import sys
import time
import types
import wave
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from typing import Any

SAMPLE_WIDTH = 2  # 16-bit PCM
LISTEN_RATE = 16_000  # what Whisper hears


class AudioTooLongError(ValueError):
    """The recording is longer than the listener accepts."""


def versions(*packages: str) -> dict[str, str]:
    """Installed versions: part of what a cached round trip depends on."""
    out = {}
    for package in packages:
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = "?"
    return out


@dataclass(frozen=True)
class Audio:
    wav: bytes  # 16-bit mono PCM WAV file
    seconds: float


def wav_bytes(samples: Any, sample_rate: int) -> Audio:
    import numpy as np

    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise RuntimeError("the speech synthesis model returned no audio")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(SAMPLE_WIDTH)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())
    return Audio(buffer.getvalue(), samples.size / sample_rate)


@functools.cache
def add_cuda_dll_dirs() -> None:
    """CTranslate2 loads cuBLAS and cuDNN by name; Windows does not search the pip wheels' folders."""
    if sys.platform != "win32":
        return
    for package in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = find_spec(package)
        except ModuleNotFoundError:
            spec = None
        for root in (spec.submodule_search_locations or []) if spec else []:
            bin_dir = os.path.join(root, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def _prepare_melo_on_windows() -> None:
    """melo always imports its Japanese front end, whose `MeCab` package is the same folder as
    python-mecab-ko's `mecab` on a case-insensitive file system; and g2pkk looks for `eunjeon` on Windows."""
    if sys.platform != "win32":
        return
    if "melo.text.japanese" not in sys.modules:
        stub = types.ModuleType("melo.text.japanese")

        def distribute_phone(n_phone: int, n_word: int) -> list[int]:
            phones_per_word = [0] * n_word
            for _ in range(n_phone):
                phones_per_word[phones_per_word.index(min(phones_per_word))] += 1
            return phones_per_word

        stub.distribute_phone = distribute_phone  # type: ignore[attr-defined]
        sys.modules["melo.text.japanese"] = stub
    import g2pkk.g2pkk
    import mecab

    g2pkk.g2pkk.G2p.check_mecab = lambda self: None
    g2pkk.g2pkk.G2p.get_mecab = lambda self: mecab.MeCab()


class MeloSpeaker:
    """Korean MeloTTS (MIT). Sampling noise is seeded, so the same text and seed give the same audio."""

    def __init__(self, device: str = "cuda", speed: float = 1.0):
        _prepare_melo_on_windows()
        import torch
        from melo.api import TTS

        self._torch = torch
        self.speed = speed
        self.device = device
        self._model = TTS(language="KR", device=device)
        self._speaker = self._model.hps.data.spk2id["KR"]
        self.sample_rate = self._model.hps.data.sampling_rate

    def describe(self) -> dict[str, Any]:
        return {
            "tts": "melotts/KR",
            "device": self.device,
            "speed": self.speed,
            "tts_versions": versions("melotts", "torch"),
        }

    def synthesize(self, spoken_text: str, seed: int = 0) -> Audio:
        self._torch.manual_seed(seed)
        samples = self._model.tts_to_file(spoken_text, self._speaker, None, speed=self.speed, quiet=True)
        return wav_bytes(samples, self.sample_rate)


class WhisperListener:
    """faster-whisper, Korean, temperature 0 only (no sampling fallback: the channel must be repeatable)."""

    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str | None = None,
        beam_size: int = 5,
        max_seconds: float | None = None,
        vad_filter: bool = False,
    ):
        add_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        self.model, self.device, self.beam_size = model, device, beam_size
        self.max_seconds = max_seconds  # None: no limit (the evaluation synthesises its own audio)
        # Off for the measurements (synthesised speech has no silence). On for a real microphone: given a
        # recording without speech, Whisper writes stock phrases of video subtitles instead of nothing.
        self.vad_filter = vad_filter
        self.compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self._model = WhisperModel(model, device=device, compute_type=self.compute_type)

    def describe(self) -> dict[str, Any]:
        return {
            "stt": f"faster-whisper/{self.model}",
            "device": self.device,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
            "stt_versions": versions("faster-whisper", "ctranslate2"),
        }

    def transcribe(self, audio: bytes) -> str:
        from faster_whisper.audio import decode_audio

        samples = decode_audio(io.BytesIO(audio), sampling_rate=LISTEN_RATE)
        if self.max_seconds is not None and len(samples) > self.max_seconds * LISTEN_RATE:
            raise AudioTooLongError(f"{len(samples) / LISTEN_RATE:.0f} s of audio")
        segments, _info = self._model.transcribe(
            samples,
            language="ko",
            beam_size=self.beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=self.vad_filter,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def timed(run, *args):
    started = time.perf_counter()
    result = run(*args)
    return result, (time.perf_counter() - started) * 1000
