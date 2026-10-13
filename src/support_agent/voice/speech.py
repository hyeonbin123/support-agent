"""Speech synthesis (MeloTTS) and recognition (faster-whisper). Needs the `voice` dependency group.

The libraries are imported inside the constructors, so importing this module needs none of them.
MeloTTS was picked for Korean in the sibling voice-translator project by re-recognition error (CER 5.5%,
human recordings 5.4%); the three Windows workarounds below come from there.
"""

# [파일 역할] 음성 합성(MeloTTS 한국어)과 음성 인식(faster-whisper large-v3-turbo) 래퍼다. 평가의 음성
# 채널(5·9단계), 11단계 재합성(voice_real.py), 서비스의 음성 입출력(service/voice_frontend.py)이 모두 이 두
# 클래스를 쓴다.
# [설계 이유] 무거운 라이브러리(torch, melo, faster_whisper, numpy, av)는 생성자나 함수 안에서만 불러온다.
# 그래서 선택 의존성 그룹 voice가 없는 환경(CI 등)에서도 이 모듈을 불러올 수 있고, tests/test_voice.py가 맨
# 앞(0열)의 import 줄에 이 이름들이 없는지 검사한다.
# 같은 입력이면 같은 소리·같은 글이 나오게 했다: 합성은 발화마다 seed를 고정하고, 인식은 temperature 0만 쓴다.
# 그래서 왕복 결과를 캐시해도 측정값이 바뀌지 않는다 (docs/voice.md "정한 것과 이유").
# [주요 구성] 버전 기록(versions) → WAV 만들기(wav_bytes) → Windows 우회(add_cuda_dll_dirs,
# _prepare_melo_on_windows) → MeloSpeaker → 길이 제한 디코더(decode_limited) → WhisperListener → 시간
# 재기(timed).

from __future__ import annotations

# 표준 라이브러리만 맨 위에서 불러온다. functools.cache: DLL 경로 등록을 프로세스당 한 번만. io·wave: WAV
# 파일을 메모리에서 만들고 읽기. os·sys: Windows 판별과 DLL 경로. types: melo의 일본어 모듈 자리에 넣을 빈
# 모듈. importlib.metadata: 설치된 패키지 버전(캐시 키에 들어감). find_spec: 패키지를 실제로 불러오지 않고
# 설치 위치만 찾기.
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

# SAMPLE_WIDTH: 합성 결과를 담는 WAV의 표본 크기(2바이트 = 16비트 PCM). LISTEN_RATE: 인식기에 넣는 표본화율(16
# kHz). voice_real.py의 디코더와 서비스 쪽도 이 값을 쓴다.
SAMPLE_WIDTH = 2  # 16-bit PCM
LISTEN_RATE = 16_000  # what Whisper hears


# 녹음이 서비스의 길이 한도를 넘을 때 decode_limited가 낸다. service/app.py가 이 예외를 잡아 '녹음이 너무
# 깁니다' 안내로 바꾼다.
class AudioTooLongError(ValueError):
    """The recording is longer than the listener accepts."""


# 패키지 이름 → 설치된 버전. 설치되지 않은 것은 "?"로 적는다(예외를 내지 않는다).
# 왜 필요한가: describe()가 이 버전을 캐시 키와 실행 기록에 넣어, 패키지 버전이 바뀌면 예전 왕복 결과를 쓰지
# 않게 한다. run.py와 voice_real.py도 manifest의 versions 항목에 이 함수를 쓴다.
def versions(*packages: str) -> dict[str, str]:
    """Installed versions: part of what a cached round trip depends on."""
    out = {}
    for package in packages:
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = "?"
    return out


# 합성 결과: 16비트 모노 WAV 파일 바이트와 길이(초). 채널은 이 바이트를 녹음 파일처럼 인식기에 넘긴다.
@dataclass(frozen=True)
class Audio:
    wav: bytes  # 16-bit mono PCM WAV file
    seconds: float


# float 파형을 16비트 모노 PCM WAV 파일 바이트로 만든다. 디스크에 쓰지 않고 메모리(BytesIO)에서 만든다.
# 파형이 비어 있으면 합성 실패로 보고 RuntimeError를 낸다(빈 소리를 인식기에 넘겨 '인식 안 됨'으로 섞이지
# 않게).
def wav_bytes(samples: Any, sample_rate: int) -> Audio:
    import numpy as np

    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise RuntimeError("the speech synthesis model returned no audio")
    # -1~1로 자르고 32767을 곱해 리틀엔디언 int16("<i2")으로 바꾼다. 자르지 않으면 범위를 넘는 표본이 int16
    # 범위를 벗어나 정수 변환 결과가 망가진다.
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(SAMPLE_WIDTH)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())
    return Audio(buffer.getvalue(), samples.size / sample_rate)


# Windows에서 CTranslate2(faster-whisper의 엔진)가 cuBLAS·cuDNN DLL을 찾게 해 준다. 이 DLL은 pip의 nvidia-* 휠
# 폴더에 있는데, Windows는 그 폴더를 찾지 않는다.
# 각 패키지의 bin 폴더를 os.add_dll_directory와 PATH 양쪽에 더한다. functools.cache로 한 프로세스에서 한 번만
# 한다. 호출 위치: WhisperListener 생성자.
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
        # nvidia.cublas 같은 이름공간 패키지는 설치 위치가 여러 곳일 수 있어 모두 돈다. 패키지가 없으면
        # 아무것도 하지 않는다(CPU 환경).
        for root in (spec.submodule_search_locations or []) if spec else []:
            bin_dir = os.path.join(root, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


# Windows에서 MeloTTS를 불러오기 전에 필요한 우회 세 가지 (voice-translator 프로젝트에서 가져옴, docs/voice.md
# "정한 것과 이유").
# 1) melo는 일본어 앞단(melo.text.japanese)을 항상 불러오는데, 그 의존성 MeCab 폴더가 대소문자를 가리지 않는
#    파일 시스템에서 python-mecab-ko의 mecab 폴더와 같은 폴더가 된다. 그래서 일본어 모듈 자리에 빈 모듈을 넣고
#    melo가 거기서 쓰는 distribute_phone만 채운다.
# 2)·3) 한국어 발음 변환기 g2pkk가 Windows에서 eunjeon 패키지를 찾으므로, mecab 확인을 건너뛰고
# python-mecab-ko의 MeCab을 돌려주게 바꾼다.
# 호출 위치: MeloSpeaker 생성자 (melo를 불러오기 전).
def _prepare_melo_on_windows() -> None:
    """melo always imports its Japanese front end, whose `MeCab` package is the same folder as
    python-mecab-ko's `mecab` on a case-insensitive file system; and g2pkk looks for `eunjeon` on Windows."""
    if sys.platform != "win32":
        return
    if "melo.text.japanese" not in sys.modules:
        stub = types.ModuleType("melo.text.japanese")

        # 음소 n_phone개를 낱말 n_word개에 고르게 나눈다: 음소 하나씩, 지금 가장 적게 받은 낱말에 준다. melo가
        # 일본어 모듈에서 가져다 쓰는 함수의 대역이다.
        def distribute_phone(n_phone: int, n_word: int) -> list[int]:
            phones_per_word = [0] * n_word
            for _ in range(n_phone):
                phones_per_word[phones_per_word.index(min(phones_per_word))] += 1
            return phones_per_word

        stub.distribute_phone = distribute_phone  # type: ignore[attr-defined]
        sys.modules["melo.text.japanese"] = stub
    # g2pkk의 G2p 클래스 메서드 두 개를 바꾼다: mecab 설치 확인은 하지 않고, 형태소 분석기는 python-mecab-ko의
    # MeCab()을 쓴다.
    import g2pkk.g2pkk
    import mecab

    g2pkk.g2pkk.G2p.check_mecab = lambda self: None
    g2pkk.g2pkk.G2p.get_mecab = lambda self: mecab.MeCab()


# 한국어 MeloTTS 합성기. 한국어 합성 모델로 고른 근거는 옆 프로젝트에서 잰 재인식 글자 오류율(5.5%, 사람 녹음
# 5.4%)이다.
# 호출 위치: run.py(평가 음성 채널), voice_real.py(11단계 재합성), service/voice_frontend.py(답을 소리로).
class MeloSpeaker:
    """Korean MeloTTS (MIT). Sampling noise is seeded, so the same text and seed give the same audio."""

    def __init__(self, device: str = "cuda", speed: float = 1.0):
        # Windows 우회를 먼저 하고 나서 torch와 melo를 불러온다. 라이브러리는 여기서만 불러오므로 모듈 자체는
        # 음성 그룹 없이도 import된다.
        _prepare_melo_on_windows()
        import torch
        from melo.api import TTS

        self._torch = torch
        self.speed = speed
        self.device = device
        self._model = TTS(language="KR", device=device)
        # 화자는 한국어 모델의 KR 화자 하나로 고정하고, 출력 표본화율은 모델 설정에서 읽는다(WAV 머리에 그대로
        # 적는다).
        self._speaker = self._model.hps.data.spk2id["KR"]
        self.sample_rate = self._model.hps.data.sampling_rate

    # 합성 설정(모델, 장치, 속도, 패키지 버전). 음성 채널(channel.py)과 voice_real.py가 이 값을 왕복 캐시 키에
    # 넣고, voice_real.channel_mismatches는 9단계 기록과 같은 합성기인지 이 값으로 확인한다.
    def describe(self) -> dict[str, Any]:
        return {
            "tts": "melotts/KR",
            "device": self.device,
            "speed": self.speed,
            "tts_versions": versions("melotts", "torch"),
        }

    # 말로 풀어 쓴 글(spoken_text)을 WAV로 합성한다. 부를 때마다 torch.manual_seed(seed)로 표본 추출 잡음을
    # 고정하므로 같은 글과 seed면 같은 소리가 나온다. 평가 채널은 seed를 derive_seed(기본 seed, 과제, 시도,
    # "voice", 순번)로 만든다.
    # 출력 경로 자리에 None을 주어 파일 대신 파형 배열을 받고, wav_bytes로 WAV 바이트를 만든다.
    def synthesize(self, spoken_text: str, seed: int = 0) -> Audio:
        self._torch.manual_seed(seed)
        samples = self._model.tts_to_file(spoken_text, self._speaker, None, speed=self.speed, quiet=True)
        return wav_bytes(samples, self.sample_rate)


# 서비스용 길이 제한 디코더. 브라우저가 보낸 녹음을 faster-whisper의 decode_audio와 같은 결과(16 kHz 모노
# float32)로 풀되, 한도를 넘는 순간 멈춘다 (같은 출력인지는 테스트가 비교한다).
# 왜 필요한가: 몇 kB짜리 파일에 몇 시간의 무음이 들 수 있고, 샘플레이트가 아주 낮으면 프레임 하나가 몇 시간이
# 된다. 다 푼 뒤나 16 kHz로 바꾼 뒤에 재면 그 사이 메모리를 수백 MB~GB 쓴다 (docs/service.md "한도"). 서비스의
# 한도는 풀었을 때 60초다.
# 호출 위치: WhisperListener.transcribe (max_seconds가 있을 때, 즉 서비스).
def decode_limited(audio: bytes, max_seconds: float) -> Any:
    """faster-whisper's decode_audio, but it stops as soon as the recording passes `max_seconds`: a few kB of
    FLAC can hold hours of silence, and decode_audio would first expand all of it in memory.

    Each frame's length is counted before the resampler sees it, because one frame can be hours long: 65,535
    samples at 5 Hz become 13,107 s at 16 kHz, gigabytes inside the resampler. The count of what comes out
    stays the exact check; a second of slack (far more than the resampler's delay) keeps the early count from
    refusing a recording that the exact one accepts."""
    import gc

    import av
    import numpy as np

    # limit은 16 kHz로 바꾼 뒤의 표본 수 한도이고(정확한 검사), seconds_in은 리샘플러에 넣기 전 원본 프레임
    # 길이의 합이다(이른 검사). 리샘플러는 decode_audio와 같은 설정(s16, 모노, 16 kHz)이다.
    limit, total, chunks = int(max_seconds * LISTEN_RATE), 0, []
    seconds_in = 0.0
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=LISTEN_RATE)

    # 리샘플러에서 나온 프레임을 모으면서 표본 수를 세고, 한도를 넘으면 바로 예외를 낸다. 나머지를 풀지 않고
    # 멈추므로 메모리가 늘지 않는다.
    def take(frames) -> None:
        nonlocal total
        for frame in frames:
            array = frame.to_ndarray().reshape(-1)
            total += array.size
            if total > limit:
                raise AudioTooLongError(f"more than {max_seconds:g} s of audio")
            chunks.append(array)

    try:
        # PyAV로 메모리의 녹음을 연다. 메타데이터 오류는 무시하고 첫 오디오 스트림만 푼다.
        with av.open(io.BytesIO(audio), mode="r", metadata_errors="ignore") as container:
            frames = container.decode(audio=0)
            while True:  # like decode_audio, a frame that fails to decode is skipped
                try:
                    frame = next(frames)
                except StopIteration:
                    break
                except av.error.InvalidDataError:
                    continue
                # 프레임을 리샘플러에 넣기 전에 길이를 먼저 센다. 프레임 하나가 몇 시간짜리여도 리샘플러
                # 안에서 GB 단위로 커지기 전에 거절한다. 1초의 여유는 리샘플러 지연을 덮어, 정확한 검사가
                # 받아들일 녹음을 이른 검사가 거절하지 않게 한다.
                seconds_in += frame.samples / frame.sample_rate
                if seconds_in > max_seconds + 1:
                    raise AudioTooLongError(f"more than {max_seconds:g} s of audio")
                take(resampler.resample(frame))
            # 리샘플러 안에 남은 표본을 내보낸다(None을 넣으면 비운다). 이것도 한도에 센다.
            take(resampler.resample(None))  # flush
    finally:
        # 예외로 끝나도 리샘플러를 지우고 가비지 컬렉션을 돌린다. 리샘플러 객체는 컬렉터가 돌아야 풀린다
        # (faster-whisper 이슈 390).
        del resampler
        gc.collect()  # faster-whisper#390: resampler objects are freed only by the collector
    # int16 표본을 32768로 나눠 -1~1 float32로 바꾼다. decode_audio와 같은 변환이다. 아무것도 없으면 길이 0
    # 배열.
    return np.concatenate(chunks).astype(np.float32) / 32768.0 if chunks else np.zeros(0, np.float32)


# faster-whisper 인식기. 측정 설정은 large-v3-turbo, GPU float16, beam 5, 언어 ko, temperature 0만, 이전 문장
# 조건 없음, VAD 끔이다 (docs/experiments.md 5단계 "조건").
# 호출 위치: run.py(평가 채널, 기본값 그대로), voice_real.py(같은 설정, transcribe_samples),
# service/voice_frontend.py(max_seconds와 vad_filter=True를 준 서비스용).
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
        # Windows에서 CUDA DLL 경로를 먼저 등록한 뒤 faster_whisper를 불러온다.
        add_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        self.model, self.device, self.beam_size = model, device, beam_size
        self.max_seconds = max_seconds  # None: no limit (the evaluation synthesises its own audio)
        # Off for the measurements (synthesised speech has no silence). On for a real microphone: given a
        # recording without speech, Whisper writes stock phrases of video subtitles instead of nothing.
        self.vad_filter = vad_filter
        # 연산 정밀도를 주지 않으면 GPU는 float16, CPU는 int8을 쓴다. 측정은 GPU float16이고, voice_real.py는
        # compute_type이 기록과 다르면 공식 실행을 거절한다.
        self.compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self._model = WhisperModel(model, device=device, compute_type=self.compute_type)

    # 인식 설정(모델, 장치, 정밀도, beam, VAD, 패키지 버전). 왕복 캐시 키와 실행 기록에 들어가고,
    # voice_real.channel_mismatches가 9단계 기록과 비교하는 값이다.
    def describe(self) -> dict[str, Any]:
        return {
            "stt": f"faster-whisper/{self.model}",
            "device": self.device,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
            "stt_versions": versions("faster-whisper", "ctranslate2"),
        }

    # 녹음 바이트를 글로 받아 적는다. 길이 한도가 없으면(평가 채널) faster-whisper의 decode_audio를 그대로
    # 써서 잰 채널을 바꾸지 않고, 한도가 있으면(서비스) decode_limited로 푼다.
    def transcribe(self, audio: bytes) -> str:
        if self.max_seconds is None:  # the measured channel: unchanged
            from faster_whisper.audio import decode_audio

            samples = decode_audio(io.BytesIO(audio), sampling_rate=LISTEN_RATE)
        else:
            samples = decode_limited(audio, self.max_seconds)
        return self.transcribe_samples(samples)

    # 이미 16 kHz로 푼 표본을 받아 적는다. 11단계는 디코딩과 인식 사이에 열화를 넣어야 해서 voice_real.py가 이
    # 메서드를 직접 부른다.
    def transcribe_samples(self, samples: Any) -> str:
        """Decoded 16 kHz mono float32 samples to text (stage 11 degrades the samples in between)."""
        # temperature=0.0 하나만 주어 실패 시 온도를 올려 다시 하는 기능을 끈다(표본 추출이 섞이면 같은 소리가
        # 다른 글이 될 수 있다). condition_on_previous_text=False로 발화 안의 앞 구간 글이 뒤 구간 인식에
        # 끼어들지 않게 한다. vad_filter는 말 없는 구간을 거른다: 서비스에서 말없이 버튼만 눌렀다 뗀 녹음을
        # Whisper가 '다음 영상에서 만나요.'로 적은 일이 있어 서비스만 켠다 (docs/voice.md).
        segments, _info = self._model.transcribe(
            samples,
            language="ko",
            beam_size=self.beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=self.vad_filter,
        )
        # 구간(segment)별 글을 공백으로 이어 한 문장으로 돌려준다. 인식된 것이 없으면 빈 문자열이다.
        return " ".join(segment.text.strip() for segment in segments).strip()


# 함수를 실행하고 (결과, 걸린 시간 ms)를 돌려준다. channel.py와 voice_real.py가 합성·인식 시간을 기록할 때
# 쓴다.
def timed(run, *args):
    started = time.perf_counter()
    result = run(*args)
    return result, (time.perf_counter() - started) * 1000
