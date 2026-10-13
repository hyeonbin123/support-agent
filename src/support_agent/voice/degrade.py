"""Telephone-call conditions for synthesised speech: stage 11 (V4-R), fixed before measuring.

The values are tau-Voice's "Realistic" condition (arXiv 2603.13686, Table 4 and Table 13; the defaults of
tau2-bench's voice_config.py for the parts the paper leaves out: the muffled segment and its cutoff). Only
the values are taken; this is our own implementation. The order is tau-Voice's: speech effect (muffling),
source effects (background noise, bursts), channel effects (telephone band and mu-law, lost frames).

Differences from tau-Voice, chosen before measuring (docs/experiments.md, stage 11):
- Noise is generated (pink or brown), and bursts are four generated sounds. tau-Voice mixes recordings
  (street, people talking; phone rings, dog barks). Babble made by our own Korean voice is left out: the
  recogniser could write down its words, which is another disturbance than noise.
- Every SNR is measured inside the telephone band (300-3400 Hz) against the utterance's own speech, not
  against a fixed reference level: the channel removes what lies outside the band, so a noise colour with its
  energy below 300 Hz does not make the condition easier.
- The background level is drawn once per utterance (15 dB +-3, uniform) instead of drifting within it.
- Out-of-turn speech and vocal tics are left out (they need our own voice again).
- The 8 kHz mu-law step is whisper-ko-ft's channel.py `telephone()` (its stage 4), split in two so that lost
  frames fall between the codec and the way back to 16 kHz. Lost frames are silence (no concealment).

Input and output are 16 kHz mono float32, what faster-whisper decodes a recording to.
"""

# [파일 역할] 11단계(V4-R A단계)에서 합성한 고객 음성에 '전화 통화 조건'을 얹는 열화 모듈이다. 16 kHz 음성을
# 받아 먹먹함 → 배경 소음 → 돌발 소음 → 음량 맞춤 → 전화 대역·8 kHz·mu-law → 프레임 손실 → 16 kHz 복원 순서로
# 바꾼다.
# [왜 필요한가] 5·9단계의 음성 수치는 모두 깨끗한 합성 음성에서 나와 실제 통화보다 쉬운 조건이다. 9단계 V4의
# 본인 확인은 이름 받아쓰기에 기대므로, 전화 음질에서 이름이 얼마나 살아남는지를 재려고 τ-Voice의 공개 열화
# 구성(Realistic)의 값만 가져와 직접 구현했다 (docs/experiments.md 11단계 "왜 하는가").
# [설계 이유] 값은 재기 전에 정해 커밋했고 개발용 결과를 보고 고치지 않는다. 무작위는 발화마다 정해진 seed에서
# 나와, 같은 seed와 길이면 같은 조건이 다시 나온다. 그래서 왕복 결과를 캐시해도 측정값이 바뀌지 않는다.
# [주요 구성] 매개변수(Realistic) → 전화 채널(mu_law_roundtrip, narrowband, widen) → 발화 하나의 무작위
# 뽑기(draw, Plan) → 소리 만들기(coloured_noise, burst_sound, _muffle) → 합치기(mix, apply, degrade).
# [연결] voice_real.py가 degrade()를 측정에, draw()를 CPU 관문(plan_gate)에, describe()를 manifest와 캐시 키에
# 쓴다. numpy와 scipy만 쓰므로 음성 모델 없이 테스트(tests/test_stage11.py)가 돈다.

from __future__ import annotations

# dataclass: 매개변수 묶음(Realistic)과 발화별 뽑기 결과(Plan)를 바꿀 수 없는 값 객체로 둔다. asdict는
# 매개변수를 기록용 dict로 바꿀 때 쓴다.
from dataclasses import asdict, dataclass

# numpy: 파형 배열 연산과 seed를 고정한 난수 생성기(Generator, SeedSequence).
# scipy.signal: 버터워스 필터 설계(butter), 위상 지연 없는 필터링(sosfiltfilt), 정수비
# 리샘플링(resample_poly).
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

# 열화 구현의 버전 문자열. describe()를 거쳐 왕복 캐시 키와 실행 기록(manifest.json의 degrade)에 들어가므로,
# 아래 무엇이든 바꾸면 이 값도 바꿔야 예전 캐시 결과가 섞이지 않는다.
DEGRADE_VERSION = "v4r-1"  # part of the cache key and the manifest: change it with any change below
# SAMPLE_RATE: 입출력 표본화율(16 kHz). faster-whisper가 녹음을 풀어 인식기에 넣는 표본화율과 같다.
# NARROW_RATE: 전화 채널 구간의 표본화율(8 kHz). 프레임 손실은 이 신호 위에서 계산한다.
SAMPLE_RATE = 16_000
NARROW_RATE = 8_000
# mu-law 압축 강도. G.711 mu-law의 값 255이다. 작은 소리를 촘촘히 양자화해 8비트로도 말소리가 남게 하는
# 방식이다.
MU = 255.0
# 전화 음성 대역(300~3400 Hz)만 통과시키는 4차 버터워스 대역 통과 필터. 모듈을 불러올 때 한 번 설계해
# 재사용한다. 전화 채널뿐 아니라 SNR을 잴 때(_band_rms)도 같은 필터를 쓴다.
BAND = butter(4, [300, 3400], btype="bandpass", fs=SAMPLE_RATE, output="sos")
# sosfiltfilt는 앞뒤로 신호를 덧대어 거르는데, 이보다 짧은 입력은 덧댈 길이가 모자라 오류가 난다. 그런 발화는
# 열화 없이 그대로 돌려준다(telephone, apply).
MIN_SAMPLES = 64  # shorter than the zero-phase filter's padding: left as it is
# 돌발 소음 네 종류의 길이(초). τ-Voice는 녹음(전화벨, 개 짖는 소리 등)을 섞지만, 이 저장소에는 쓸 녹음이 없고
# 내려받지 않기로 해서 소리를 직접 생성한다 (docs/experiments.md 11단계 "다르게 정한 것").
# Generated bursts (seconds): a clatter (decaying broadband noise), a ringing phone (two tones alternating
# every 25 ms), a car horn (420 Hz and its harmonics), a siren (650 -> 1300 -> 650 Hz).
BURST_SECONDS = {"clatter": 0.25, "ring": 1.0, "horn": 0.6, "siren": 1.5}
# 돌발 소음의 시작과 끝을 10 ms에 걸쳐 키우고 줄인다. 갑자기 켜지고 꺼지는 소리가 딸깍 잡음을 만들지 않게
# 하려는 것이다.
_FADE_SECONDS = 0.01
# 생성 잡음은 20 Hz 아래 성분을 0으로 둔다. 1/f 모양을 그대로 두면 0 Hz 근처에서 값이 끝없이 커지기 때문이다.
_LOWEST_NOISE_HZ = 20.0


# τ-Voice Realistic 조건의 매개변수 묶음 (논문 표 4·표 13, 논문에 없는 먹먹함의 길이와 차단 주파수는
# tau2-bench voice_config.py의 기본값). frozen이라 실행 중에 값이 바뀌지 않고, describe()가 이 값을 그대로
# 기록한다.
@dataclass(frozen=True)
class Realistic:
    # 배경 소음: SNR 15 dB ±3 dB를 발화마다 한 번 고르게 뽑는다. τ-Voice는 한 통화 안에서 천천히 흔들지만,
    # 발화가 평균 5.7초라 차이가 작아 발화 단위로 정했다. 색은 분홍·갈색 중 하나.
    noise_snr_db: float = 15.0
    noise_snr_spread_db: float = 3.0  # uniform in [15 - 3, 15 + 3] per utterance
    noise_colours: tuple[str, ...] = ("pink", "brown")
    # 돌발 소음: 분당 평균 1회(포아송, 발화 길이에 비례). 하나마다 SNR을 −5 ~ +10 dB에서 고르게 뽑고 네 소리
    # 중 하나를 고른다.
    burst_per_minute: float = 1.0  # Poisson
    burst_snr_db: tuple[float, float] = (-5.0, 10.0)  # uniform per burst
    burst_kinds: tuple[str, ...] = ("clatter", "ring", "horn", "siren")
    # 먹먹함: 발화의 20%에서 한 구간 500 ms를 1500 Hz 저역 통과로 바꾸고, 앞뒤 100 ms에 걸쳐 섞어 들어가고
    # 나온다. 송화기를 손으로 가리는 것 같은 말소리 효과다.
    muffle_probability: float = 0.2  # of utterances
    muffle_ms: int = 500  # one low-passed segment ...
    muffle_transition_ms: int = 100  # ... faded in and out over this
    muffle_cutoff_hz: float = 1500.0
    # 프레임 손실: 20 ms 프레임(G.711 패킷 하나)에 Gilbert–Elliott 모형을 적용한다. 평균 손실 2%, 잃는 구간의
    # 평균 길이 100 ms(= 5프레임). 잃은 프레임은 메우지 않고 무음으로 둔다.
    frame_ms: int = 20  # one G.711 packet
    loss_rate: float = 0.02  # Gilbert-Elliott: share of lost frames ...
    loss_burst_ms: float = 100.0  # ... and the mean length of a run of them
    # 섞은 소리의 최댓값이 0.99를 넘으면 발화 전체 음량을 낮춘다. mu-law 단계가 -1~1 밖을 잘라 내므로, 잘림
    # 왜곡이라는 다른 열화가 섞이지 않게 미리 줄이는 것이다.
    peak: float = 0.99  # a louder mixture is turned down before the codec instead of clipped


# 등록한 기본 조건. 함수들의 기본 인자로 쓰고, voice_real.py의 CPU 관문도 이 값의 비율로 뽑힌 조건을 검사한다.
REALISTIC = Realistic()


# 매개변수와 버전을 JSON으로 쓸 수 있는 dict로 돌려준다 (튜플은 리스트로 바꾼다).
# 호출 위치: voice_real.py _run()이 manifest.json의 degrade 항목과 measure()의 캐시 키(description)에 넣는다.
# 값이 하나라도 바뀌면 캐시 키가 달라져 예전 왕복 결과를 다시 쓰지 않는다.
def describe(params: Realistic = REALISTIC) -> dict:
    """What the cache key and the manifest record."""
    values = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(params).items()}
    return {"degrade": DEGRADE_VERSION, **values}


# ---------------------------------------------------------------- the telephone (whisper-ko-ft channel.py)


# 8비트 mu-law로 압축 → 256단계 양자화 → 복원을 한 번 거친 파형을 돌려준다. whisper-ko-ft channel.py의 같은
# 이름 함수와 같은 구현이고(G.711의 구간 표가 아니라 연속 mu-law를 256단계로 양자화), 그 저장소 코드로 계산한
# 출력값을 테스트에 고정해 두었다 (docs/experiments.md 11단계 표 5번).
def mu_law_roundtrip(samples: np.ndarray) -> np.ndarray:
    """Compress to 8-bit mu-law codes and expand back."""
    # mu-law 공식은 -1~1 범위를 가정하므로 먼저 잘라 낸다. 잘림 자체가 왜곡이 되므로 mix()는 섞은 소리의
    # 최댓값을 미리 0.99 아래로 낮춘다.
    clipped = np.clip(samples, -1.0, 1.0)
    compressed = np.sign(clipped) * np.log1p(MU * np.abs(clipped)) / np.log1p(MU)
    # -1~1을 0~255 정수 코드로 반올림한다. 여기서 양자화 잡음(정보 손실)이 생기고, 다음 두 줄이 압축의
    # 역함수로 원래 진폭 척도로 되돌린다.
    codes = np.round((compressed + 1.0) / 2.0 * 255.0)  # 256 levels
    restored = codes / 255.0 * 2.0 - 1.0
    return (np.sign(restored) * np.expm1(np.abs(restored) * np.log1p(MU)) / MU).astype(np.float32)


# 전화 채널의 앞 절반: 대역 통과(위상 지연 없는 sosfiltfilt) → 8 kHz로 다운샘플 → mu-law 왕복.
# 왜 telephone()을 둘로 나눴나: 프레임 손실을 코덱 뒤, 16 kHz 복원 전(8 kHz 신호 위)에 넣으려고 narrowband와
# widen으로 갈랐다. 둘을 이어 붙이면 telephone()과 출력이 같다는 것을 테스트가 확인한다.
def narrowband(samples: np.ndarray) -> np.ndarray:
    """16 kHz audio through the 300-3400 Hz band, at 8 kHz, after the mu-law codec."""
    return mu_law_roundtrip(resample_poly(sosfiltfilt(BAND, samples), 1, 2))


# 전화 채널의 뒤 절반: 8 kHz 신호를 16 kHz로 되돌리고 원래 길이로 자른다. 리샘플 과정에서 길이가 한두 표본
# 달라질 수 있어 자른다. 4 kHz 위의 고역은 되살아나지 않으므로 전화 음질은 그대로 남는다.
def widen(narrow: np.ndarray, length: int) -> np.ndarray:
    """The 8 kHz signal back at 16 kHz, cut to the original length."""
    return resample_poly(narrow, 2, 1)[:length].astype(np.float32)


# 소음·손실 없이 전화 채널만 거친 소리. 테스트가 whisper-ko-ft의 telephone()과 같은 값을 내는지, narrowband와
# widen을 이은 것과 같은지를 이 함수로 확인한다. 측정 경로(apply)는 이 함수 대신 narrowband와 widen을 직접
# 부른다.
def telephone(samples: np.ndarray) -> np.ndarray:
    """16 kHz audio as it would sound after a narrowband telephone channel, still at 16 kHz."""
    if len(samples) < MIN_SAMPLES:
        return samples.astype(np.float32)
    return widen(narrowband(samples), len(samples))


# ---------------------------------------------------------------- what is drawn for one utterance


# 16 kHz 표본 length개를 8 kHz로 줄였을 때의 20 ms 프레임 수(마지막 프레임은 짧을 수 있다).
# (length + 1) // 2는 resample_poly(…, 1, 2)가 내는 8 kHz 표본 수이고, -(-a // b)는 올림 나눗셈이다. 8 kHz에서
# 20 ms는 160표본이다.
def frame_count(length: int) -> int:
    """20 ms frames of the 8 kHz signal of `length` samples at 16 kHz (the last one may be short)."""
    narrow = (length + 1) // 2
    per_frame = NARROW_RATE * REALISTIC.frame_ms // 1000
    return -(-narrow // per_frame)


# Gilbert–Elliott 2상태 마르코프 연쇄로 잃을 프레임을 뽑는다. 나쁜 상태의 프레임은 모두 잃고 좋은 상태의
# 프레임은 하나도 잃지 않는 단순형이다.
# 공식: 나쁜 상태를 떠날 확률 leave = 1/burst_frames로 두면 잃는 구간의 평균 길이가 burst_frames(5프레임 = 100
# ms)가 된다. 들어갈 확률 enter = rate·leave/(1−rate)로 두면 정상 상태에서 나쁜 상태의 비율
# enter/(enter+leave)가 정확히 rate(2%)가 된다.
# 난수를 frames + 1개 한 번에 뽑아 두므로, 같은 rng 상태면 같은 손실 위치가 나온다.
def gilbert_elliott(frames: int, rate: float, burst_frames: float, rng: np.random.Generator) -> np.ndarray:
    """Lost frames of a two-state chain: every frame of the bad state is lost, none of the good state. Leaving
    the bad state has probability 1/burst_frames (mean run), entering it is set so that `rate` are lost."""
    leave = 1.0 / burst_frames
    enter = rate * leave / (1.0 - rate)
    draws = rng.random(frames + 1)
    lost = np.zeros(frames, dtype=bool)
    # 첫 프레임의 상태는 정상 상태 비율(rate)로 정한다. 발화가 연쇄의 중간에서 시작한 것처럼 다뤄, 발화
    # 앞부분의 손실률이 낮게 치우치지 않게 한다.
    bad = draws[0] < rate  # the first frame from the chain's stationary share
    for i in range(frames):
        lost[i] = bad
        # 다음 프레임의 상태: 지금 나쁘면 1 − leave 확률로 계속 나쁘고, 좋으면 enter 확률로 나빠진다.
        bad = draws[i + 1] < (1.0 - leave if bad else enter)
    return lost


# 발화 하나에 대해 뽑힌 열화 조건 전부. 소리 없이 길이와 seed만으로 만들 수 있어서, CPU
# 관문(voice_real.plan_gate)은 이것만 뽑아 등록한 비율과 비교하고, 실제 측정(apply)은 같은 Plan으로 소리를
# 바꾼다. 위치는 16 kHz 표본 번호로 들고 있다.
@dataclass(frozen=True)
class Plan:
    muffle_start: int | None  # first sample of the muffled span (ramp, segment, ramp); None: not muffled
    colour: str
    snr_db: float
    noise_seed: int
    bursts: tuple[tuple[str, int, float, int], ...]  # (kind, first sample, SNR dB, seed of its sound)
    frames: int
    lost: tuple[int, ...]  # indices of lost 20 ms frames

    # 기록용 dict. 표본 번호를 초 단위로 바꿔 사람이 읽기 쉽게 하고, degrade()가 gain을 더해
    # utterances.jsonl의 plan 항목으로 남긴다. voice_real.report()는 이 키(muffle_start_s, lost, bursts,
    # colour, snr_db)로 실제로 들어간 조건과 조건별 글자 오류율을 센다.
    def to_dict(self) -> dict:
        return {
            "muffle_start_s": None if self.muffle_start is None else self.muffle_start / SAMPLE_RATE,
            "colour": self.colour,
            "snr_db": self.snr_db,
            "noise_seed": self.noise_seed,
            "bursts": [
                {"kind": kind, "start_s": start / SAMPLE_RATE, "snr_db": snr, "seed": seed}
                for kind, start, snr, seed in self.bursts
            ],
            "frames": self.frames,
            "lost": list(self.lost),
        }


# seed와 길이만으로 발화 하나의 무작위 조건을 모두 뽑는다.
# 왜 효과마다 난수 흐름을 나누나: SeedSequence.spawn(4)로 먹먹함·소음·돌발·손실의 생성기를 따로 만들면, 한
# 효과의 뽑기 개수가 달라져도(예: 돌발 소음 수) 다른 효과의 뽑기가 밀리지 않는다 (docs/experiments.md 11단계
# "무작위는 … 효과마다 따로 갈라 쓴다").
# seed는 voice_real.py가 derive_seed(기본 seed, 과제, 시도, "degrade", 순번)로 만든다. 관문은 기록된 길이로,
# 측정은 다시 합성한 소리의 길이로 뽑으므로 둘의 조건이 조금 다를 수 있다.
def draw(seed: int, length: int, params: Realistic = REALISTIC) -> Plan:
    """Everything random about one utterance of `length` samples. Each effect has its own stream, so the
    draws of one do not move when another changes."""
    muffle_rng, noise_rng, burst_rng, loss_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
    )
    muffle_start = None
    # 먹먹함: 20% 확률로 켜고, 들어가는 구간(100 ms) + 유지(500 ms) + 나오는 구간(100 ms) = 700 ms가 발화 안에
    # 들어가도록 시작 위치를 고르게 뽑는다. 발화가 그보다 짧으면 0에서 시작하고 _muffle이 넘치는 부분을
    # 자른다.
    if muffle_rng.random() < params.muffle_probability:
        span = SAMPLE_RATE * (params.muffle_ms + 2 * params.muffle_transition_ms) // 1000
        muffle_start = int(muffle_rng.integers(0, max(length - span, 0) + 1))
    # 배경 소음: 색을 반반으로 고르고, SNR을 [12, 18] dB에서 고르게 뽑는다. 잡음 파형 자체는 여기서 뽑은
    # noise_seed로 mix()에서 만든다 (Plan에는 파형이 아니라 seed만 남긴다).
    colour = params.noise_colours[int(noise_rng.integers(len(params.noise_colours)))]
    low, high = (
        params.noise_snr_db - params.noise_snr_spread_db,
        params.noise_snr_db + params.noise_snr_spread_db,
    )
    snr_db = float(noise_rng.uniform(low, high))
    noise_seed = int(noise_rng.integers(2**31))
    # 돌발 소음: 개수는 포아송(기댓값 = 발화 길이(분) × 분당 1회), 하나마다 종류·시작 위치·SNR·소리 seed를
    # 뽑는다. 시작 위치가 발화 끝 근처면 mix()에서 넘치는 부분이 잘린다.
    bursts = []
    for _ in range(int(burst_rng.poisson(length / SAMPLE_RATE / 60.0 * params.burst_per_minute))):
        kind = params.burst_kinds[int(burst_rng.integers(len(params.burst_kinds)))]
        start = int(burst_rng.integers(0, max(length, 1)))
        snr = float(burst_rng.uniform(*params.burst_snr_db))
        bursts.append((kind, start, snr, int(burst_rng.integers(2**31))))
    # 프레임 손실: 100 ms / 20 ms = 5프레임을 평균 묶음 길이로 넘기고, 잃은 프레임의 번호만 튜플로 남긴다.
    frames = frame_count(length)
    lost = gilbert_elliott(frames, params.loss_rate, params.loss_burst_ms / params.frame_ms, loss_rng)
    return Plan(
        muffle_start, colour, snr_db, noise_seed, tuple(bursts), frames, tuple(np.flatnonzero(lost).tolist())
    )


# ---------------------------------------------------------------- sounds


# 전화 대역(300~3400 Hz) 안의 RMS. SNR을 이 값으로 재는 이유: 채널이 대역 밖을 걸러 내므로, 힘이 대부분 300 Hz
# 아래에 있는 갈색 잡음을 전 대역으로 재면 실제로 들리는 소음이 훨씬 약해져 갈색이 분홍보다 쉬운 조건이 된다
# (모듈 docstring, docs/experiments.md 11단계).
def _band_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(sosfiltfilt(BAND, x) ** 2)))


# 색 잡음 생성: 백색 가우스 잡음을 주파수 영역으로 옮겨 진폭에 f^(−지수/2)를 곱한다. 진폭이 f^(−1/2)이면 힘이
# 1/f(분홍), f^(−1)이면 1/f²(갈색)이다. 20 Hz 아래는 0으로 둔다. 마지막에 RMS 1로 맞춰 두고, 크기는 mix()가
# SNR에 맞게 정한다. "white"는 표에만 있고 등록한 조건(분홍·갈색)에서는 쓰지 않는다.
def coloured_noise(length: int, colour: str, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise with power falling as 1/f (pink) or 1/f^2 (brown) above 20 Hz, none below; unit RMS."""
    exponent = {"white": 0.0, "pink": 1.0, "brown": 2.0}[colour]
    spectrum = np.fft.rfft(rng.standard_normal(length))
    freqs = np.fft.rfftfreq(length, 1 / SAMPLE_RATE)
    shape = np.zeros_like(freqs)
    keep = freqs >= _LOWEST_NOISE_HZ
    shape[keep] = freqs[keep] ** (-exponent / 2)
    noise = np.fft.irfft(spectrum * shape, length)
    return noise / np.sqrt(np.mean(noise**2))


# 돌발 소음 하나의 파형을 만든다. 크기는 mix()가 SNR에 맞게 다시 정하므로 여기서는 모양만 정한다.
# - clatter(부딪는 소리): 시간 상수 50 ms로 줄어드는 잡음.
# - ring(전화벨): 1000 Hz와 1250 Hz를 25 ms마다 번갈아. 주파수를 누적합해 위상을 이어 가므로 바뀌는 순간에
#   끊김 잡음이 없다.
# - horn(경적): 420 Hz와 그 배음 6개를 1/k 크기로 더한 소리.
# - siren(사이렌): 650 → 1300 → 650 Hz로 삼각형 모양으로 오르내리는 소리.
# 끝으로 앞뒤 10 ms를 서서히 키우고 줄인다(짧은 소리는 길이의 절반까지만).
def burst_sound(kind: str, rng: np.random.Generator) -> np.ndarray:
    n = int(SAMPLE_RATE * BURST_SECONDS[kind])
    t = np.arange(n) / SAMPLE_RATE
    if kind == "clatter":
        sound = rng.standard_normal(n) * np.exp(-t / 0.05)
    elif kind == "ring":
        freq = np.where((t // 0.025) % 2 == 0, 1000.0, 1250.0)
        sound = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)
    elif kind == "horn":
        sound = sum(np.sin(2 * np.pi * 420.0 * k * t) / k for k in range(1, 7))
    elif kind == "siren":
        freq = 650.0 + 650.0 * (1.0 - np.abs(2.0 * t / BURST_SECONDS[kind] - 1.0))
        sound = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)
    else:
        raise ValueError(f"unknown burst {kind!r}")
    fade = min(int(SAMPLE_RATE * _FADE_SECONDS), n // 2)
    envelope = np.ones(n)
    envelope[:fade] = np.linspace(0.0, 1.0, fade)
    envelope[n - fade :] = np.linspace(1.0, 0.0, fade)
    return np.asarray(sound, dtype=np.float64) * envelope


# 먹먹함: 1500 Hz 4차 버터워스 저역 통과한 신호(low)와 원래 신호를 가중치 window로 섞는다. x + w·(low − x)는 w
# = 0이면 원래 소리, w = 1이면 저역 통과한 소리이고, 앞뒤 100 ms의 직선 가중치로 자연스럽게 넘어간다. 발화
# 끝을 넘는 가중치는 잘린다.
def _muffle(x: np.ndarray, start: int, params: Realistic) -> np.ndarray:
    ramp = SAMPLE_RATE * params.muffle_transition_ms // 1000
    hold = SAMPLE_RATE * params.muffle_ms // 1000
    weight = np.concatenate([np.linspace(0.0, 1.0, ramp), np.ones(hold), np.linspace(1.0, 0.0, ramp)])
    window = np.zeros(len(x))
    end = min(start + len(weight), len(x))
    window[start:end] = weight[: end - start]
    low = sosfiltfilt(butter(4, params.muffle_cutoff_hz, btype="lowpass", fs=SAMPLE_RATE, output="sos"), x)
    return x + window * (low - x)


# ---------------------------------------------------------------- putting it together


# 말소리 효과와 소리 환경(먹먹함, 배경 소음, 돌발 소음)을 전화 채널 앞에서 섞는다. 반환: (섞은 신호, 최댓값을
# 넘지 않게 곱한 음량 배율).
# 순서는 τ-Voice와 같다: 말소리 효과 → 소리 환경 → (apply의) 채널.
def mix(samples: np.ndarray, plan: Plan, params: Realistic = REALISTIC) -> tuple[np.ndarray, float]:
    """The speech and source effects, before the telephone: (mixture, gain applied to stay under the peak)."""
    x = np.asarray(samples, dtype=np.float64)
    # SNR의 기준이 되는 말소리 크기는 먹먹함을 넣기 전의 발화 전체로 잰다 (docs/experiments.md 11단계 "말소리
    # 쪽은 그 발화 전체(먹먹함 전)"). 먹먹함이 있는 발화에서 소음이 덩달아 작아지지 않게 하려는 것이다.
    speech = _band_rms(x)
    if plan.muffle_start is not None:
        x = _muffle(x, plan.muffle_start, params)
    noise = coloured_noise(len(x), plan.colour, np.random.default_rng(plan.noise_seed))
    # 배경 소음 배율 = 말소리 RMS × 10^(−SNR/20) / 잡음의 대역 RMS. 이렇게 하면 대역 안에서 잰 말소리 대
    # 잡음의 비가 정확히 뽑힌 SNR(dB)이 된다.
    x = x + noise * (speech * 10 ** (-plan.snr_db / 20) / _band_rms(noise))
    for kind, start, snr, seed in plan.bursts:
        sound = burst_sound(kind, np.random.default_rng(seed))
        # 돌발 소음도 같은 공식으로 하나마다 뽑힌 SNR에 맞춘다. 발화 끝을 넘는 부분은 잘라서 더한다.
        sound *= speech * 10 ** (-snr / 20) / _band_rms(sound)
        piece = sound[: len(x) - start]
        x[start : start + len(piece)] += piece
    # 최댓값이 params.peak(0.99)를 넘으면 전체를 같은 비율로 낮춘다. 배율(gain)은 기록에 남아
    # voice_real.report()가 음량을 낮춘 발화 수를 센다.
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    gain = params.peak / peak if peak > params.peak else 1.0
    return x * gain, gain


# Plan 하나로 조건 전체를 적용한다: mix → narrowband(대역, 8 kHz, mu-law) → 잃은 프레임을 0으로 → widen(16
# kHz). 반환은 인식기가 받는 16 kHz float32와 음량 배율이다.
# 잃은 프레임을 8 kHz 신호에서 지우는 이유: 실제 전화망에서 패킷은 코덱을 거친 뒤 잃으므로, 코덱과 16 kHz 복원
# 사이에 넣는다. 메움(손실 은닉)은 하지 않는다.
def apply(samples: np.ndarray, plan: Plan, params: Realistic = REALISTIC) -> tuple[np.ndarray, float]:
    """The whole condition: (16 kHz float32 as the recogniser gets it, gain)."""
    if len(samples) < MIN_SAMPLES:
        return np.asarray(samples, dtype=np.float32), 1.0
    mixed, gain = mix(samples, plan, params)
    narrow = narrowband(mixed)
    # 8 kHz에서 20 ms 프레임은 160표본이다. 마지막 프레임이 짧으면 슬라이스가 신호 끝에서 멈춘다.
    per_frame = NARROW_RATE * params.frame_ms // 1000
    for frame in plan.lost:
        narrow[frame * per_frame : (frame + 1) * per_frame] = 0.0
    return widen(narrow, len(samples)), gain


# 측정에서 부르는 진입점: seed로 조건을 뽑고(draw) 적용한(apply) 뒤, 열화한 소리와 기록용 조건(Plan.to_dict +
# gain)을 돌려준다.
# 호출 위치: voice_real.py _run()이 measure(degrade=dg.degrade)로 넘기고, measure()가 발화마다
# degrade(samples, u.degrade_seed)로 부른다.
def degrade(samples: np.ndarray, seed: int, params: Realistic = REALISTIC) -> tuple[np.ndarray, dict]:
    """(degraded 16 kHz float32, what was done: the plan and the gain)."""
    plan = draw(seed, len(samples), params)
    out, gain = apply(samples, plan, params)
    return out, plan.to_dict() | {"gain": gain}
