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

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

DEGRADE_VERSION = "v4r-1"  # part of the cache key and the manifest: change it with any change below
SAMPLE_RATE = 16_000
NARROW_RATE = 8_000
MU = 255.0
BAND = butter(4, [300, 3400], btype="bandpass", fs=SAMPLE_RATE, output="sos")
MIN_SAMPLES = 64  # shorter than the zero-phase filter's padding: left as it is
# Generated bursts (seconds): a clatter (decaying broadband noise), a ringing phone (two tones alternating
# every 25 ms), a car horn (420 Hz and its harmonics), a siren (650 -> 1300 -> 650 Hz).
BURST_SECONDS = {"clatter": 0.25, "ring": 1.0, "horn": 0.6, "siren": 1.5}
_FADE_SECONDS = 0.01
_LOWEST_NOISE_HZ = 20.0


@dataclass(frozen=True)
class Realistic:
    noise_snr_db: float = 15.0
    noise_snr_spread_db: float = 3.0  # uniform in [15 - 3, 15 + 3] per utterance
    noise_colours: tuple[str, ...] = ("pink", "brown")
    burst_per_minute: float = 1.0  # Poisson
    burst_snr_db: tuple[float, float] = (-5.0, 10.0)  # uniform per burst
    burst_kinds: tuple[str, ...] = ("clatter", "ring", "horn", "siren")
    muffle_probability: float = 0.2  # of utterances
    muffle_ms: int = 500  # one low-passed segment ...
    muffle_transition_ms: int = 100  # ... faded in and out over this
    muffle_cutoff_hz: float = 1500.0
    frame_ms: int = 20  # one G.711 packet
    loss_rate: float = 0.02  # Gilbert-Elliott: share of lost frames ...
    loss_burst_ms: float = 100.0  # ... and the mean length of a run of them
    peak: float = 0.99  # a louder mixture is turned down before the codec instead of clipped


REALISTIC = Realistic()


def describe(params: Realistic = REALISTIC) -> dict:
    """What the cache key and the manifest record."""
    values = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(params).items()}
    return {"degrade": DEGRADE_VERSION, **values}


# ---------------------------------------------------------------- the telephone (whisper-ko-ft channel.py)


def mu_law_roundtrip(samples: np.ndarray) -> np.ndarray:
    """Compress to 8-bit mu-law codes and expand back."""
    clipped = np.clip(samples, -1.0, 1.0)
    compressed = np.sign(clipped) * np.log1p(MU * np.abs(clipped)) / np.log1p(MU)
    codes = np.round((compressed + 1.0) / 2.0 * 255.0)  # 256 levels
    restored = codes / 255.0 * 2.0 - 1.0
    return (np.sign(restored) * np.expm1(np.abs(restored) * np.log1p(MU)) / MU).astype(np.float32)


def narrowband(samples: np.ndarray) -> np.ndarray:
    """16 kHz audio through the 300-3400 Hz band, at 8 kHz, after the mu-law codec."""
    return mu_law_roundtrip(resample_poly(sosfiltfilt(BAND, samples), 1, 2))


def widen(narrow: np.ndarray, length: int) -> np.ndarray:
    """The 8 kHz signal back at 16 kHz, cut to the original length."""
    return resample_poly(narrow, 2, 1)[:length].astype(np.float32)


def telephone(samples: np.ndarray) -> np.ndarray:
    """16 kHz audio as it would sound after a narrowband telephone channel, still at 16 kHz."""
    if len(samples) < MIN_SAMPLES:
        return samples.astype(np.float32)
    return widen(narrowband(samples), len(samples))


# ---------------------------------------------------------------- what is drawn for one utterance


def frame_count(length: int) -> int:
    """20 ms frames of the 8 kHz signal of `length` samples at 16 kHz (the last one may be short)."""
    narrow = (length + 1) // 2
    per_frame = NARROW_RATE * REALISTIC.frame_ms // 1000
    return -(-narrow // per_frame)


def gilbert_elliott(frames: int, rate: float, burst_frames: float, rng: np.random.Generator) -> np.ndarray:
    """Lost frames of a two-state chain: every frame of the bad state is lost, none of the good state. Leaving
    the bad state has probability 1/burst_frames (mean run), entering it is set so that `rate` are lost."""
    leave = 1.0 / burst_frames
    enter = rate * leave / (1.0 - rate)
    draws = rng.random(frames + 1)
    lost = np.zeros(frames, dtype=bool)
    bad = draws[0] < rate  # the first frame from the chain's stationary share
    for i in range(frames):
        lost[i] = bad
        bad = draws[i + 1] < (1.0 - leave if bad else enter)
    return lost


@dataclass(frozen=True)
class Plan:
    muffle_start: int | None  # first sample of the muffled span (ramp, segment, ramp); None: not muffled
    colour: str
    snr_db: float
    noise_seed: int
    bursts: tuple[tuple[str, int, float, int], ...]  # (kind, first sample, SNR dB, seed of its sound)
    frames: int
    lost: tuple[int, ...]  # indices of lost 20 ms frames

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


def draw(seed: int, length: int, params: Realistic = REALISTIC) -> Plan:
    """Everything random about one utterance of `length` samples. Each effect has its own stream, so the
    draws of one do not move when another changes."""
    muffle_rng, noise_rng, burst_rng, loss_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
    )
    muffle_start = None
    if muffle_rng.random() < params.muffle_probability:
        span = SAMPLE_RATE * (params.muffle_ms + 2 * params.muffle_transition_ms) // 1000
        muffle_start = int(muffle_rng.integers(0, max(length - span, 0) + 1))
    colour = params.noise_colours[int(noise_rng.integers(len(params.noise_colours)))]
    low, high = (
        params.noise_snr_db - params.noise_snr_spread_db,
        params.noise_snr_db + params.noise_snr_spread_db,
    )
    snr_db = float(noise_rng.uniform(low, high))
    noise_seed = int(noise_rng.integers(2**31))
    bursts = []
    for _ in range(int(burst_rng.poisson(length / SAMPLE_RATE / 60.0 * params.burst_per_minute))):
        kind = params.burst_kinds[int(burst_rng.integers(len(params.burst_kinds)))]
        start = int(burst_rng.integers(0, max(length, 1)))
        snr = float(burst_rng.uniform(*params.burst_snr_db))
        bursts.append((kind, start, snr, int(burst_rng.integers(2**31))))
    frames = frame_count(length)
    lost = gilbert_elliott(frames, params.loss_rate, params.loss_burst_ms / params.frame_ms, loss_rng)
    return Plan(
        muffle_start, colour, snr_db, noise_seed, tuple(bursts), frames, tuple(np.flatnonzero(lost).tolist())
    )


# ---------------------------------------------------------------- sounds


def _band_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(sosfiltfilt(BAND, x) ** 2)))


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


def mix(samples: np.ndarray, plan: Plan, params: Realistic = REALISTIC) -> tuple[np.ndarray, float]:
    """The speech and source effects, before the telephone: (mixture, gain applied to stay under the peak)."""
    x = np.asarray(samples, dtype=np.float64)
    speech = _band_rms(x)
    if plan.muffle_start is not None:
        x = _muffle(x, plan.muffle_start, params)
    noise = coloured_noise(len(x), plan.colour, np.random.default_rng(plan.noise_seed))
    x = x + noise * (speech * 10 ** (-plan.snr_db / 20) / _band_rms(noise))
    for kind, start, snr, seed in plan.bursts:
        sound = burst_sound(kind, np.random.default_rng(seed))
        sound *= speech * 10 ** (-snr / 20) / _band_rms(sound)
        piece = sound[: len(x) - start]
        x[start : start + len(piece)] += piece
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    gain = params.peak / peak if peak > params.peak else 1.0
    return x * gain, gain


def apply(samples: np.ndarray, plan: Plan, params: Realistic = REALISTIC) -> tuple[np.ndarray, float]:
    """The whole condition: (16 kHz float32 as the recogniser gets it, gain)."""
    if len(samples) < MIN_SAMPLES:
        return np.asarray(samples, dtype=np.float32), 1.0
    mixed, gain = mix(samples, plan, params)
    narrow = narrowband(mixed)
    per_frame = NARROW_RATE * params.frame_ms // 1000
    for frame in plan.lost:
        narrow[frame * per_frame : (frame + 1) * per_frame] = 0.0
    return widen(narrow, len(samples)), gain


def degrade(samples: np.ndarray, seed: int, params: Realistic = REALISTIC) -> tuple[np.ndarray, dict]:
    """(degraded 16 kHz float32, what was done: the plan and the gain)."""
    plan = draw(seed, len(samples), params)
    out, gain = apply(samples, plan, params)
    return out, plan.to_dict() | {"gain": gain}
