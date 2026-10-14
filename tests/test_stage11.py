"""11단계(V4-R, A단계): 9단계 V4 개발용 실행의 고객 발화를 같은 seed로 다시 합성해, 깨끗한 그대로와 전화
통화 조건을 거친 뒤로 두 번 받아 적는다(docs/experiments.md).

신호 테스트는 numpy와 scipy(`voice` 의존성 그룹)가 필요하고, 없으면 건너뛴다. 실행기와 보고서는 가짜 음성
모델과 손으로 만든 기록으로 시험하므로 어디서나 돈다.
"""

# [파일 역할] 11단계 V4-R A단계(전화 통화 조건에서 인식만 재기)를 검증하는 테스트 묶음이다. 구역: 열화
# 구현(voice/degrade.py: 전화 대역·μ-law, 등록한 τ-Voice Realistic 값, 생성 소음, 돌발 소음, 먹먹함, 프레임
# 손실, seed 고정), 원본 발화(9단계 V4 개발용 기록), 실행기(가짜 음성 모델, 캐시), 결과 표(voice_real report),
# 인식기가 디코딩한 표본을 받는 경로.
# [왜 중요한가] 이 단계는 같은 소리를 깨끗한 채로 한 번, 열화를 거쳐 한 번 받아 적어 차이를 열화의 몫으로
# 본다(docs/experiments.md 11단계). 열화 값이 등록한 값과 다르거나, seed가 고정되지 않거나, 깨끗한 쪽이 9단계
# 채널과 다른 소리가 되면 짝지은 차이를 열화 탓으로 읽을 수 없다. 측정 조건이라 채택 판정은 없다.
# [방법] 신호 테스트는 numpy·scipy(voice 그룹)가 있을 때만 돌고 없으면 건너뛴다. 실행기와 표는 가짜 음성
# 모델과 손으로 만든 기록으로 어디서나 돈다.

from __future__ import annotations

# math: SNR(dB) 계산과 포아송 표준편차. Path: docs/experiments.md 원문을 읽어 등록한 값이 문서에 있는지 본다.
import io
import json
import math
from pathlib import Path

import pytest

# voice_real: 11단계 실행기·관문·표(검증 대상). derive_seed: 합성 seed('voice')와 열화 seed('degrade')가 기록
# 규칙대로인지 본다. normalize_heard: 두 받아쓰기에 모두 V2 정규화를 거친다. Audio: 가짜 합성기의 출력 형식.
from support_agent import voice_real
from support_agent.config import derive_seed
from support_agent.paths import REPORTS
from support_agent.voice.normalize import normalize_heard
from support_agent.voice.speech import Audio

# 원본 실행: 9단계 V4 개발용 기록(reports/에 커밋됨). 이 기록의 고객 발화 929개를 다시 합성한다.
SOURCE = REPORTS / voice_real.SOURCE_RUN


# 열화 구현(voice/degrade.py) 테스트. 순서는 말소리 효과(먹먹함) → 소리 환경(배경·돌발 소음, 음량) → 채널(전화
# 대역, 8 kHz, μ-law, 프레임 손실) → 16 kHz.
# ==================================================================== 열화 (voice/degrade.py)


# degrade 모듈. numpy·scipy가 없으면 이 픽스처를 쓰는 테스트를 모두 건너뛴다(CI는 voice 그룹 없이 돈다).
@pytest.fixture(scope="module")
def dg():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from support_agent.voice import degrade

    return degrade


# numpy 모듈(없으면 건너뜀).
@pytest.fixture(scope="module")
def np():
    return pytest.importorskip("numpy")


# 16 kHz 사인파 시험 신호.
def tone(np, frequency, seconds=1.0, amplitude=0.5):
    t = np.arange(int(16_000 * seconds)) / 16_000
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


# FFT로 잰 주파수 구간 [low, high)의 에너지. 대역 밖 소리가 걸러졌는지 볼 때 쓴다.
def band_energy(np, samples, low, high):
    spectrum = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1 / 16_000)
    return float(spectrum[(freqs >= low) & (freqs < high)].sum())


# 전화 대역과 μ-law: whisper-ko-ft channel.py의 telephone()과 같은 구현이어야 해서, 그 코드로 계산한
# 출력값으로 고정한다.
# --- 전화 대역과 mu-law: whisper-ko-ft channel.py의 구현이며, 그 출력값으로 고정한다


# 네 사인파(두 개는 대역 밖)를 넣은 출력의 표본 8개, 전체 에너지, μ-law 왕복 값이 whisper-ko-ft 코드로 계산한
# 기준값과 같은지 본다. 같은 PC에서 그 프로젝트가 잰 '전화 대역·μ-law만의 영향'과 같은 구현이어야 두 결과를
# 이어 읽을 수 있다(docs/experiments.md 11단계 "왜 하는가").
def test_telephone_gives_what_whisper_ko_fts_channel_gives(dg, np):
    """같은 입력(사인파 넷, 그중 둘은 대역 밖)에 whisper-ko-ft src/whisper_ko_ft/channel.py(커밋 51616d5의
    코드, numpy 2.4.6, scipy 1.17.1)로 계산한 기준값."""
    t = np.arange(8000) / 16_000
    x = (
        0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 1800 * t)
        + 0.1 * np.sin(2 * np.pi * 5000 * t)
        + 0.05 * np.sin(2 * np.pi * 120 * t)
    ).astype(np.float32)
    out = dg.telephone(x)
    assert out.dtype == np.float32 and len(out) == 8000
    expected = [
        0.02753211371600628, 0.32141777873039246, 0.3818966746330261, -0.17847180366516113,
        -0.35569456219673157, -0.37445613741874695, 0.17847181856632233, -0.18846286833286285,
    ]  # fmt: skip
    assert out[[123, 777, 2048, 3001, 4567, 5555, 6999, 7890]] == pytest.approx(expected, abs=1e-6)
    assert float(np.sum(out.astype(np.float64) ** 2)) == pytest.approx(509.7189788331785, rel=1e-6)
    levels = dg.mu_law_roundtrip(np.array([-1.0, -0.5, -0.01, 0.0, 0.003, 0.25, 0.9, 1.0]))
    assert levels == pytest.approx(
        [-1.0, -0.49667662382125854, -0.01022530347108841, 8.621159213362262e-05,
         0.0031326664611697197, 0.24569807946681976, 0.9163658022880554, 1.0], abs=1e-7
    )  # fmt: skip


# telephone()을 narrowband(8 kHz까지)와 widen(16 kHz로 복원) 두 함수로 나눠도 이어 붙인 출력이 같은지 본다.
# 프레임 손실을 코덱과 복원 사이에 넣으려고 나눈 것이라, 나누면서 결과가 바뀌면 안 된다.
def test_telephone_is_the_narrow_band_widened_again(dg, np):
    x = np.random.default_rng(3).normal(0, 0.1, 12_345).astype(np.float32)
    assert np.array_equal(dg.telephone(x), dg.widen(dg.narrowband(x), len(x)))
    assert len(dg.narrowband(x)) == 6173  # 8 kHz


# 출력이 입력과 같은 길이의 float32인지 본다. 리샘플링으로 길이가 달라지면 인식기 입력과 기록이 어긋난다.
def test_telephone_keeps_length_and_dtype(dg, np):
    out = dg.telephone(tone(np, 1000, 0.73))
    assert out.dtype == np.float32 and len(out) == int(16_000 * 0.73)


# 말소리 대역(1 kHz)의 소리는 채널을 지나도 원래와 상관계수 0.98 넘게 남는지 본다.
def test_a_speech_band_tone_survives_the_telephone(dg, np):
    original = tone(np, 1000)
    assert np.corrcoef(original, dg.telephone(original))[0, 1] > 0.98


# 대역 밖 소리(100 Hz, 6 kHz)는 에너지가 2% 아래로 줄어드는지 본다.
def test_tones_outside_the_band_are_removed(dg, np):
    for frequency in (100, 6000):
        out = dg.telephone(tone(np, frequency))
        assert band_energy(np, out, 0, 8000) < 0.02 * band_energy(np, tone(np, frequency), 0, 8000)


# 8 kHz로 낮췄다가 되돌린 뒤 4.2 kHz 위에는 거의 아무것도 없는지 본다(되돌려도 높은 소리는 살아나지 않음).
def test_nothing_is_left_above_4khz(dg, np):
    out = dg.telephone(np.random.default_rng(0).normal(0, 0.1, 16_000).astype(np.float32))
    assert band_energy(np, out, 4200, 8000) < 0.001 * band_energy(np, out, 300, 3400)


# μ-law 왕복이 8비트(256단계) 이하로 양자화하고, 작은 소리(0.01)는 오차가 작게 남는지 본다. 작은 소리를 촘촘히
# 양자화하는 μ-law의 성질이다.
def test_mu_law_uses_at_most_256_levels_and_keeps_quiet_samples(dg, np):
    restored = dg.mu_law_roundtrip(np.linspace(-1, 1, 100_000).astype(np.float32))
    assert len(np.unique(restored)) <= 256
    assert abs(dg.mu_law_roundtrip(np.array([np.float32(0.01)]))[0] - 0.01) < 0.002


# 필터 패딩보다 짧은 입력은 바꾸지 않고 돌려주고, 전체 열화도 그대로 두며 음량 배율이 1인지 본다.
def test_very_short_input_is_returned_unchanged(dg, np):
    short = np.zeros(10, dtype=np.float32)
    assert np.array_equal(dg.telephone(short), short)
    out, info = dg.degrade(short, seed=1)
    assert np.array_equal(out, short) and info["gain"] == 1.0


# 조건: τ-Voice Realistic 값을 재기 전에 등록했다.
# --- 조건: tau-Voice "Realistic" 값, 재기 전에 고정


# 코드의 열화 값(배경 소음 15 dB ±3, 분홍·갈색, 돌발 소음 분당 1회 −5~+10 dB와 네 종류, 먹먹함 20%·500 ms·전환
# 100 ms·1500 Hz, 프레임 20 ms·손실 2%·묶음 100 ms, 최댓값 0.99)과 버전 v4r-1이 등록한 표와 같고, 설명이
# JSON으로 왕복되는지 본다. 이 설명이 manifest와 캐시 키에 들어간다.
def test_the_registered_values_are_tau_voice_realistic(dg):
    p = dg.REALISTIC
    assert (p.noise_snr_db, p.noise_snr_spread_db, p.noise_colours) == (15.0, 3.0, ("pink", "brown"))
    assert (p.burst_per_minute, p.burst_snr_db) == (1.0, (-5.0, 10.0))
    assert p.burst_kinds == ("clatter", "ring", "horn", "siren")
    assert (p.muffle_probability, p.muffle_ms, p.muffle_transition_ms, p.muffle_cutoff_hz) == (
        0.2, 500, 100, 1500.0
    )  # fmt: skip
    assert (p.frame_ms, p.loss_rate, p.loss_burst_ms, p.peak) == (20, 0.02, 100.0, 0.99)
    described = dg.describe()
    assert described["degrade"] == dg.DEGRADE_VERSION == "v4r-1"
    assert described["noise_snr_db"] == 15.0 and described["burst_snr_db"] == [-5.0, 10.0]
    assert json.loads(json.dumps(described)) == described


# 생성 잡음의 스펙트럼 기울기가 색에 맞고(분홍 −1, 갈색 −2, 흰색 0, 로그-로그), RMS가 1로 정규화되며, 20 Hz
# 아래 성분이 없는지 본다. 녹음을 쓰지 않고 소음을 생성하기로 했으므로 생성한 소음이 그 색의 성질을 갖는지가
# 중요하다.
def test_coloured_noise_has_the_slope_of_its_colour_and_nothing_below_20hz(dg, np):
    from scipy.signal import welch

    for colour, slope in (("pink", -1.0), ("brown", -2.0), ("white", 0.0)):
        noise = dg.coloured_noise(160_000, colour, np.random.default_rng(5))
        assert math.isclose(float(np.sqrt(np.mean(noise**2))), 1.0, rel_tol=1e-9)
        freqs, power = welch(noise, fs=16_000, nperseg=8192)
        keep = (freqs >= 100) & (freqs <= 4000)
        fitted = np.polyfit(np.log10(freqs[keep]), np.log10(power[keep]), 1)[0]
        assert fitted == pytest.approx(slope, abs=0.15), colour
    spectrum = np.abs(np.fft.rfft(dg.coloured_noise(160_000, "brown", np.random.default_rng(6)))) ** 2
    assert spectrum[np.fft.rfftfreq(160_000, 1 / 16_000) < 20].sum() < 1e-12 * spectrum.sum()


# Gilbert–Elliott 손실이 평균 2%를, 평균 5프레임(100 ms) 묶음으로 잃는지 본다.
def test_gilbert_elliott_loses_two_percent_in_bursts_of_100_ms(dg, np):
    lost = dg.gilbert_elliott(300_000, 0.02, 5, np.random.default_rng(7))
    assert lost.mean() == pytest.approx(0.02, abs=0.003)
    starts = np.flatnonzero(lost & ~np.concatenate(([False], lost[:-1])))
    assert lost.sum() / len(starts) == pytest.approx(5.0, abs=0.4)  # 연달아 잃는 20 ms 프레임 수의 평균


# 열화 조건(plan)이 seed로 정해지고(같은 seed는 같은 조건, 다른 seed는 다른 조건), JSON으로 왕복되며, 프레임
# 수가 맞는지 본다. 열화 seed가 (기본 seed, 과제, 시도, 'degrade', 순번)에서 나오므로 같은 명령은 같은 조건을
# 다시 낸다.
def test_the_plan_is_fixed_by_the_seed(dg):
    a, b, c = dg.draw(11, 80_000), dg.draw(11, 80_000), dg.draw(12, 80_000)
    assert a == b and a != c
    assert json.loads(json.dumps(a.to_dict())) == a.to_dict()
    assert a.frames == math.ceil(40_000 / 160)


# 5초 발화 3,000개의 조건이 등록한 비율을 따르는지 본다: 먹먹함 약 20%, SNR 12~18 dB에 평균 15, 분홍 반, 돌발
# 소음 분당 1회(5초에 1/12), 네 종류 모두, 손실 약 2%. CPU 관문(voice_real plan)과 같은 성격의 확인이다.
def test_the_plans_of_many_utterances_follow_the_registered_rates(dg, np):
    plans = [dg.draw(seed, 5 * 16_000) for seed in range(3000)]
    assert sum(p.muffle_start is not None for p in plans) / 3000 == pytest.approx(0.2, abs=0.03)
    snrs = [p.snr_db for p in plans]
    assert 12.0 <= min(snrs) and max(snrs) <= 18.0 and sum(snrs) / 3000 == pytest.approx(15.0, abs=0.3)
    assert sum(p.colour == "pink" for p in plans) / 3000 == pytest.approx(0.5, abs=0.05)
    bursts = [b for p in plans for b in p.bursts]
    assert len(bursts) / 3000 == pytest.approx(5 / 60, abs=0.015)  # 분당 1회
    assert {b[0] for b in bursts} == set(dg.REALISTIC.burst_kinds)
    assert all(-5.0 <= b[2] <= 10.0 and 0 <= b[1] < 80_000 for b in bursts)
    frames = sum(p.frames for p in plans)
    assert sum(len(p.lost) for p in plans) / frames == pytest.approx(0.02, abs=0.004)


# 전화 대역(300–3400 Hz) 필터를 거친 RMS. SNR을 전 대역이 아니라 전화 대역 안에서 재기로 했기 때문이다(갈색
# 잡음은 힘 대부분이 300 Hz 아래라 전 대역으로 재면 쉬운 조건이 된다).
def band_rms(dg, np, x):
    from scipy.signal import sosfiltfilt

    return float(np.sqrt(np.mean(sosfiltfilt(dg.BAND, np.asarray(x, dtype=np.float64)) ** 2)))


# 모든 효과를 끈 조건에서 시험할 효과 하나만 켠 plan을 만든다.
def quiet_plan(dg, length, **changes):
    """모든 효과를 끈 plan: 소음은 말소리보다 300 dB 아래, 먹먹함·손실·돌발 소음 없음."""
    base = dict(muffle_start=None, colour="pink", snr_db=300.0, noise_seed=1, bursts=(), frames=0, lost=())
    base.update(changes)
    base["frames"] = dg.frame_count(length)
    return dg.Plan(**base)


# 배경 소음을 섞은 뒤 전화 대역 안에서 잰 SNR이 뽑힌 값(13.5 dB)과 0.05 dB 안으로 같은지 분홍·갈색 모두 본다.
def test_the_background_noise_is_at_the_drawn_snr_inside_the_telephone_band(dg, np):
    speech = tone(np, 700, 3.0, 0.2) + tone(np, 2200, 3.0, 0.05)
    for colour in ("pink", "brown"):
        plan = quiet_plan(dg, len(speech), colour=colour, snr_db=13.5, noise_seed=4)
        mixed, gain = dg.mix(speech, plan)
        assert gain == 1.0
        snr = 20 * math.log10(band_rms(dg, np, speech) / band_rms(dg, np, mixed - speech))
        assert snr == pytest.approx(13.5, abs=0.05), colour


# 돌발 소음(전화벨)이 정한 위치에서 정한 길이만큼만 들어가고, 그 구간의 SNR이 뽑힌 값(−2 dB)인지 본다.
def test_a_burst_is_at_its_snr_and_place(dg, np):
    speech = tone(np, 700, 3.0, 0.2)
    plan = quiet_plan(dg, len(speech), bursts=(("ring", 16_000, -2.0, 9),))
    mixed, _gain = dg.mix(speech, plan)
    added = mixed - speech
    ring = int(16_000 * dg.BURST_SECONDS["ring"])
    assert np.abs(added[:16_000]).max() < 1e-6 and np.abs(added[16_000 + ring :]).max() < 1e-6
    burst_rms = band_rms(dg, np, added[16_000 : 16_000 + ring])
    assert 20 * math.log10(band_rms(dg, np, speech) / burst_rms) == pytest.approx(-2.0, abs=0.1)


# 발화 끝 가까이에서 시작한 돌발 소음은 끝에서 잘리고 발화 길이는 그대로인지 본다.
def test_a_burst_near_the_end_is_cut_there(dg, np):
    speech = tone(np, 700, 1.0, 0.2)
    plan = quiet_plan(dg, len(speech), bursts=(("siren", 15_000, 5.0, 2),))
    mixed, _gain = dg.mix(speech, plan)
    assert len(mixed) == 16_000 and np.abs((mixed - speech)[15_000:]).max() > 0


# 네 종류의 돌발 소음이 정한 길이의 유한한 소리이고, 힘의 상당 부분이 전화 대역 안에 있으며(채널이 걸러 내
# 없어지지 않게), 앞뒤를 서서히 키우고 줄여 딸깍 소리가 없는지 본다.
@pytest.mark.parametrize("kind", ["clatter", "ring", "horn", "siren"])
def test_every_burst_kind_is_sound_in_the_telephone_band(dg, np, kind):
    sound = dg.burst_sound(kind, np.random.default_rng(0))
    assert len(sound) == int(16_000 * dg.BURST_SECONDS[kind]) and np.isfinite(sound).all()
    assert band_rms(dg, np, sound) > 0.1 * float(np.sqrt(np.mean(sound**2)))
    assert sound[0] == 0 and abs(sound[-1]) < 1e-3  # 서서히 들어가고 나온다: 딸깍 소리 없음


# 먹먹함이 정한 구간(500 ms와 앞뒤 100 ms 전환)에서만 3 kHz 소리를 크게 낮추고, 구간 밖은 그대로 두는지 본다.
def test_muffling_lowers_high_frequencies_only_in_its_segment(dg, np):
    speech = tone(np, 3000, 3.0, 0.2)
    plan = quiet_plan(dg, len(speech), muffle_start=8000)
    mixed, _gain = dg.mix(speech, plan)
    plateau = slice(8000 + 1600 + 200, 8000 + 1600 + 8000 - 200)  # 500 ms 안, 경사 구간에서 먼 곳
    assert np.abs(mixed[plateau]).max() < 0.02 * 0.2
    span = 8000 + 2 * 1600 + 8000
    assert np.abs(mixed[:8000] - speech[:8000]).max() < 1e-4
    assert np.abs(mixed[span:] - speech[span:]).max() < 1e-4


# 잃은 20 ms 프레임 두 개가 채널을 지난 뒤 무음이고(메움 없음), 그 앞뒤의 소리는 남는지 본다. 필터 가장자리
# 표본은 빼고 본다.
def test_lost_frames_are_silence_after_the_channel(dg, np):
    speech = tone(np, 700, 1.0, 0.3)
    plan = quiet_plan(dg, len(speech), lost=(10, 11))
    out = dg.apply(speech, plan)[0]
    gap = out[10 * 320 + 40 : 12 * 320 - 40]  # 20 ms 프레임 둘의 16 kHz 표본, 필터 가장자리 제외
    assert np.abs(gap).max() < 1e-3
    assert np.abs(out[5 * 320 : 9 * 320]).max() > 0.2 and np.abs(out[13 * 320 : 20 * 320]).max() > 0.2


# 큰 돌발 소음으로 섞은 소리의 최댓값이 0.99를 넘으면, 코덱에서 잘리는 대신 발화 전체를 낮춰 최댓값을 0.99로
# 맞추는지 본다.
def test_a_loud_mixture_is_turned_down_before_the_codec_instead_of_clipped(dg, np):
    speech = tone(np, 700, 2.0, 0.95)
    plan = quiet_plan(dg, len(speech), bursts=(("horn", 4000, -5.0, 1),))
    mixed, gain = dg.mix(speech, plan)
    assert gain < 1.0 and np.abs(mixed).max() == pytest.approx(0.99, abs=1e-6)


# 전체 열화가 seed로 고정되고(같은 seed는 출력·정보가 같고, 다른 seed는 다름), 16 kHz float32 같은 길이를
# 내며, 돌려준 정보가 뽑힌 조건에 음량 배율을 더한 것인지 본다. 이 정보가 발화마다 기록에 남는다.
def test_degrade_is_fixed_by_its_seed_and_returns_16k_float32(dg, np):
    speech = tone(np, 700, 4.0, 0.2) + tone(np, 1900, 4.0, 0.05)
    a, info_a = dg.degrade(speech, seed=21)
    b, info_b = dg.degrade(speech, seed=21)
    c, _ = dg.degrade(speech, seed=22)
    assert a.dtype == np.float32 and len(a) == len(speech)
    assert np.array_equal(a, b) and info_a == info_b and not np.array_equal(a, c)
    assert info_a == dg.draw(21, len(speech)).to_dict() | {"gain": info_a["gain"]}


# 원본 발화: 9단계 V4 개발용 기록에서 다시 합성할 발화를 읽는다.
# ==================================================================== 원본 발화


# 원본 실행의 manifest와 발화 목록(모듈에서 한 번만 읽음).
@pytest.fixture(scope="module")
def source():
    return voice_real.source_utterances(SOURCE)


# 원본이 9단계 V4 개발용 공식 실행이고, 발화 929개가 96개 에피소드·24개 과제에서 나오는지
# 본다(docs/experiments.md 11단계 "이 단계가 재는 것").
def test_the_source_is_the_stage_9_v4_development_run(source):
    manifest, utterances = source
    assert manifest["config"]["voice"] == "V4" and manifest["tasks_file"] == "dev" and manifest["official"]
    assert len(utterances) == 929 and len({(u.task_id, u.trial) for u in utterances}) == 96
    assert len({u.task_id for u in utterances}) == 24


# 다시 합성할 seed가 9단계 채널이 쓴 seed('voice')와 같고 열화 seed('degrade')는 그와 다른지, 모든 발화의 말한
# 글·풀어 쓴 글·기록된 받아쓰기가 원본 기록과 같은지 본다. 같은 말을 같은 seed로 합성해야 깨끗한 쪽이 9단계와
# 같은 소리가 된다.
def test_the_seeds_are_the_ones_the_channel_used_and_a_new_one_for_the_degradation(source):
    _manifest, utterances = source
    first = utterances[0]
    assert (first.task_id, first.trial, first.index) == ("dev-001", 0, 0)
    assert first.seed == derive_seed(1000, "dev-001", 0, "voice", 0)
    assert first.degrade_seed == derive_seed(1000, "dev-001", 0, "degrade", 0)
    assert first.seed != first.degrade_seed
    records = [json.loads(line) for line in (SOURCE / "episodes.jsonl").read_text("utf-8").splitlines()]
    voice = {(e["task_id"], e["trial"]): e["voice"] for e in records}
    for u in utterances:
        recorded = voice[(u.task_id, u.trial)][u.index]
        assert (u.said, u.spoken, u.recorded_heard, u.recorded_text) == (
            recorded["said"], recorded["spoken"], recorded["heard"], recorded["text"]
        )  # fmt: skip


# 발화마다 과제 고객의 등록 이름이 붙고, 말한 글에 그 이름이 든 발화가 165개인지 본다. 이 165개가 1차 지표
# '이름 정확 생존'의 분모다.
def test_each_utterance_knows_the_registered_name_of_its_customer(source):
    from support_agent.analyze import seed_names
    from support_agent.tasks import load_tasks

    _manifest, utterances = source
    names, tasks = seed_names(), {t.id: t for t in load_tasks("dev")}
    assert all(u.name == names[tasks[u.task_id].customer_id] for u in utterances)
    assert sum(voice_real.name_in(u.name, u.said) for u in utterances) == 165


# 시험용 기록과 음성이 아닌 실행은 원본으로 받지 않는지 본다. 시험용 기록은 열지 않는다는 규칙과, 다시 합성할
# 발화가 있어야 한다는 전제를 코드가 지킨다.
def test_test_records_and_text_runs_are_not_sources(tmp_path):
    for config, tasks_file, message in (
        ({"voice": "V4", "base_seed": 1}, "test", "development"),
        ({"voice": "V0", "base_seed": 1}, "dev", "voice run"),
    ):
        run = tmp_path / f"{tasks_file}-{config['voice']}"
        run.mkdir()
        manifest = {"config": config, "tasks_file": tasks_file, "voice_channel": None, "task_sha256": {}}
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run / "episodes.jsonl").write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            voice_real.source_utterances(run)


# manifest의 과제 해시가 지금 과제 파일과 다르면(과제가 바뀜) 원본으로 받지 않는지 본다.
def test_a_source_whose_task_file_changed_is_refused(tmp_path):
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    manifest["task_sha256"]["dev-001"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "episodes.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="dev-001"):
        voice_real.source_utterances(tmp_path)


# 음성 모델 설명(MeloTTS·torch·faster-whisper·ctranslate2 버전, 장치, fp16, beam 5, VAD 끔)이 원본 실행과
# 같으면 차이가 없고, 버전이나 장치가 하나라도 다르면 그 항목을 짚는지 본다. 다르면 공식 실행을 시작하지
# 않는다(재현 확인 규칙).
def test_the_speech_models_must_be_the_ones_of_the_source_run():
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    tts = {"tts": "melotts/KR", "device": "cuda", "speed": 1.0,
           "tts_versions": {"melotts": "0.1.2", "torch": "2.11.0+cu128"}}  # fmt: skip
    stt = {"stt": "faster-whisper/large-v3-turbo", "device": "cuda", "compute_type": "float16",
           "beam_size": 5, "vad_filter": False,
           "stt_versions": {"faster-whisper": "1.2.1", "ctranslate2": "4.8.2"}}  # fmt: skip
    assert voice_real.channel_mismatches(manifest, tts, stt) == []
    other = dict(stt, stt_versions={"faster-whisper": "1.2.2", "ctranslate2": "4.8.2"})
    assert voice_real.channel_mismatches(manifest, tts, other) == ["stt_versions"]
    assert voice_real.channel_mismatches(manifest, dict(tts, device="cpu"), stt) == ["tts device"]


# 실행기: 가짜 음성 모델로 발화마다 기록 한 줄을 남기고, 캐시에서 이어 가는지 본다.
# ==================================================================== 실행기 (가짜 음성 모델)


# 가짜 합성기: 풀어 쓴 글을 그대로 '소리'로 돌려주고 불린 글과 seed를 기록한다.
class FakeSpeaker:
    def __init__(self):
        self.calls = []

    def describe(self):
        return {"tts": "fake", "device": "cpu"}

    def synthesize(self, spoken_text, seed=0):
        self.calls.append((spoken_text, seed))
        return Audio(spoken_text.encode("utf-8"), 0.5 + len(spoken_text) / 10)


# 가짜 인식기: 깨끗한 소리는 그대로, 열화를 거친 소리는 마지막 글자를 잃은 채 받아 적는다. 두 받아쓰기가
# 다르게 나와 짝이 섞이지 않는지 볼 수 있다.
class FakeListener:
    """깨끗한 소리는 말한 그대로 듣고, 열화를 거친 소리는 마지막 글자를 잃는다."""

    def describe(self):
        return {"stt": "fake", "device": "cpu"}

    def transcribe_samples(self, samples):
        text = samples["text"]
        return text[:-1] if samples.get("degraded") else text


# 가짜 디코더와 가짜 열화: 열화 표시를 붙이고 seed로 정해지는 조건 정보를 돌려준다.
def fake_decode(wav):
    return {"text": wav.decode("utf-8")}


def fake_degrade(samples, seed):
    return dict(samples, degraded=True), {"muffle_start_s": None, "lost": [seed % 3], "gain": 1.0}


# 원본 발화 앞의 n개.
def few(source, n=6):
    return source[1][:n]


# measure가 발화마다 원본의 풀어 쓴 글과 합성 seed로 합성하고, 깨끗한·열화한 받아쓰기와 각각의 V2 정규화, 뽑힌
# 조건을 한 줄씩 utterances.jsonl에 남기는지 본다. 같은 캐시로 다시 돌리면 모두 캐시에서 같은 기록이 나오고,
# 열화 설명이 바뀌면 캐시 키가 달라져 아무것도 캐시에서 오지 않는지 확인한다(캐시 키에 열화의 버전과 값이
# 들어감).
def test_the_runner_writes_one_record_per_utterance_and_resumes_from_its_cache(tmp_path, source):
    utterances = few(source)
    speaker, cache = FakeSpeaker(), tmp_path / "cache.jsonl"
    kwargs = dict(decode=fake_decode, degrade=fake_degrade, description={"degrade": "test-1"})
    out1 = tmp_path / "one"
    out1.mkdir()
    records = voice_real.measure(utterances, speaker, FakeListener(), out=out1, cache_path=cache, **kwargs)
    assert [(s, seed) for s, seed in speaker.calls] == [(u.spoken, u.seed) for u in utterances]
    assert [r["index"] for r in records] == [u.index for u in utterances] and not any(
        r["cached"] for r in records
    )
    first = records[0]
    assert (
        first["clean_heard"] == utterances[0].spoken and first["degraded_heard"] == utterances[0].spoken[:-1]
    )
    assert first["clean_text"] == normalize_heard(first["clean_heard"])
    assert first["degraded_text"] == normalize_heard(first["degraded_heard"])
    assert first["plan"]["lost"] == [utterances[0].degrade_seed % 3]
    written = [json.loads(line) for line in (out1 / "utterances.jsonl").read_text("utf-8").splitlines()]
    assert written == records

    out2 = tmp_path / "two"
    out2.mkdir()
    again = voice_real.measure(
        utterances, FakeSpeaker(), FakeListener(), out=out2, cache_path=cache, **kwargs
    )
    assert all(r["cached"] for r in again)
    assert [{k: v for k, v in r.items() if k != "cached"} for r in again] == [
        {k: v for k, v in r.items() if k != "cached"} for r in records
    ]
    # 열화 매개변수가 다르면 키도 달라서 캐시에서 오는 것이 없다.
    out3 = tmp_path / "three"
    out3.mkdir()
    changed = dict(kwargs, description={"degrade": "test-2"})
    other = voice_real.measure(
        utterances, FakeSpeaker(), FakeListener(), out=out3, cache_path=cache, **changed
    )
    assert not any(r["cached"] for r in other)


# 정규화를 None으로 주면(V1 원본) 받아쓰기를 그대로 쓰는지 본다.
def test_a_v1_source_has_no_normaliser(tmp_path, source):
    out = tmp_path / "v1"
    out.mkdir()
    records = voice_real.measure(
        few(source, 1), FakeSpeaker(), FakeListener(), out=out, cache_path=None, decode=fake_decode,
        degrade=fake_degrade, description={}, normalizer=None,
    )  # fmt: skip
    assert records[0]["clean_text"] == records[0]["clean_heard"]


# 공식 실행은 작업 트리가 깨끗해야 하고, 발화 일부(--limit)나 다른 장치(--device cpu)로는 시작하지 않는지
# 본다. git 호출은 가짜로 바꿔 깨끗함/더러움을 흉내 낸다.
def test_the_official_run_needs_a_clean_tree_all_utterances_and_the_source_device(monkeypatch):
    monkeypatch.setattr(voice_real, "_git", lambda *args: "abc" if args[0] == "rev-parse" else " M src/x.py")
    with pytest.raises(SystemExit):
        voice_real.main(["run", str(SOURCE), "--official"])
    monkeypatch.setattr(voice_real, "_git", lambda *args: "abc" if args[0] == "rev-parse" else "")
    for extra in (["--limit", "3"], ["--device", "cpu"]):
        with pytest.raises(SystemExit):
            voice_real.main(["run", str(SOURCE), "--official", *extra])


# CPU 관문(plan_gate)이 발화 40개에 대해 먹먹함·잃은 프레임·돌발 소음을 세어 보고하고 통과 여부를 내는지 본다.
def test_the_plan_gate_counts_what_the_drawn_conditions_would_do(source):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    lines, passed = voice_real.plan_gate(few(source, 40))
    text = "\n".join(lines)
    assert "muffled" in text and "lost frames" in text and "bursts" in text
    assert isinstance(passed, bool)


# 관문과 재현 확인의 상수(먹먹함 15~25%, 손실 1~3%, 돌발 소음 ±3 표준편차, 재현 90%, GPU 사용 상한 6,000 MiB,
# 부트스트랩 seed 20261005)가 등록한 값인지, 돌발 소음 범위가 포아송 기댓값 ±3√기댓값인지 본다.
def test_the_plan_gate_bounds_are_the_registered_ones():
    assert voice_real.GATE_MUFFLED == (0.15, 0.25) and voice_real.GATE_LOSS == (0.01, 0.03)
    assert voice_real.GATE_BURST_SIGMAS == 3.0 and voice_real.REPRODUCE_MIN == 0.90
    assert voice_real.GPU_LIMIT_MIB == 6000 and voice_real.BOOTSTRAP_SEED == 20261005
    lo, hi = voice_real.burst_bounds(60 * 88.0)
    assert (lo, hi) == (88 - 3 * math.sqrt(88), 88 + 3 * math.sqrt(88))


# 결과 표: 기록에서 지표와 짝지은 구간을 다시 계산한다(모델 없음).
# ==================================================================== 보고서


# utterances.jsonl 한 줄을 손으로 만든다. recorded를 주면 9단계에 기록된 받아쓰기를 깨끗한 쪽과 다르게 둔다.
def record(task, said, clean, degraded, *, name="정예준", recorded=None, plan=None, trial=0, index=0):
    return {
        "task_id": task, "trial": trial, "index": index, "name": name, "said": said,
        "recorded_heard": clean if recorded is None else recorded,
        "recorded_text": normalize_heard(clean if recorded is None else recorded),
        "clean_heard": clean, "clean_text": normalize_heard(clean),
        "degraded_heard": degraded, "degraded_text": normalize_heard(degraded),
        "plan": plan or {"muffle_start_s": None, "colour": "pink", "snr_db": 15.0, "bursts": [], "frames": 10,
                         "lost": [], "gain": 1.0},
    }  # fmt: skip


# 이름 생존은 공백만 무시하고 글자까지 같아야 하는지(정예순은 아님), NFKC 정규화를 하는지 본다.
# verify_caller의 k=0 기준과 같다.
def test_name_survival_is_the_exact_name_with_spaces_ignored():
    assert voice_real.name_in("정예준", "네, 정 예준입니다.")
    assert not voice_real.name_in("정예준", "네, 정예순입니다.")
    assert voice_real.name_in("정예준", "ｊ정예준")  # NFKC


# 발화 하나의 지표별 (분자, 분모)를 깨끗한 쪽과 열화한 쪽으로 세는지 본다: 이름, 주문 번호(V2 정규화로
# 5-91001이 O-91001로 돌아옴), 그대로 전달, 글자 오류(두 글자), 인식 안 됨. 이름이 없는 발화는 이름 분모에
# 들지 않는다.
def test_the_counts_of_one_record():
    r = record(
        "dev-001",
        "정예준이고 주문은 O-91001이에요.",
        "정예준이고 주문은 5-91001이에요.",
        "정예순이고 주문은 5-9100이에요.",
    )
    counts = voice_real.counts(r)
    assert counts["name"] == ((1, 1), (0, 1))
    assert counts["id"] == ((1, 1), (0, 1))
    assert counts["exact"] == ((1, 1), (0, 1))
    assert counts["cer"][0] == (0, len("정예준이고주문은o-91001이에요".replace("-", "")))
    assert counts["cer"][1][0] == 2  # 준 대신 순, 숫자 하나 잃음
    assert counts["empty"] == ((0, 1), (0, 1))
    other = voice_real.counts(record("dev-001", "환불 금액 알려 주세요.", "환불 금액 알려 주세요.", ""))
    assert other["name"] == ((0, 0), (0, 0)) and other["empty"] == ((0, 1), (1, 1))


# 과제 단위로 다시 뽑는 부트스트랩이 같은 비율이면 (0, 0, 0)을, 나빠졌으면 합친 비율의 차이(−0.5)를 점추정으로
# 내고, 분모가 0인 과제는 영향이 없으며, 고정 seed라 같은 결과를 다시 내는지 본다.
def test_the_task_bootstrap_of_a_pooled_difference():
    same = {"a": ((1, 2), (1, 2)), "b": ((3, 4), (3, 4))}
    assert voice_real.pooled_difference(same) == (0.0, 0.0, 0.0)
    worse = {"a": ((2, 2), (1, 2)), "b": ((4, 4), (2, 4)), "c": ((0, 0), (0, 0))}
    diff, low, high = voice_real.pooled_difference(worse)
    assert diff == pytest.approx(-0.5) and low <= diff <= high
    assert voice_real.pooled_difference(worse) == (diff, low, high)  # 고정 seed


# 보고 표에 발화 수, 이름·주문 번호 생존, 재현 확인(기록과 같은 인식 2/3), 효과별 발화 수, 음량을 줄인 발화
# 수가 나오는지 본다.
def test_the_report_tables():
    plan_lost = {"muffle_start_s": 0.1, "colour": "brown", "snr_db": 13.0, "bursts": [{"kind": "ring"}],
                 "frames": 10, "lost": [3], "gain": 0.9}  # fmt: skip
    records = [
        record("dev-001", "정예준입니다.", "정예준입니다.", "정예순입니다."),
        record("dev-001", "O-91001이에요.", "5-91001이에요.", "5-91001이에요.", index=1, plan=plan_lost),
        record("dev-002", "네, 맞아요.", "네, 맞아요.", "네 맞아요", recorded="네, 맞아요?", trial=1),
    ]
    table = voice_real.report(records)
    assert "| 발화 | 3 |" in table
    assert "| 이름 (정확히, k = 0) | 1/1 (100.0%) | 0/1 (0.0%) |" in table
    assert "| 주문·접수 번호 | 1/1 (100.0%) | 1/1 (100.0%) |" in table
    assert "기록과 같은 인식: 2/3" in table
    assert (
        "프레임 손실 있음 | 1 |" in table and "먹먹함 있음 | 1 |" in table and "돌발 소음 있음 | 1 |" in table
    )
    assert "줄인 음량(gain < 1) 1" in table


# voice_real report 명령이 실행 폴더의 utterances.jsonl을 읽어 표를 출력하는지 본다.
def test_report_reads_a_run_directory(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    rows = [record("dev-001", "정예준입니다.", "정예준입니다.", "정예준입니다.")]
    (run / "utterances.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), "utf-8")
    voice_real.main(["report", str(run)])
    assert "| 발화 | 1 |" in capsys.readouterr().out


# docs/experiments.md의 11단계 절에 원본 실행 이름, 열화 버전, 등록한 값들, 부트스트랩 seed, 재현 기준 90%가
# 적혀 있는지 본다. 문서와 코드의 값이 따로 놀지 않게 한다.
def test_the_experiments_rules_name_this_source_and_these_values():
    doc = (Path(__file__).resolve().parents[1] / "docs" / "experiments.md").read_text(encoding="utf-8")
    stage = doc[doc.index("## 11단계") :]
    assert voice_real.SOURCE_RUN in stage and "v4r-1" in stage
    for value in ("15 dB", "±3 dB", "1500 Hz", "500 ms", "−5 ~ +10 dB", "2%", "100 ms", "20 ms", "0.99"):
        assert value in stage, value
    assert str(voice_real.BOOTSTRAP_SEED) in stage and "90%" in stage


# 인식기: 녹음 바이트 대신 디코딩한 표본을 받아도 같게 받아 적어야 한다.
# ==================================================================== 인식기는 디코딩한 표본도 받는다


# WhisperListener.transcribe(녹음)와 transcribe_samples(decode_audio로 디코딩한 표본)가 모델에 같은 표본과
# 같은 옵션(한국어, beam 5, temperature 0, 이전 글 조건 없음, VAD 끔)을 넘기는지 본다. 11단계는 디코딩한
# 표본에 열화를 얹고 transcribe_samples로 받아 적으므로, 깨끗한 쪽이 9단계 채널과 같은 경로여야 한다.
def test_the_listener_hears_decoded_samples_the_way_it_hears_a_recording():
    np = pytest.importorskip("numpy")
    decode_audio = pytest.importorskip("faster_whisper.audio").decode_audio
    from support_agent.voice.speech import WhisperListener, wav_bytes

    class Recording:
        def __init__(self):
            self.calls = []

        def transcribe(self, samples, **options):
            self.calls.append((np.asarray(samples).copy(), options))
            return [type("S", (), {"text": " 네 "})()], None

    listener = object.__new__(WhisperListener)
    listener.max_seconds, listener.beam_size, listener.vad_filter = None, 5, False
    listener._model = Recording()
    wav = wav_bytes(0.3 * np.sin(np.arange(44_100) / 7), 44_100).wav
    assert listener.transcribe(wav) == "네"
    samples = decode_audio(io.BytesIO(wav), sampling_rate=16_000)
    assert listener.transcribe_samples(samples) == "네"
    (first, options1), (second, options2) = listener._model.calls
    assert np.array_equal(first, second) and options1 == options2
    assert options1 == {"language": "ko", "beam_size": 5, "temperature": 0.0,
                        "condition_on_previous_text": False, "vad_filter": False}  # fmt: skip
