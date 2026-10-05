"""Stage 11 (V4-R, stage A): the customer utterances of the stage 9 V4 development run, synthesised again with
the same seeds and recognised twice, clean and through telephone-call conditions (docs/experiments.md).

The signal tests need numpy and scipy (the `voice` dependency group) and are skipped without them. The runner
and the report are tested with fake speech models and hand-made records, so they run anywhere.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pytest

from support_agent import voice_real
from support_agent.config import derive_seed
from support_agent.paths import REPORTS
from support_agent.voice.normalize import normalize_heard
from support_agent.voice.speech import Audio

SOURCE = REPORTS / voice_real.SOURCE_RUN


# ==================================================================== the degradation (voice/degrade.py)


@pytest.fixture(scope="module")
def dg():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from support_agent.voice import degrade

    return degrade


@pytest.fixture(scope="module")
def np():
    return pytest.importorskip("numpy")


def tone(np, frequency, seconds=1.0, amplitude=0.5):
    t = np.arange(int(16_000 * seconds)) / 16_000
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def band_energy(np, samples, low, high):
    spectrum = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1 / 16_000)
    return float(spectrum[(freqs >= low) & (freqs < high)].sum())


# --- the telephone band and mu-law: the implementation of whisper-ko-ft's channel.py, fixed by its values


def test_telephone_gives_what_whisper_ko_fts_channel_gives(dg, np):
    """Reference values computed with whisper-ko-ft src/whisper_ko_ft/channel.py (commit 51616d5's code,
    numpy 2.4.6, scipy 1.17.1) on the same input: four tones, two of them outside the band."""
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


def test_telephone_is_the_narrow_band_widened_again(dg, np):
    x = np.random.default_rng(3).normal(0, 0.1, 12_345).astype(np.float32)
    assert np.array_equal(dg.telephone(x), dg.widen(dg.narrowband(x), len(x)))
    assert len(dg.narrowband(x)) == 6173  # 8 kHz


def test_telephone_keeps_length_and_dtype(dg, np):
    out = dg.telephone(tone(np, 1000, 0.73))
    assert out.dtype == np.float32 and len(out) == int(16_000 * 0.73)


def test_a_speech_band_tone_survives_the_telephone(dg, np):
    original = tone(np, 1000)
    assert np.corrcoef(original, dg.telephone(original))[0, 1] > 0.98


def test_tones_outside_the_band_are_removed(dg, np):
    for frequency in (100, 6000):
        out = dg.telephone(tone(np, frequency))
        assert band_energy(np, out, 0, 8000) < 0.02 * band_energy(np, tone(np, frequency), 0, 8000)


def test_nothing_is_left_above_4khz(dg, np):
    out = dg.telephone(np.random.default_rng(0).normal(0, 0.1, 16_000).astype(np.float32))
    assert band_energy(np, out, 4200, 8000) < 0.001 * band_energy(np, out, 300, 3400)


def test_mu_law_uses_at_most_256_levels_and_keeps_quiet_samples(dg, np):
    restored = dg.mu_law_roundtrip(np.linspace(-1, 1, 100_000).astype(np.float32))
    assert len(np.unique(restored)) <= 256
    assert abs(dg.mu_law_roundtrip(np.array([np.float32(0.01)]))[0] - 0.01) < 0.002


def test_very_short_input_is_returned_unchanged(dg, np):
    short = np.zeros(10, dtype=np.float32)
    assert np.array_equal(dg.telephone(short), short)
    out, info = dg.degrade(short, seed=1)
    assert np.array_equal(out, short) and info["gain"] == 1.0


# --- the conditions: tau-Voice "Realistic" values, fixed before measuring


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


def test_gilbert_elliott_loses_two_percent_in_bursts_of_100_ms(dg, np):
    lost = dg.gilbert_elliott(300_000, 0.02, 5, np.random.default_rng(7))
    assert lost.mean() == pytest.approx(0.02, abs=0.003)
    starts = np.flatnonzero(lost & ~np.concatenate(([False], lost[:-1])))
    assert lost.sum() / len(starts) == pytest.approx(5.0, abs=0.4)  # mean run of lost 20 ms frames


def test_the_plan_is_fixed_by_the_seed(dg):
    a, b, c = dg.draw(11, 80_000), dg.draw(11, 80_000), dg.draw(12, 80_000)
    assert a == b and a != c
    assert json.loads(json.dumps(a.to_dict())) == a.to_dict()
    assert a.frames == math.ceil(40_000 / 160)


def test_the_plans_of_many_utterances_follow_the_registered_rates(dg, np):
    plans = [dg.draw(seed, 5 * 16_000) for seed in range(3000)]
    assert sum(p.muffle_start is not None for p in plans) / 3000 == pytest.approx(0.2, abs=0.03)
    snrs = [p.snr_db for p in plans]
    assert 12.0 <= min(snrs) and max(snrs) <= 18.0 and sum(snrs) / 3000 == pytest.approx(15.0, abs=0.3)
    assert sum(p.colour == "pink" for p in plans) / 3000 == pytest.approx(0.5, abs=0.05)
    bursts = [b for p in plans for b in p.bursts]
    assert len(bursts) / 3000 == pytest.approx(5 / 60, abs=0.015)  # one a minute
    assert {b[0] for b in bursts} == set(dg.REALISTIC.burst_kinds)
    assert all(-5.0 <= b[2] <= 10.0 and 0 <= b[1] < 80_000 for b in bursts)
    frames = sum(p.frames for p in plans)
    assert sum(len(p.lost) for p in plans) / frames == pytest.approx(0.02, abs=0.004)


def band_rms(dg, np, x):
    from scipy.signal import sosfiltfilt

    return float(np.sqrt(np.mean(sosfiltfilt(dg.BAND, np.asarray(x, dtype=np.float64)) ** 2)))


def quiet_plan(dg, length, **changes):
    """A plan with every effect off: noise 300 dB under the speech, nothing muffled, lost or burst."""
    base = dict(muffle_start=None, colour="pink", snr_db=300.0, noise_seed=1, bursts=(), frames=0, lost=())
    base.update(changes)
    base["frames"] = dg.frame_count(length)
    return dg.Plan(**base)


def test_the_background_noise_is_at_the_drawn_snr_inside_the_telephone_band(dg, np):
    speech = tone(np, 700, 3.0, 0.2) + tone(np, 2200, 3.0, 0.05)
    for colour in ("pink", "brown"):
        plan = quiet_plan(dg, len(speech), colour=colour, snr_db=13.5, noise_seed=4)
        mixed, gain = dg.mix(speech, plan)
        assert gain == 1.0
        snr = 20 * math.log10(band_rms(dg, np, speech) / band_rms(dg, np, mixed - speech))
        assert snr == pytest.approx(13.5, abs=0.05), colour


def test_a_burst_is_at_its_snr_and_place(dg, np):
    speech = tone(np, 700, 3.0, 0.2)
    plan = quiet_plan(dg, len(speech), bursts=(("ring", 16_000, -2.0, 9),))
    mixed, _gain = dg.mix(speech, plan)
    added = mixed - speech
    ring = int(16_000 * dg.BURST_SECONDS["ring"])
    assert np.abs(added[:16_000]).max() < 1e-6 and np.abs(added[16_000 + ring :]).max() < 1e-6
    burst_rms = band_rms(dg, np, added[16_000 : 16_000 + ring])
    assert 20 * math.log10(band_rms(dg, np, speech) / burst_rms) == pytest.approx(-2.0, abs=0.1)


def test_a_burst_near_the_end_is_cut_there(dg, np):
    speech = tone(np, 700, 1.0, 0.2)
    plan = quiet_plan(dg, len(speech), bursts=(("siren", 15_000, 5.0, 2),))
    mixed, _gain = dg.mix(speech, plan)
    assert len(mixed) == 16_000 and np.abs((mixed - speech)[15_000:]).max() > 0


@pytest.mark.parametrize("kind", ["clatter", "ring", "horn", "siren"])
def test_every_burst_kind_is_sound_in_the_telephone_band(dg, np, kind):
    sound = dg.burst_sound(kind, np.random.default_rng(0))
    assert len(sound) == int(16_000 * dg.BURST_SECONDS[kind]) and np.isfinite(sound).all()
    assert band_rms(dg, np, sound) > 0.1 * float(np.sqrt(np.mean(sound**2)))
    assert sound[0] == 0 and abs(sound[-1]) < 1e-3  # faded in and out: no click


def test_muffling_lowers_high_frequencies_only_in_its_segment(dg, np):
    speech = tone(np, 3000, 3.0, 0.2)
    plan = quiet_plan(dg, len(speech), muffle_start=8000)
    mixed, _gain = dg.mix(speech, plan)
    plateau = slice(8000 + 1600 + 200, 8000 + 1600 + 8000 - 200)  # inside the 500 ms, away from the ramps
    assert np.abs(mixed[plateau]).max() < 0.02 * 0.2
    span = 8000 + 2 * 1600 + 8000
    assert np.abs(mixed[:8000] - speech[:8000]).max() < 1e-4
    assert np.abs(mixed[span:] - speech[span:]).max() < 1e-4


def test_lost_frames_are_silence_after_the_channel(dg, np):
    speech = tone(np, 700, 1.0, 0.3)
    plan = quiet_plan(dg, len(speech), lost=(10, 11))
    out = dg.apply(speech, plan)[0]
    gap = out[10 * 320 + 40 : 12 * 320 - 40]  # 16 kHz samples of two 20 ms frames, without the filter's edge
    assert np.abs(gap).max() < 1e-3
    assert np.abs(out[5 * 320 : 9 * 320]).max() > 0.2 and np.abs(out[13 * 320 : 20 * 320]).max() > 0.2


def test_a_loud_mixture_is_turned_down_before_the_codec_instead_of_clipped(dg, np):
    speech = tone(np, 700, 2.0, 0.95)
    plan = quiet_plan(dg, len(speech), bursts=(("horn", 4000, -5.0, 1),))
    mixed, gain = dg.mix(speech, plan)
    assert gain < 1.0 and np.abs(mixed).max() == pytest.approx(0.99, abs=1e-6)


def test_degrade_is_fixed_by_its_seed_and_returns_16k_float32(dg, np):
    speech = tone(np, 700, 4.0, 0.2) + tone(np, 1900, 4.0, 0.05)
    a, info_a = dg.degrade(speech, seed=21)
    b, info_b = dg.degrade(speech, seed=21)
    c, _ = dg.degrade(speech, seed=22)
    assert a.dtype == np.float32 and len(a) == len(speech)
    assert np.array_equal(a, b) and info_a == info_b and not np.array_equal(a, c)
    assert info_a == dg.draw(21, len(speech)).to_dict() | {"gain": info_a["gain"]}


# ==================================================================== the source utterances


@pytest.fixture(scope="module")
def source():
    return voice_real.source_utterances(SOURCE)


def test_the_source_is_the_stage_9_v4_development_run(source):
    manifest, utterances = source
    assert manifest["config"]["voice"] == "V4" and manifest["tasks_file"] == "dev" and manifest["official"]
    assert len(utterances) == 929 and len({(u.task_id, u.trial) for u in utterances}) == 96
    assert len({u.task_id for u in utterances}) == 24


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


def test_each_utterance_knows_the_registered_name_of_its_customer(source):
    from support_agent.analyze import seed_names
    from support_agent.tasks import load_tasks

    _manifest, utterances = source
    names, tasks = seed_names(), {t.id: t for t in load_tasks("dev")}
    assert all(u.name == names[tasks[u.task_id].customer_id] for u in utterances)
    assert sum(voice_real.name_in(u.name, u.said) for u in utterances) == 165


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


def test_a_source_whose_task_file_changed_is_refused(tmp_path):
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    manifest["task_sha256"]["dev-001"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "episodes.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="dev-001"):
        voice_real.source_utterances(tmp_path)


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


# ==================================================================== the runner, with fake speech models


class FakeSpeaker:
    def __init__(self):
        self.calls = []

    def describe(self):
        return {"tts": "fake", "device": "cpu"}

    def synthesize(self, spoken_text, seed=0):
        self.calls.append((spoken_text, seed))
        return Audio(spoken_text.encode("utf-8"), 0.5 + len(spoken_text) / 10)


class FakeListener:
    """Hears the clean audio as it was said, and loses the last character through the degradation."""

    def describe(self):
        return {"stt": "fake", "device": "cpu"}

    def transcribe_samples(self, samples):
        text = samples["text"]
        return text[:-1] if samples.get("degraded") else text


def fake_decode(wav):
    return {"text": wav.decode("utf-8")}


def fake_degrade(samples, seed):
    return dict(samples, degraded=True), {"muffle_start_s": None, "lost": [seed % 3], "gain": 1.0}


def few(source, n=6):
    return source[1][:n]


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
    # Other degradation parameters are another key: nothing comes from the cache.
    out3 = tmp_path / "three"
    out3.mkdir()
    changed = dict(kwargs, description={"degrade": "test-2"})
    other = voice_real.measure(
        utterances, FakeSpeaker(), FakeListener(), out=out3, cache_path=cache, **changed
    )
    assert not any(r["cached"] for r in other)


def test_a_v1_source_has_no_normaliser(tmp_path, source):
    out = tmp_path / "v1"
    out.mkdir()
    records = voice_real.measure(
        few(source, 1), FakeSpeaker(), FakeListener(), out=out, cache_path=None, decode=fake_decode,
        degrade=fake_degrade, description={}, normalizer=None,
    )  # fmt: skip
    assert records[0]["clean_text"] == records[0]["clean_heard"]


def test_the_official_run_needs_a_clean_tree_all_utterances_and_the_source_device(monkeypatch):
    monkeypatch.setattr(voice_real, "_git", lambda *args: "abc" if args[0] == "rev-parse" else " M src/x.py")
    with pytest.raises(SystemExit):
        voice_real.main(["run", str(SOURCE), "--official"])
    monkeypatch.setattr(voice_real, "_git", lambda *args: "abc" if args[0] == "rev-parse" else "")
    for extra in (["--limit", "3"], ["--device", "cpu"]):
        with pytest.raises(SystemExit):
            voice_real.main(["run", str(SOURCE), "--official", *extra])


def test_the_plan_gate_counts_what_the_drawn_conditions_would_do(source):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    lines, passed = voice_real.plan_gate(few(source, 40))
    text = "\n".join(lines)
    assert "muffled" in text and "lost frames" in text and "bursts" in text
    assert isinstance(passed, bool)


def test_the_plan_gate_bounds_are_the_registered_ones():
    assert voice_real.GATE_MUFFLED == (0.15, 0.25) and voice_real.GATE_LOSS == (0.01, 0.03)
    assert voice_real.GATE_BURST_SIGMAS == 3.0 and voice_real.REPRODUCE_MIN == 0.90
    assert voice_real.GPU_LIMIT_MIB == 6000 and voice_real.BOOTSTRAP_SEED == 20261005
    lo, hi = voice_real.burst_bounds(60 * 88.0)
    assert (lo, hi) == (88 - 3 * math.sqrt(88), 88 + 3 * math.sqrt(88))


# ==================================================================== the report


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


def test_name_survival_is_the_exact_name_with_spaces_ignored():
    assert voice_real.name_in("정예준", "네, 정 예준입니다.")
    assert not voice_real.name_in("정예준", "네, 정예순입니다.")
    assert voice_real.name_in("정예준", "ｊ정예준")  # NFKC


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
    assert counts["cer"][1][0] == 2  # 순 for 준, a lost digit
    assert counts["empty"] == ((0, 1), (0, 1))
    other = voice_real.counts(record("dev-001", "환불 금액 알려 주세요.", "환불 금액 알려 주세요.", ""))
    assert other["name"] == ((0, 0), (0, 0)) and other["empty"] == ((0, 1), (1, 1))


def test_the_task_bootstrap_of_a_pooled_difference():
    same = {"a": ((1, 2), (1, 2)), "b": ((3, 4), (3, 4))}
    assert voice_real.pooled_difference(same) == (0.0, 0.0, 0.0)
    worse = {"a": ((2, 2), (1, 2)), "b": ((4, 4), (2, 4)), "c": ((0, 0), (0, 0))}
    diff, low, high = voice_real.pooled_difference(worse)
    assert diff == pytest.approx(-0.5) and low <= diff <= high
    assert voice_real.pooled_difference(worse) == (diff, low, high)  # fixed seed


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


def test_report_reads_a_run_directory(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    rows = [record("dev-001", "정예준입니다.", "정예준입니다.", "정예준입니다.")]
    (run / "utterances.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), "utf-8")
    voice_real.main(["report", str(run)])
    assert "| 발화 | 1 |" in capsys.readouterr().out


def test_the_experiments_rules_name_this_source_and_these_values():
    doc = (Path(__file__).resolve().parents[1] / "docs" / "experiments.md").read_text(encoding="utf-8")
    stage = doc[doc.index("## 11단계") :]
    assert voice_real.SOURCE_RUN in stage and "v4r-1" in stage
    for value in ("15 dB", "±3 dB", "1500 Hz", "500 ms", "−5 ~ +10 dB", "2%", "100 ms", "20 ms", "0.99"):
        assert value in stage, value
    assert str(voice_real.BOOTSTRAP_SEED) in stage and "90%" in stage


# ==================================================================== the listener takes decoded samples


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
