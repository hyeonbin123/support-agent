"""Stage 11 (V4-R, stage A): what telephone-call conditions do to the recogniser's transcripts of the customer
utterances of the stage 9 V4 development run. No LLM, no verdict (docs/experiments.md, stage 11).

Each utterance is synthesised again with the seed the channel used (same spoken text, same MeloTTS seed),
decoded to 16 kHz the way the listener decodes a recording, and recognised twice by the same model: as it is
(clean), and after voice/degrade.py with a seed of its own. Both transcripts go through the run's normaliser.

    uv run --no-sync python -m support_agent.voice_real plan reports/<V4 dev run>     (CPU: drawn conditions)
    uv run --no-sync python -m support_agent.voice_real run reports/<V4 dev run> --official --label v4r-a
    uv run --no-sync python -m support_agent.voice_real report reports/<stage A run>

`run` needs the `voice` dependency group. It writes to outputs/v4r/<run_id>/, or to reports/ with --official
(clean working tree, every utterance, the source run's speech models). Round trips are cached in
outputs/voice-cache/v4r-roundtrips.jsonl under a key holding the speech models, the degradation's version and
values, both seeds and the text, so a run that stopped can be started again.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import re
import sys
import time
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from support_agent.config import derive_seed
from support_agent.paths import OUTPUTS, REPORTS
from support_agent.run import _git, gpu_memory_used_mib, safe_name
from support_agent.voice.metrics import ENTITY_LABELS, edit_distance, entities, squeeze, survived
from support_agent.voice.normalize import normalize_heard
from support_agent.voice.speech import timed

STAGE = "v4r-a"
SOURCE_RUN = "20261004-154016-qwen3.5-4b-P0-R0-G0-F0-V4-v4-k0"  # stage 9, V4, development tasks
CACHE = OUTPUTS / "voice-cache" / "v4r-roundtrips.jsonl"
# Registered before measuring (docs/experiments.md, stage 11).
REPRODUCE_MIN = 0.90  # share of clean transcripts equal to the recorded ones, to tie the numbers to stage 9
GATE_MUFFLED = (0.15, 0.25)  # CPU gate on the drawn conditions: share of muffled utterances
GATE_LOSS = (0.01, 0.03)  # share of lost frames
GATE_BURST_SIGMAS = 3.0  # bursts within the Poisson mean +- 3 standard deviations
# The two speech models took about 3.1 GB in stage 9; above this much memory in use by anything else, the
# 11 GB card would have under 2 GB to spare and could spill into shared memory.
GPU_LIMIT_MIB = 6000
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 20261005
_SAME_CHANNEL_KEYS = (
    ("tts", "tts"),
    ("speed", "tts"),
    ("tts_versions", "tts"),
    ("stt", "stt"),
    ("compute_type", "stt"),
    ("beam_size", "stt"),
    ("vad_filter", "stt"),
    ("stt_versions", "stt"),
)
_KEEP = object()  # the default normaliser: normalize_heard (None means none)


@dataclass(frozen=True)
class Utterance:
    task_id: str
    trial: int
    index: int  # position in the episode's voice log: the channel's seed index
    seed: int  # the channel's synthesis seed
    degrade_seed: int
    name: str  # the registered name of the task's customer
    said: str
    spoken: str
    recorded_heard: str
    recorded_text: str
    recorded_audio_seconds: float


def name_in(name: str, text: str) -> bool:
    """The registered name, letter for letter, with spaces ignored: what verify_caller (k = 0) accepts."""
    squeezed = "".join(unicodedata.normalize("NFKC", text).split())
    return "".join(unicodedata.normalize("NFKC", name).split()) in squeezed


def source_utterances(run_dir: Path) -> tuple[dict[str, Any], list[Utterance]]:
    """The synthesised customer utterances of a voice run, in the order of the run's records."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from support_agent import db
    from support_agent.seed import build_seed_engine
    from support_agent.tasks import load_tasks

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if "test" in Path(str(manifest.get("tasks_file", ""))).stem:
        raise ValueError("stage A uses development records only: test records are not opened")
    if config.get("voice", "V0") == "V0" or not manifest.get("voice_channel"):
        raise ValueError("not a voice run: nothing was synthesised")
    tasks = {t.id: t for t in load_tasks(manifest["tasks_file"])}
    changed = [i for i, sha in manifest["task_sha256"].items() if i not in tasks or tasks[i].sha256() != sha]
    if changed:
        raise ValueError(f"the task file changed since the run: {', '.join(changed)}")
    engine = build_seed_engine()
    if db.state_hash(db.dump_db(engine)) != manifest["seed_hash"]:
        raise ValueError("the seed database changed since the run")
    with Session(engine) as session:
        names = {c.id: c.name for c in session.scalars(select(db.Customer))}

    base_seed = config["base_seed"]
    episodes = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text("utf-8").splitlines()]
    out = []
    for e in sorted(episodes, key=lambda e: (e["task_id"], e["trial"])):
        task_id, trial = e["task_id"], e["trial"]
        for index, u in enumerate(e.get("voice") or []):
            if not re.search("[가-힣]", u["spoken"]):  # the channel synthesised nothing
                continue
            out.append(
                Utterance(
                    task_id=task_id,
                    trial=trial,
                    index=index,
                    seed=derive_seed(base_seed, task_id, trial, "voice", index),
                    degrade_seed=derive_seed(base_seed, task_id, trial, "degrade", index),
                    name=names[tasks[task_id].customer_id],
                    said=u["said"],
                    spoken=u["spoken"],
                    recorded_heard=u["heard"],
                    recorded_text=u["text"],
                    recorded_audio_seconds=u["audio_seconds"],
                )
            )
    return manifest, out


def channel_mismatches(manifest: dict[str, Any], tts: dict[str, Any], stt: dict[str, Any]) -> list[str]:
    """What differs from the source run's speech models. Its manifest merged both descriptions into one, so
    its `device` is the recogniser's; stage 9 ran both models on the same device."""
    recorded = manifest["voice_channel"]
    sides = {"tts": tts, "stt": stt}
    out = [key for key, side in _SAME_CHANNEL_KEYS if recorded.get(key) != sides[side].get(key)]
    out += [
        f"{side} device" for side in ("tts", "stt") if sides[side].get("device") != recorded.get("device")
    ]
    return out


# ---------------------------------------------------------------- the measurement


def measure(
    utterances: list[Utterance],
    speaker: Any,
    listener: Any,
    *,
    decode: Callable[[bytes], Any],
    degrade: Callable[[Any, int], tuple[Any, dict[str, Any]]],
    description: dict[str, Any],
    cache_path: Path | None,
    out: Path,
    normalizer: Any = _KEEP,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Synthesise, decode, recognise clean, degrade, recognise again; one record per utterance."""
    normalise = normalize_heard if normalizer is _KEEP else (normalizer or (lambda text: text))
    identity = json.dumps([speaker.describe(), listener.describe(), description], sort_keys=True)
    cache: dict[str, dict[str, Any]] = {}
    if cache_path and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            cache[entry["key"]] = entry
    records, started = [], time.perf_counter()
    with (out / "utterances.jsonl").open("w", encoding="utf-8", newline="\n") as written:
        for n, u in enumerate(utterances, 1):
            key = hashlib.sha256(f"{identity}|{u.seed}|{u.degrade_seed}|{u.spoken}".encode()).hexdigest()
            entry = cache.get(key)
            cached = entry is not None
            if entry is None:
                audio, tts_ms = timed(speaker.synthesize, u.spoken, u.seed)
                samples = decode(audio.wav)
                clean, clean_ms = timed(listener.transcribe_samples, samples)
                degraded, plan = degrade(samples, u.degrade_seed)
                heard, degraded_ms = timed(listener.transcribe_samples, degraded)
                entry = {
                    "key": key,
                    "clean_heard": clean,
                    "degraded_heard": heard,
                    "plan": plan,
                    "audio_seconds": round(audio.seconds, 3),
                    "tts_ms": round(tts_ms, 1),
                    "clean_stt_ms": round(clean_ms, 1),
                    "degraded_stt_ms": round(degraded_ms, 1),
                }
                cache[key] = entry
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with cache_path.open("a", encoding="utf-8", newline="\n") as kept:
                        kept.write(json.dumps(entry, ensure_ascii=False) + "\n")
            record = asdict(u) | {k: v for k, v in entry.items() if k != "key"}
            record |= {
                "clean_text": normalise(entry["clean_heard"]),
                "degraded_text": normalise(entry["degraded_heard"]),
                "cached": cached,
            }
            records.append(record)
            written.write(json.dumps(record, ensure_ascii=False) + "\n")
            written.flush()
            if progress and (n % 25 == 0 or n == len(utterances)):
                done = sum(r["cached"] for r in records)
                progress(
                    f"{n}/{len(utterances)} utterances ({done} cached), {time.perf_counter() - started:.0f} s"
                )
    return records


# ---------------------------------------------------------------- the CPU gate on the drawn conditions


def burst_bounds(seconds: float, per_minute: float = 1.0) -> tuple[float, float]:
    expected = seconds / 60.0 * per_minute
    spread = GATE_BURST_SIGMAS * math.sqrt(expected)
    return expected - spread, expected + spread


def plan_gate(utterances: list[Utterance]) -> tuple[list[str], bool]:
    """Draw every utterance's conditions for its recorded length and check them against the registered rates.
    It checks the implementation at the size of the data; it measures nothing."""
    from support_agent.voice.degrade import REALISTIC, SAMPLE_RATE, draw

    p = REALISTIC
    plans = [draw(u.degrade_seed, round(u.recorded_audio_seconds * SAMPLE_RATE)) for u in utterances]
    seconds = sum(u.recorded_audio_seconds for u in utterances)
    muffled = sum(plan.muffle_start is not None for plan in plans) / len(plans)
    frames, lost = sum(plan.frames for plan in plans), sum(len(plan.lost) for plan in plans)
    bursts = sum(len(plan.bursts) for plan in plans)
    low, high = burst_bounds(seconds, p.burst_per_minute)
    snrs = [plan.snr_db for plan in plans]
    colours = {c: sum(plan.colour == c for plan in plans) for c in p.noise_colours}
    in_range = p.noise_snr_db - p.noise_snr_spread_db <= min(snrs) and max(snrs) <= (
        p.noise_snr_db + p.noise_snr_spread_db
    )
    checks = [
        (
            f"muffled {muffled:.1%} of {len(plans)} utterances, registered {GATE_MUFFLED[0]:.0%}"
            f"-{GATE_MUFFLED[1]:.0%}",
            GATE_MUFFLED[0] <= muffled <= GATE_MUFFLED[1],
        ),
        (
            f"lost frames {lost}/{frames} ({lost / frames:.2%}), registered "
            f"{GATE_LOSS[0]:.0%}-{GATE_LOSS[1]:.0%}",
            GATE_LOSS[0] <= lost / frames <= GATE_LOSS[1],
        ),
        (
            f"bursts {bursts} in {seconds / 60:.1f} min of audio, registered {low:.1f}-{high:.1f}",
            low <= bursts <= high,
        ),
        (
            f"background SNR {min(snrs):.2f}-{max(snrs):.2f} dB (mean {mean(snrs):.2f}), colours "
            + ", ".join(f"{c} {n}" for c, n in colours.items()),
            in_range and all(colours.values()),
        ),
    ]
    lines = [f"{text}: {'ok' if ok else 'OUT'}" for text, ok in checks]
    passed = all(ok for _text, ok in checks)
    lines.append("gate: " + ("PASS" if passed else "FAIL"))
    return lines, passed


# ---------------------------------------------------------------- the report


def _side(r: dict[str, Any], text: str) -> dict[str, tuple[int, int]]:
    said = r["said"]
    out = {
        "cer": (edit_distance(squeeze(said), squeeze(text)), len(squeeze(said))),
        "exact": (int(squeeze(said) == squeeze(text)), 1),
        "empty": (int(not text), 1),
        "name": (int(name_in(r["name"], text)), 1) if name_in(r["name"], said) else (0, 0),
    }
    found = entities(said)
    for kind in ENTITY_LABELS:
        mine = [entity for k, entity in found if k == kind]
        out[kind] = (sum(survived(kind, entity, text) for entity in mine), len(mine))
    return out


def counts(r: dict[str, Any]) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    """Per metric: ((clean numerator, denominator), (degraded numerator, denominator))."""
    clean, degraded = _side(r, r["clean_text"]), _side(r, r["degraded_text"])
    return {key: (clean[key], degraded[key]) for key in clean}


def pooled_difference(
    by_task: dict[str, tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[float, float, float]:
    """Degraded minus clean of the pooled rate, with a 95% interval: tasks are resampled with their utterances
    (both sides of an utterance stay together)."""

    def rate(tasks: list[str]) -> float | None:
        cn = sum(by_task[t][0][0] for t in tasks)
        cd = sum(by_task[t][0][1] for t in tasks)
        dn = sum(by_task[t][1][0] for t in tasks)
        dd = sum(by_task[t][1][1] for t in tasks)
        return dn / dd - cn / cd if cd and dd else None

    tasks = sorted(by_task)
    point = rate(tasks)
    if point is None:
        return math.nan, math.nan, math.nan
    rng = random.Random(BOOTSTRAP_SEED)
    rounds = sorted(
        d
        for d in (rate([rng.choice(tasks) for _ in tasks]) for _ in range(BOOTSTRAP_ROUNDS))
        if d is not None
    )
    return point, rounds[int(0.025 * len(rounds))], rounds[int(0.975 * len(rounds)) - 1]


_ROWS = (
    ("cer", "글자 오류율"),
    ("exact", "그대로 전달"),
    ("empty", "인식 안 됨"),
    ("name", "이름 (정확히, k = 0)"),
    *((kind, label) for kind, label in ENTITY_LABELS.items() if kind == "id"),
    *((kind, label) for kind, label in ENTITY_LABELS.items() if kind != "id"),
)


def _cell(key: str, n: int, d: int) -> str:
    if not d:
        return "-"
    return f"{n / d:.2%}" if key == "cer" else f"{n}/{d} ({n / d:.1%})"


def _cer(records: list[dict[str, Any]], field: str) -> float:
    errors = sum(edit_distance(squeeze(r["said"]), squeeze(r[field])) for r in records)
    length = sum(len(squeeze(r["said"])) for r in records)
    return errors / length if length else math.nan


def report(records: list[dict[str, Any]]) -> str:
    per_record = [(r, counts(r)) for r in records]
    tasks = sorted({r["task_id"] for r in records})
    lines = ["| 항목 | 값 |", "|---|---|", f"| 발화 | {len(records)} |", f"| 과제 | {len(tasks)} |", ""]

    same = sum(r["clean_heard"] == r["recorded_heard"] for r in records)
    share = same / len(records) if records else math.nan
    lines.append(
        f"재현 확인: 기록과 같은 인식: {same}/{len(records)} ({share:.1%}), 기준 {REPRODUCE_MIN:.0%} 이상 → "
        f"{'충족' if share >= REPRODUCE_MIN else '미달'}. 글자 오류율은 기록된 인식 "
        f"{_cer(records, 'recorded_text'):.2%}, "
        f"다시 합성한 깨끗한 음성 {_cer(records, 'clean_text'):.2%}"
    )
    lines += [
        "",
        f"| 지표 | 깨끗한 음성 | 열화한 음성 | 차이 [95% 구간, 과제 {len(tasks)}개 부트스트랩] |",
        "|---|---|---|---|",
    ]
    for key, label in _ROWS:
        by_task: dict[str, list[list[int]]] = {t: [[0, 0], [0, 0]] for t in tasks}
        for r, c in per_record:
            for side in (0, 1):
                by_task[r["task_id"]][side][0] += c[key][side][0]
                by_task[r["task_id"]][side][1] += c[key][side][1]
        frozen = {t: ((v[0][0], v[0][1]), (v[1][0], v[1][1])) for t, v in by_task.items()}
        cn, cd = sum(v[0][0] for v in frozen.values()), sum(v[0][1] for v in frozen.values())
        dn, dd = sum(v[1][0] for v in frozen.values()), sum(v[1][1] for v in frozen.values())
        diff, low, high = pooled_difference(frozen)
        digits = 2 if key == "cer" else 1
        gap = (
            "-"
            if math.isnan(diff)
            else f"{diff * 100:+.{digits}f}%p [{low * 100:+.{digits}f}, {high * 100:+.{digits}f}]"
        )
        lines.append(f"| {label} | {_cell(key, cn, cd)} | {_cell(key, dn, dd)} | {gap} |")

    plans = [r["plan"] for r in records]
    frames = sum(p["frames"] for p in plans)
    lost = sum(len(p["lost"]) for p in plans)
    muffled = sum(p["muffle_start_s"] is not None for p in plans)
    snrs = [p["snr_db"] for p in plans]
    colours = {c: sum(p["colour"] == c for p in plans) for c in ("pink", "brown")}
    if plans:
        lines += [
            "",
            f"실제로 들어간 조건: 먹먹함 {muffled}/{len(plans)} ({muffled / len(plans):.1%}), "
            f"손실 프레임 {lost}/{frames} ({lost / frames if frames else 0:.2%}), "
            f"돌발 소음 {sum(len(p['bursts']) for p in plans)}개, 배경 소음 SNR "
            f"평균 {mean(snrs):.1f} dB [{min(snrs):.1f}, {max(snrs):.1f}], 분홍 {colours['pink']} / 갈색 "
            f"{colours['brown']}, 줄인 음량(gain < 1) {sum(p.get('gain', 1.0) < 1.0 for p in plans)}",
        ]
    groups = (
        ("프레임 손실 있음", lambda p: bool(p["lost"])),
        ("프레임 손실 없음", lambda p: not p["lost"]),
        ("먹먹함 있음", lambda p: p["muffle_start_s"] is not None),
        ("먹먹함 없음", lambda p: p["muffle_start_s"] is None),
        ("돌발 소음 있음", lambda p: bool(p["bursts"])),
        ("돌발 소음 없음", lambda p: not p["bursts"]),
        ("분홍 잡음", lambda p: p["colour"] == "pink"),
        ("갈색 잡음", lambda p: p["colour"] == "brown"),
    )
    lines += [
        "",
        "| 조건별 (기술용, 구간 없음) | 발화 | 깨끗한 글자 오류율 | 열화한 글자 오류율 |",
        "|---|---|---|---|",
    ]
    for label, chosen in groups:
        mine = [r for r in records if chosen(r["plan"])]
        if mine:
            lines.append(
                f"| {label} | {len(mine)} | {_cer(mine, 'clean_text'):.2%} "
                f"| {_cer(mine, 'degraded_text'):.2%} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------- command line


def _load_models(device: str):
    from faster_whisper.audio import decode_audio

    from support_agent.voice import degrade as dg
    from support_agent.voice.speech import LISTEN_RATE, MeloSpeaker, WhisperListener

    print(f"loading the speech models (tts on {device}, stt on {device}) ...", flush=True)
    speaker, listener = MeloSpeaker(device=device), WhisperListener(device=device)

    def decode(wav: bytes):
        return decode_audio(io.BytesIO(wav), sampling_rate=LISTEN_RATE)

    return speaker, listener, decode, dg


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--", ".", ":(exclude)reports"))
    if args.official and (dirty or not commit):
        parser.error("--official needs git and a clean working tree, so that the commit describes the code")
    if args.official and (args.limit is not None or args.device != "cuda"):
        parser.error("--official measures every utterance on the source run's device (cuda)")
    manifest, utterances = source_utterances(args.source)
    if args.limit is not None:
        utterances = utterances[: args.limit]
    used = gpu_memory_used_mib() if args.device == "cuda" else None
    if used is not None and used > GPU_LIMIT_MIB:
        sys.exit(f"{used} MiB of GPU memory are in use (limit {GPU_LIMIT_MIB}): stop Ollama, or wait")

    speaker, listener, decode, dg = _load_models(args.device)
    mismatches = channel_mismatches(manifest, speaker.describe(), listener.describe())
    if mismatches:
        if args.official:
            sys.exit(f"the speech models differ from the source run's: {', '.join(mismatches)}")
        print(f"warning: not the source run's speech models ({', '.join(mismatches)}): a harness check only")

    started = datetime.now(UTC)
    run_id = "-".join(p for p in (started.strftime("%Y%m%d-%H%M%S"), STAGE, safe_name(args.label)) if p)
    out = (REPORTS if args.official else OUTPUTS / "v4r") / run_id
    out.mkdir(parents=True, exist_ok=False)
    from support_agent.voice.speech import versions

    record = {
        "run_id": run_id,
        "stage": "11-A",
        "started_at": started.isoformat(timespec="seconds"),
        "official": args.official,
        "commit": commit,
        "dirty": dirty,
        "source_run": args.source.name,
        "source_commit": manifest["commit"],
        "source_tasks_file": manifest["tasks_file"],
        "base_seed": manifest["config"]["base_seed"],
        "utterances": len(utterances),
        "limit": args.limit,
        "speaker": speaker.describe(),
        "listener": listener.describe(),
        "normalizer": manifest["voice_channel"].get("normalizer"),
        "degrade": dg.describe(),
        "channel_mismatches": mismatches,
        "gpu_mib_used_at_start": used,
        "versions": {"python": sys.version.split()[0]}
        | versions("numpy", "scipy", "av", "faster-whisper", "ctranslate2", "melotts", "torch"),
    }
    (out / "manifest.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1), "utf-8", newline="\n"
    )
    print(f"{run_id}: {len(utterances)} utterances -> {out}", flush=True)
    normalizer = normalize_heard if record["normalizer"] == "normalize_heard" else None
    begun = time.perf_counter()
    try:
        measure(
            utterances,
            speaker,
            listener,
            decode=decode,
            degrade=dg.degrade,
            description=dg.describe(),
            cache_path=args.cache,
            out=out,
            normalizer=normalizer,
            progress=lambda text: print(text, flush=True),
        )
    finally:  # also after a crash: how far it got
        written = (
            (out / "utterances.jsonl").read_text("utf-8").splitlines()
            if (out / "utterances.jsonl").exists()
            else []
        )
        summary = {
            "utterances": len(written),
            "finished": len(written) == len(utterances),
            "cached": sum('"cached": true' in line for line in written),
            "wall_seconds": round(time.perf_counter() - begun, 1),
            "ended_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=1), "utf-8", newline="\n")
        print(json.dumps(summary), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="CPU gate: the drawn conditions").add_argument("source", type=Path)
    run = commands.add_parser("run", help="synthesise and recognise (speech models)")
    run.add_argument("source", type=Path)
    run.add_argument("--official", action="store_true", help="write to reports/ (needs a clean tree)")
    run.add_argument("--label", default="")
    run.add_argument("--limit", type=int, default=None, help="only the first N utterances (not official)")
    run.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    run.add_argument("--cache", type=Path, default=CACHE)
    commands.add_parser("report", help="tables from a stage A run").add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    if args.command == "plan":
        _manifest, utterances = source_utterances(args.source)
        lines, passed = plan_gate(utterances)
        print("\n".join(lines))
        if not passed:
            sys.exit(1)
    elif args.command == "run":
        _run(args, parser)
    else:
        lines = (args.run_dir / "utterances.jsonl").read_text(encoding="utf-8").splitlines()
        print(report([json.loads(line) for line in lines if line.strip()]))


if __name__ == "__main__":
    main()
