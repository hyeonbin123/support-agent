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

# [파일 역할] 11단계 A단계(V4-R)의 실행기다. 9단계 V4 개발용 실행의 고객 발화를 그대로 다시 합성해, 깨끗한
# 음성과 전화 통화 조건으로 열화한 음성(voice/degrade.py)을 같은 인식기로 받아 적고 둘을 비교하는 표를 낸다.
# LLM과 에이전트는 없다.
# [왜 필요한가] 지금까지의 음성 수치는 깨끗한 합성 음성에서 나와 실제 통화보다 쉬운 조건이다. 9단계 V4의 본인
# 확인은 이름 받아쓰기에 기대므로, 같은 소리의 두 조건을 짝지어 '열화가 이름·번호를 얼마나 잃게 하는가'만 따로
# 잰다 (docs/experiments.md 11단계).
# [흐름] plan: 소리 없이 열화 조건만 뽑아 등록한 비율과 맞는지 보는 CPU 관문 → run: 다시 합성 → 16 kHz로 풀기
# → 깨끗한 인식 → 열화 → 다시 인식, 발화마다 utterances.jsonl 한 줄 → report: 기록에서 표와 과제 부트스트랩
# 구간을 다시 계산.
# [설계 이유] 공식 실행은 작업 트리가 깨끗하고, 음성 모델·장치가 원본 실행과 같아야 시작한다. 시험용 기록은
# 원본으로 받지 않는다. 왕복은 캐시해 멈춘 실행을 이어 갈 수 있다(키에 모델·열화 버전과 값·두 seed·글이 들어
# 있어 같은 키면 같은 결과다).

from __future__ import annotations

# 표준 라이브러리: argparse(plan·run·report 명령), hashlib(캐시 키 sha256), io(WAV 바이트를 디코더에 넘김),
# json(기록 읽고 쓰기), math·random·statistics(부트스트랩, 포아송 범위, 평균), re·unicodedata(한글 판별, 이름
# 비교의 NFKC), time·datetime(시간 재기, 실행 id).
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

# 프로젝트 모듈: derive_seed는 음성 채널이 쓴 것과 같은 seed 규칙, run.py의 도우미는 git 커밋·깨끗한 트리
# 확인, GPU 메모리 확인, 라벨을 파일 이름으로 바꾸기. 지표 함수는 analyze voice와 같은 글자 오류율·엔티티
# 규칙을 쓰려고 voice/metrics.py에서, 정규화는 V2·V4 채널과 같게 normalize_heard를 가져온다.
from support_agent.config import derive_seed
from support_agent.paths import OUTPUTS, REPORTS
from support_agent.run import _git, gpu_memory_used_mib, safe_name
from support_agent.voice.metrics import ENTITY_LABELS, edit_distance, entities, squeeze, survived
from support_agent.voice.normalize import normalize_heard
from support_agent.voice.speech import timed

# STAGE: 실행 id에 붙는 단계 이름. SOURCE_RUN: 다시 합성할 원본(9단계 V4 개발용 24과제 × 4회). CACHE: 왕복
# 결과 캐시 파일 (outputs/는 커밋하지 않는다).
STAGE = "v4r-a"
SOURCE_RUN = "20261004-154016-qwen3.5-4b-P0-R0-G0-F0-V4-v4-k0"  # stage 9, V4, development tasks
CACHE = OUTPUTS / "voice-cache" / "v4r-roundtrips.jsonl"
# Registered before measuring (docs/experiments.md, stage 11).
# 재기 전에 등록한 값들 (docs/experiments.md 11단계 "재현 확인", "관문").
# - REPRODUCE_MIN: 다시 합성한 깨끗한 음성의 받아쓰기가 9단계 기록과 글자까지 같은 발화가 90% 이상이면 깨끗한
#   쪽 수치를 9단계 채널과 같은 것으로 적는다.
# - GATE_MUFFLED, GATE_LOSS: CPU 관문에서 뽑힌 조건의 먹먹한 발화 15~25%, 잃은 프레임 1~3%.
# - GATE_BURST_SIGMAS: 돌발 소음 수가 포아송 기댓값 ±3 표준편차 안.
REPRODUCE_MIN = 0.90  # share of clean transcripts equal to the recorded ones, to tie the numbers to stage 9
GATE_MUFFLED = (0.15, 0.25)  # CPU gate on the drawn conditions: share of muffled utterances
GATE_LOSS = (0.01, 0.03)  # share of lost frames
GATE_BURST_SIGMAS = 3.0  # bursts within the Poisson mean +- 3 standard deviations
# The two speech models took about 3.1 GB in stage 9; above this much memory in use by anything else, the
# 11 GB card would have under 2 GB to spare and could spill into shared memory.
# 시작할 때 GPU 전체 사용량이 6,000 MiB를 넘으면 시작하지 않는다 (docs/experiments.md 11단계 "예상 GPU 시간").
# 넘으면 Ollama를 내리거나 기다리라는 메시지로 끝난다.
GPU_LIMIT_MIB = 6000
# 차이의 95% 구간: 과제 24개 단위로 다시 뽑는 부트스트랩 10,000회, 등록한 고정 seed 20261005.
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 20261005
# 원본 실행의 manifest(voice_channel)와 같아야 하는 음성 설정 키와, 그 키가 합성(tts)·인식(stt) 중 어느 쪽
# describe()에 있는지. 하나라도 다르면 공식 실행을 거절한다 (재현 확인 규칙).
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
# measure()의 normalizer 기본값 표시용 객체. 인자를 주지 않으면 normalize_heard를 쓰고, None을 주면 정규화하지
# 않는다(원본이 V1인 경우). None과 '안 줌'을 구분하려고 따로 둔다.
_KEEP = object()  # the default normaliser: normalize_heard (None means none)


# 다시 합성할 고객 발화 하나. 원본 기록에서 읽은 값(말한 글, 말로 푼 글, 기록된 받아쓰기, 음성 길이)과 이번에
# 쓸 두 seed를 함께 들고 있다.
@dataclass(frozen=True)
class Utterance:
    task_id: str
    trial: int
    index: int  # position in the episode's voice log: the channel's seed index
    seed: int  # the channel's synthesis seed
    # 열화 조건의 seed: derive_seed(기본 seed, 과제, 시도, "degrade", 순번). 합성 seed와 역할 이름만 달라 서로
    # 독립이다.
    degrade_seed: int
    name: str  # the registered name of the task's customer
    said: str
    spoken: str
    # 원본 실행에 기록된 받아쓰기·정규화 뒤 글·음성 길이. 재현 확인(깨끗한 받아쓰기가 같은가)과 CPU 관문(이
    # 길이로 조건을 뽑음)에 쓴다.
    recorded_heard: str
    recorded_text: str
    recorded_audio_seconds: float


# 글 안에 등록 이름이 글자까지 같게(공백만 무시) 들어 있는지. 9단계 V4의 verify_caller가 받아들이는 기준(허용
# 오차 k = 0)과 같게 맞춘 '이름 정확 생존' 지표의 판정이다.
def name_in(name: str, text: str) -> bool:
    """The registered name, letter for letter, with spaces ignored: what verify_caller (k = 0) accepts."""
    squeezed = "".join(unicodedata.normalize("NFKC", text).split())
    return "".join(unicodedata.normalize("NFKC", name).split()) in squeezed


# 원본 음성 실행의 기록에서 다시 합성할 발화 목록을 만든다. 원본이 지금 코드·데이터와 같은 조건인지 먼저
# 확인한다: 시험용 기록이 아니어야 하고, 음성 실행이어야 하고, 과제 파일과 seed DB의 해시가 실행 때와 같아야
# 한다. 하나라도 어긋나면 ValueError.
# 호출 위치: main()의 plan, _run().
def source_utterances(run_dir: Path) -> tuple[dict[str, Any], list[Utterance]]:
    """The synthesised customer utterances of a voice run, in the order of the run's records."""
    # DB·과제 모듈은 고객 이름을 얻는 데만 필요해 함수 안에서 불러온다.
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from support_agent import db
    from support_agent.seed import build_seed_engine
    from support_agent.tasks import load_tasks

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    # 시험용 과제의 기록은 열지 않는다는 규칙을 코드로 막는다 (A단계는 개발용 기록만 쓴다).
    if "test" in Path(str(manifest.get("tasks_file", ""))).stem:
        raise ValueError("stage A uses development records only: test records are not opened")
    if config.get("voice", "V0") == "V0" or not manifest.get("voice_channel"):
        raise ValueError("not a voice run: nothing was synthesised")
    # 실행 뒤에 과제 파일이 바뀌었으면 고객·말한 글의 근거가 달라지므로 거절한다 (manifest의 과제별 sha256과
    # 비교).
    tasks = {t.id: t for t in load_tasks(manifest["tasks_file"])}
    changed = [i for i, sha in manifest["task_sha256"].items() if i not in tasks or tasks[i].sha256() != sha]
    if changed:
        raise ValueError(f"the task file changed since the run: {', '.join(changed)}")
    # seed DB를 다시 만들어 상태 해시를 비교한다. 같아야 과제 고객의 등록 이름이 원본 실행 때와 같다.
    engine = build_seed_engine()
    if db.state_hash(db.dump_db(engine)) != manifest["seed_hash"]:
        raise ValueError("the seed database changed since the run")
    with Session(engine) as session:
        names = {c.id: c.name for c in session.scalars(select(db.Customer))}

    base_seed = config["base_seed"]
    episodes = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text("utf-8").splitlines()]
    out = []
    # (과제, 시도) 순으로 정렬해 실행마다 같은 순서로 발화를 처리한다. index는 그 에피소드의 음성 기록 안
    # 순번이고, 채널이 합성 seed를 만들 때 쓴 순번과 같다.
    for e in sorted(episodes, key=lambda e: (e["task_id"], e["trial"])):
        task_id, trial = e["task_id"], e["trial"]
        for index, u in enumerate(e.get("voice") or []):
            # 말로 푼 글에 한글이 없으면 채널이 합성하지 않은 발화이므로 뺀다 (channel.py hear()와 같은 조건).
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


# 지금 올린 합성기·인식기의 describe()가 원본 실행의 음성 설정과 다른 키 목록을 돌려준다. 원본 manifest는 두
# 설정을 한 dict로 합쳐 device가 하나뿐이라, 양쪽 장치를 모두 그 값과 비교한다.
# 호출 위치: _run(). 다르면 공식 실행은 멈추고, 아니면 경고만 내고 하니스 확인으로 돈다.
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


# 측정 본체: 발화마다 합성 → 디코딩 → 깨끗한 인식 → 열화 → 다시 인식을 하고, 기록 한 줄을 utterances.jsonl에
# 쓴다.
# 합성기·인식기·디코더·열화 함수를 인자로 받으므로 테스트는 가짜 모델로 이 함수를 그대로 돌린다
# (tests/test_stage11.py). 반환은 기록 목록이다.
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
    # 정규화 선택: 기본은 V2의 normalize_heard, None이면 받아쓰기 그대로. 두 받아쓰기에 같은 정규화를
    # 적용한다.
    normalise = normalize_heard if normalizer is _KEEP else (normalizer or (lambda text: text))
    # 캐시 키의 공통 부분: 합성기·인식기 설정과 열화의 버전·값. 이 중 하나라도 바뀌면 키가 달라져 예전 결과를
    # 쓰지 않는다.
    identity = json.dumps([speaker.describe(), listener.describe(), description], sort_keys=True)
    cache: dict[str, dict[str, Any]] = {}
    # 이전 실행이 남긴 캐시를 읽어 둔다. 실행이 멈추면 같은 명령을 다시 돌려 끝난 발화를 다시 쓰는
    # 규칙(docs/experiments.md 11단계)을 이것으로 지킨다.
    if cache_path and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            cache[entry["key"]] = entry
    records, started = [], time.perf_counter()
    # 기록은 발화마다 바로 쓰고 flush한다. 도중에 멈춰도 어디까지 했는지가 파일에 남는다.
    with (out / "utterances.jsonl").open("w", encoding="utf-8", newline="\n") as written:
        for n, u in enumerate(utterances, 1):
            # 발화별 캐시 키: 공통 부분 + 합성 seed + 열화 seed + 말로 푼 글. 같은 키면 합성·인식이 결정적이라
            # 같은 결과다.
            key = hashlib.sha256(f"{identity}|{u.seed}|{u.degrade_seed}|{u.spoken}".encode()).hexdigest()
            entry = cache.get(key)
            cached = entry is not None
            if entry is None:
                # 합성 → WAV를 녹음 읽듯 16 kHz로 풀기 → 깨끗한 인식 → 같은 표본에 열화 → 다시 인식. 두 인식이
                # 같은 소리에서 출발하므로 차이는 열화의 몫이다.
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
                # 새로 계산한 결과는 캐시 파일 끝에 한 줄씩 덧붙인다.
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with cache_path.open("a", encoding="utf-8", newline="\n") as kept:
                        kept.write(json.dumps(entry, ensure_ascii=False) + "\n")
            # 기록 한 줄 = 발화 정보 + 왕복 결과(키 제외) + 정규화한 두 글 + 캐시에서 왔는지. report()는 이
            # 줄들만으로 표를 다시 낸다.
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


# 돌발 소음 수의 허용 범위: 포아송 분포는 평균과 분산이 같으므로 기댓값(오디오 분 × 분당 횟수) ± 3·√기댓값.
def burst_bounds(seconds: float, per_minute: float = 1.0) -> tuple[float, float]:
    expected = seconds / 60.0 * per_minute
    spread = GATE_BURST_SIGMAS * math.sqrt(expected)
    return expected - spread, expected + spread


# CPU 관문: 모든 발화에 대해 기록된 길이로 열화 조건만 뽑아(소리 없음) 등록한 비율과 맞는지 본다. 먹먹한 발화
# 비율, 잃은 프레임 비율, 돌발 소음 수, 배경 SNR 범위와 두 색이 모두 나왔는지. 반환은 (출력 줄, 통과 여부).
# 왜 필요한가: GPU를 쓰기 전에 구현이 데이터 크기에서 등록한 값대로 뽑는지 확인한다. 어기면 GPU로 가지 않는다.
# 실제 실행은 다시 합성한 소리의 길이로 뽑으므로 수가 조금 다를 수 있다.
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
    # 배경 SNR은 모두 [15 − 3, 15 + 3] dB 안이어야 한다.
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


# 한 기록의 한쪽(깨끗한 또는 열화한 글)에 대해 지표별 (분자, 분모)를 센다. 글자 오류율은 (편집 거리, 말한 글자
# 수), 그대로 전달·인식 안 됨은 (0 또는 1, 1).
# 이름은 말한 글에 등록 이름이 있을 때만 분모에 넣는다. 엔티티는 말한 글에서 뽑아 metrics.survived로 센다.
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


# 기록 하나의 지표별 ((깨끗한 분자, 분모), (열화한 분자, 분모)).
def counts(r: dict[str, Any]) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    """Per metric: ((clean numerator, denominator), (degraded numerator, denominator))."""
    clean, degraded = _side(r, r["clean_text"]), _side(r, r["degraded_text"])
    return {key: (clean[key], degraded[key]) for key in clean}


# 열화 − 깨끗함의 합친 비율 차이와 95% 구간. 비율은 뽑힌 과제를 모두 합쳐(분자 합 / 분모 합) 계산하고, 과제를
# 다시 뽑을 때 그 과제의 발화와 두 조건이 함께 뽑힌다 (docs/experiments.md 11단계 "구간").
# 분모가 0인 뽑기는 버린다. 점 추정부터 분모가 0이면 nan을 돌려준다.
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
    # 고정 seed로 과제 목록과 같은 크기만큼 복원 추출을 반복하고, 정렬한 차이의 2.5%·97.5% 위치를 구간으로
    # 쓴다.
    rng = random.Random(BOOTSTRAP_SEED)
    rounds = sorted(
        d
        for d in (rate([rng.choice(tasks) for _ in tasks]) for _ in range(BOOTSTRAP_ROUNDS))
        if d is not None
    )
    return point, rounds[int(0.025 * len(rounds))], rounds[int(0.975 * len(rounds)) - 1]


# 표의 행 순서: 글자 오류율, 그대로 전달, 인식 안 됨, 이름(정확히), 주문·접수 번호(id), 그다음 나머지
# 엔티티(이메일, 전화번호, 금액, 날짜).
_ROWS = (
    ("cer", "글자 오류율"),
    ("exact", "그대로 전달"),
    ("empty", "인식 안 됨"),
    ("name", "이름 (정확히, k = 0)"),
    *((kind, label) for kind, label in ENTITY_LABELS.items() if kind == "id"),
    *((kind, label) for kind, label in ENTITY_LABELS.items() if kind != "id"),
)


# 표 칸: 글자 오류율은 비율만, 나머지는 '분자/분모 (비율)'. 분모가 0이면 '-'.
def _cell(key: str, n: int, d: int) -> str:
    if not d:
        return "-"
    return f"{n / d:.2%}" if key == "cer" else f"{n}/{d} ({n / d:.1%})"


# 기록 묶음의 글자 오류율 = 편집 거리 합 / 말한 글자 수 합. 재현 확인 줄과 조건별 표에 쓴다.
def _cer(records: list[dict[str, Any]], field: str) -> float:
    errors = sum(edit_distance(squeeze(r["said"]), squeeze(r[field])) for r in records)
    length = sum(len(squeeze(r["said"])) for r in records)
    return errors / length if length else math.nan


# utterances.jsonl의 기록에서 표를 다시 낸다. 모델 없이 돈다. 순서: 재현 확인 → 지표 표(깨끗한, 열화한, 차이와
# 구간) → 실제로 들어간 조건 → 조건별 글자 오류율(기술용, 구간 없음).
# 호출 위치: main()의 report.
def report(records: list[dict[str, Any]]) -> str:
    per_record = [(r, counts(r)) for r in records]
    tasks = sorted({r["task_id"] for r in records})
    lines = ["| 항목 | 값 |", "|---|---|", f"| 발화 | {len(records)} |", f"| 과제 | {len(tasks)} |", ""]

    # 재현 확인: 다시 합성한 깨끗한 받아쓰기가 9단계 기록과 글자까지 같은 비율. 90% 이상이면 깨끗한 쪽 수치를
    # 9단계 채널과 이어 적는다 (2026-10-08 결과에서는 99.7%, README).
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
        # 지표마다 과제별로 깨끗한·열화한 쪽의 분자·분모를 모은 뒤, 바꿀 수 없는 튜플로 굳혀
        # pooled_difference에 넘긴다.
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

    # 실제로 들어간 조건 요약: 먹먹함 비율, 잃은 프레임 비율, 돌발 소음 수, SNR 범위, 색 분포, 음량을 낮춘
    # 발화 수. 등록한 비율과 맞았는지 사람이 확인하는 줄이다.
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
    # 효과별로 발화를 나눠 본 글자 오류율. 효과가 있는 발화와 없는 발화를 비교하는 기술용 표이고 구간은 내지
    # 않는다.
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


# 음성 모델과 열화 모듈을 불러온다 (음성 의존성 그룹이 필요해 run 명령에서만 부른다).
# decode는 faster-whisper가 녹음을 읽을 때와 같은 decode_audio로 WAV를 16 kHz로 푼다. 깨끗한 음성이 9단계
# 채널과 같은 소리가 되게 하려는 것이다 (docs/experiments.md 11단계 "다르게 정한 것").
def _load_models(device: str):
    from faster_whisper.audio import decode_audio

    from support_agent.voice import degrade as dg
    from support_agent.voice.speech import LISTEN_RATE, MeloSpeaker, WhisperListener

    print(f"loading the speech models (tts on {device}, stt on {device}) ...", flush=True)
    speaker, listener = MeloSpeaker(device=device), WhisperListener(device=device)

    def decode(wav: bytes):
        return decode_audio(io.BytesIO(wav), sampling_rate=LISTEN_RATE)

    return speaker, listener, decode, dg


# run 명령: 시작 조건 확인 → 모델 적재 → 실행 기록(manifest.json) → measure → 끝나거나 멈추면 summary.json.
def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    # 공식 실행은 git 커밋이 코드를 설명할 수 있게 작업 트리가 깨끗해야 한다(reports/ 아래 변경은 제외). 또
    # 모든 발화를 원본과 같은 장치(cuda)로 재야 한다.
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--", ".", ":(exclude)reports"))
    if args.official and (dirty or not commit):
        parser.error("--official needs git and a clean working tree, so that the commit describes the code")
    if args.official and (args.limit is not None or args.device != "cuda"):
        parser.error("--official measures every utterance on the source run's device (cuda)")
    manifest, utterances = source_utterances(args.source)
    if args.limit is not None:
        utterances = utterances[: args.limit]
    # GPU 메모리를 이미 많이 쓰고 있으면 시작하지 않는다. 두 음성 모델이 들어갈 자리가 모자라 공유 메모리로
    # 넘칠 수 있기 때문이다 (위 GPU_LIMIT_MIB 주석).
    used = gpu_memory_used_mib() if args.device == "cuda" else None
    if used is not None and used > GPU_LIMIT_MIB:
        sys.exit(f"{used} MiB of GPU memory are in use (limit {GPU_LIMIT_MIB}): stop Ollama, or wait")

    speaker, listener, decode, dg = _load_models(args.device)
    # 음성 모델·버전·장치가 원본과 하나라도 다르면 공식 실행은 멈추고, 공식이 아니면 경고만 낸다 (CPU 하니스
    # 확인에서는 장치가 달라 경고가 나오는 것이 맞다).
    mismatches = channel_mismatches(manifest, speaker.describe(), listener.describe())
    if mismatches:
        if args.official:
            sys.exit(f"the speech models differ from the source run's: {', '.join(mismatches)}")
        print(f"warning: not the source run's speech models ({', '.join(mismatches)}): a harness check only")

    # 실행 id = 시각-단계-라벨. 공식이면 reports/(커밋 대상), 아니면 outputs/v4r/에 쓴다. 같은 폴더가 있으면
    # 덮어쓰지 않고 실패한다(exist_ok=False).
    started = datetime.now(UTC)
    run_id = "-".join(p for p in (started.strftime("%Y%m%d-%H%M%S"), STAGE, safe_name(args.label)) if p)
    out = (REPORTS if args.official else OUTPUTS / "v4r") / run_id
    out.mkdir(parents=True, exist_ok=False)
    from support_agent.voice.speech import versions

    # 실행 기록: 커밋, 원본 실행과 그 커밋·과제 파일, 기본 seed, 발화 수, 두 모델 설정, 정규화, 열화 버전과
    # 값, 불일치 목록, 시작 때 GPU 사용량, 패키지 버전. 나중에 이 실행을 다시 해석할 근거다.
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
    # 원본 실행이 쓴 정규화를 그대로 따른다: V2·V4면 normalize_heard, V1이면 없음.
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


# 명령줄: plan(CPU 관문, 통과 못 하면 종료 코드 1), run(측정), report(표). 명령 예는 모듈 docstring과
# docs/experiments.md 11단계 "명령"에 있다.
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
