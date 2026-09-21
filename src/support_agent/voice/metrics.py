"""What the speech channel did to the customer's words, computed from episode records. No model here."""

from __future__ import annotations

import random
import re
from collections.abc import Iterable
from statistics import median
from typing import Any

BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 20260921
# Written-form entities the agent needs to get right. Checked one after the other; a match is cut out of the
# text before the next pattern runs, so "O-91010" is not counted again as a number.
ENTITY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("email", r"[A-Za-z0-9._+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"),
    ("phone", r"(?<!\d)01\d[- .]?\d{3,4}[- .]?\d{4}(?!\d)"),
    ("id", r"(?<![A-Za-z])[A-Za-z]{1,3}(?:-[A-Za-z0-9]+)+"),
    ("amount", r"\d[\d,]*\s*원"),
    ("date", r"\d{1,2}월\s*\d{1,2}일"),
)
ENTITY_LABELS = {
    "email": "이메일",
    "phone": "전화번호",
    "id": "주문·접수 번호",
    "amount": "금액",
    "date": "날짜",
}
_IGNORED = re.compile(r"[\s.,?!~'\"“”‘’()\[\]·:;-]")


def squeeze(text: str) -> str:
    """The form in which two transcripts are compared: no spaces, no punctuation, lower case."""
    return _IGNORED.sub("", text).lower()


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def entities(said: str) -> list[tuple[str, str]]:
    found, rest = [], said
    for kind, pattern in ENTITY_PATTERNS:
        found += [(kind, match) for match in re.findall(pattern, rest)]
        rest = re.sub(pattern, " ", rest)
    return found


def survived(kind: str, entity: str, text: str) -> bool:
    """True when the agent can read the entity in `text` exactly as the tools need it."""
    if kind == "phone":
        return re.sub(r"\D", "", entity) in re.sub(r"[- .]", "", text)
    if kind == "amount":
        return re.sub(r"[,\s]", "", entity) in re.sub(r"[,\s]", "", text)
    if kind == "date":
        return re.sub(r"\s", "", entity) in re.sub(r"\s", "", text)
    return entity.lower() in text.lower()  # email, id: the very string


def utterances(episodes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [u for e in episodes for u in e.get("voice") or []]


def channel_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    heard = utterances(episodes)
    errors = sum(edit_distance(squeeze(u["said"]), squeeze(u["text"])) for u in heard)
    length = sum(len(squeeze(u["said"])) for u in heard)
    by_kind: dict[str, list[bool]] = {}
    for u in heard:
        for kind, entity in entities(u["said"]):
            by_kind.setdefault(kind, []).append(survived(kind, entity, u["text"]))
    fresh = [u for u in heard if not u.get("cached")]
    return {
        "utterances": len(heard),
        "exact": sum(squeeze(u["said"]) == squeeze(u["text"]) for u in heard),
        "nothing_heard": sum(not u["text"] for u in heard),
        "cer": errors / length if length else float("nan"),
        "audio_seconds": sum(u["audio_seconds"] for u in heard),
        "tts_ms_p50": median(u["tts_ms"] for u in fresh) if fresh else float("nan"),
        "stt_ms_p50": median(u["stt_ms"] for u in fresh) if fresh else float("nan"),
        "entities": {kind: (sum(v), len(v)) for kind, v in by_kind.items()},
    }


def identified(episode: dict[str, Any]) -> bool:
    """The customer was verified at least once: the first thing every task needs."""
    return any(call["name"] == "find_customer" and call["ok"] for call in episode["tool_calls"])


def identified_by_task(episodes: list[dict[str, Any]]) -> dict[str, float]:
    by_task: dict[str, list[bool]] = {}
    for e in episodes:
        if e["status"] != "infra_error":
            by_task.setdefault(e["task_id"], []).append(identified(e))
    return {task: sum(v) / len(v) for task, v in by_task.items()}


def paired_mean_difference(base: dict[str, float], other: dict[str, float]) -> tuple[float, float, float]:
    """Mean over tasks of (other - base) with a 95% interval; tasks are resampled and keep their pair."""
    if set(base) != set(other):
        raise ValueError("both runs must cover the same tasks")
    diffs = [other[task] - base[task] for task in sorted(base)]
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(BOOTSTRAP_ROUNDS))
    return (
        sum(diffs) / len(diffs),
        means[int(0.025 * BOOTSTRAP_ROUNDS)],
        means[int(0.975 * BOOTSTRAP_ROUNDS) - 1],
    )


def voice_table(runs: dict[str, list[dict[str, Any]]]) -> str:
    """Markdown tables for docs/experiments.md. The first run is the base of the paired differences."""
    names = list(runs)
    lines = [
        "| 실행 | 발화 | 그대로 전달 | 인식 안 됨 | 글자 오류율 | 음성 길이 합 | 합성 p50 | 인식 p50 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        s = channel_summary(runs[name])
        if not s["utterances"]:
            lines.append(f"| {name} | (텍스트) | | | | | | |")
            continue
        lines.append(
            f"| {name} | {s['utterances']} | {s['exact']} ({s['exact'] / s['utterances']:.1%}) "
            f"| {s['nothing_heard']} | {s['cer']:.2%} | {s['audio_seconds'] / 60:.1f}분 "
            f"| {s['tts_ms_p50']:.0f} ms | {s['stt_ms_p50']:.0f} ms |"
        )

    lines += [
        "",
        "| 실행 | " + " | ".join(ENTITY_LABELS.values()) + " |",
        "|---|" + "---|" * len(ENTITY_LABELS),
    ]
    for name in names:
        found = channel_summary(runs[name])["entities"]
        if not found:
            continue
        cells = []
        for kind in ENTITY_LABELS:
            ok, total = found.get(kind, (0, 0))
            cells.append(f"{ok}/{total} ({ok / total:.0%})" if total else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    base = identified_by_task(runs[names[0]])
    lines += ["", "| 실행 | 본인 확인에 성공한 에피소드 | 기준과의 차이 [95% 구간] |", "|---|---|---|"]
    for index, name in enumerate(names):
        rate = identified_by_task(runs[name])
        mean = sum(rate.values()) / len(rate)
        if index == 0:
            lines.append(f"| {name} (기준) | {mean:.1%} | |")
        else:
            diff, low, high = paired_mean_difference(base, rate)
            lines.append(f"| {name} | {mean:.1%} | {diff:+.1%}p [{low:+.1%}, {high:+.1%}] |")
    return "\n".join(lines)


def worst(episodes: list[dict[str, Any]], limit: int = 40) -> str:
    """Utterances that lost an entity: the material the normaliser is built from (development tasks only)."""
    out, seen = [], set()
    for u in utterances(episodes):
        lost = [entity for kind, entity in entities(u["said"]) if not survived(kind, entity, u["text"])]
        if lost and u["said"] not in seen:
            seen.add(u["said"])
            out.append(f"- said : {u['said']}\n  heard: {u['heard']}\n  lost : {lost}")
    return "\n".join(out[:limit]) + f"\n({len(out)} distinct utterances lost an entity)"
