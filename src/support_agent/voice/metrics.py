"""What the speech channel did to the customer's words, computed from episode records. No model here."""

# [파일 역할] 음성 채널(5·9·11단계)이 고객의 말에 무엇을 했는지를 에피소드 기록(episodes.jsonl의 voice
# 항목)만으로 다시 계산한다. 글자 오류율, 엔티티(이메일·전화번호·주문 번호·금액·날짜) 생존율, 본인 확인
# 성공률과 과제 단위로 짝지은 부트스트랩 구간을 낸다.
# [왜 필요한가] 음성 인식의 글자 오류율은 낮아도 성공률이 크게 떨어질 수 있다. 틀린 글자가 도구 인자가 되는
# 값에 몰리기 때문이다 (README 5단계 결과). 그래서 글자 오류율만이 아니라 '도구가 받는 형태 그대로 남았는가'를
# 엔티티별로 센다.
# [설계 이유] 모델을 부르지 않는 순수 계산이라 GPU 없이 기록에서 언제든 같은 수치를 다시 낸다. 부트스트랩
# seed도 고정이다.
# [연결] analyze.py의 voice·voice-worst·smoke 명령과 voice_real.py(11단계 표)가 이 모듈의 함수를 쓴다.

from __future__ import annotations

# random: seed를 고정한 부트스트랩 표본 추출. re: 엔티티 정규식. median: 합성·인식 시간의 p50.
import random
import re
from collections.abc import Iterable
from statistics import median
from typing import Any

# 본인 확인 성공률 차이의 부트스트랩 반복 수(10,000회, docs/experiments.md 5단계 "지표")와 고정 seed. seed가
# 고정이라 같은 기록이면 같은 구간이 나온다.
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 20260921
# Written-form entities the agent needs to get right. Checked one after the other; a match is cut out of the
# text before the next pattern runs, so "O-91010" is not counted again as a number.
ENTITY_PATTERNS: tuple[tuple[str, str], ...] = (
    # 이메일: 영문·숫자·._+- 로 된 아이디 @ 점으로 이어진 도메인.
    ("email", r"[A-Za-z0-9._+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"),
    # 휴대전화 번호: 01x 뒤에 3~4자리와 4자리, 구분자(- 공백 .)는 있어도 없어도 된다. 앞뒤에 숫자가 더 붙으면
    # 번호가 아니다.
    ("phone", r"(?<!\d)01\d[- .]?\d{3,4}[- .]?\d{4}(?!\d)"),
    # 주문·접수 번호: 영문 1~3자 뒤에 '-'로 이어진 덩어리 (O-91010, RT-O-91009-1, AD-C-9005-1 같은 형태).
    ("id", r"(?<![A-Za-z])[A-Za-z]{1,3}(?:-[A-Za-z0-9]+)+"),
    # 금액: 숫자(쉼표 허용) 뒤에 '원'. 날짜: 'N월 N일'.
    ("amount", r"\d[\d,]*\s*원"),
    ("date", r"\d{1,2}월\s*\d{1,2}일"),
)
# 표에 쓰는 엔티티의 한국어 이름. voice_table()과 voice_real.py의 표(_ROWS)가 이 순서와 이름을 쓴다.
ENTITY_LABELS = {
    "email": "이메일",
    "phone": "전화번호",
    "id": "주문·접수 번호",
    "amount": "금액",
    "date": "날짜",
}
# 비교할 때 지우는 문자: 공백과 문장부호. 인식기가 쉼표나 마침표를 다르게 찍은 것은 오류로 세지 않으려는
# 것이다.
_IGNORED = re.compile(r"[\s.,?!~'\"“”‘’()\[\]·:;-]")


# 두 글을 비교하는 정규형. 글자 오류율, '그대로 전달' 여부, voice_real.py의 같은 지표가 모두 이 형태로
# 비교한다.
def squeeze(text: str) -> str:
    """The form in which two transcripts are compared: no spaces, no punctuation, lower case."""
    return _IGNORED.sub("", text).lower()


# 두 문자열의 편집 거리(레벤슈타인: 삽입·삭제·치환 각 1). 외부 라이브러리 없이 동적 계획법으로 계산하고, 이전
# 행 하나만 들고 있어 메모리는 b의 길이에 비례한다. 글자 오류율 = 편집 거리 합 / 말한 글자 수 합의 분자다.
def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            # 세 경우 중 최솟값: a의 글자 삭제(위 칸 + 1), b의 글자 삽입(왼쪽 칸 + 1), 치환 또는 일치(대각선 +
            # 글자가 다르면 1).
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


# 고객이 쓴 글(said)에서 엔티티를 (종류, 원문) 목록으로 뽑는다. ENTITY_PATTERNS 순서대로 찾고, 찾은 부분은
# 공백으로 지운 뒤 다음 패턴을 돌린다. 그래서 "O-91010"의 숫자가 금액이나 다른 엔티티로 한 번 더 세어지지
# 않는다.
def entities(said: str) -> list[tuple[str, str]]:
    found, rest = [], said
    for kind, pattern in ENTITY_PATTERNS:
        found += [(kind, match) for match in re.findall(pattern, rest)]
        rest = re.sub(pattern, " ", rest)
    return found


# 엔티티가 에이전트가 받은 글(text)에 '도구 인자로 쓸 수 있는 형태로' 남았는지 본다. 종류마다 허용하는 표기
# 차이가 다르다. 더 긴 값의 일부로만 들어 있는 경우는 남은 것으로 치지 않는다.
def survived(kind: str, entity: str, text: str) -> bool:
    """True when the agent can read the entity in `text` as the tools need it, and not as a part of a
    longer one ("O-12" inside "O-123", "1,000원" inside "11,000원")."""
    # 전화번호: 같은 숫자열이면 구분자를 어떻게 넣었든 인정한다. 숫자 사이마다 선택적 구분자를 넣은 정규식을
    # 만들고, 앞뒤에 숫자가 더 붙지 않아야 한다.
    if kind == "phone":  # any grouping of the same digits
        wanted = r"[- .]?".join(re.sub(r"\D", "", entity))
        return re.search(rf"(?<!\d){wanted}(?!\d)", text) is not None
    # 금액: 양쪽 모두 숫자 사이 쉼표와 공백을 지우고 비교한다. 앞에 숫자나 쉼표가 없어야 하므로 "1,000원"이
    # "11,000원" 안에서 맞았다고 치지 않는다.
    if kind == "amount":
        wanted = re.sub(r"[,\s]", "", entity)
        return (
            re.search(rf"(?<![\d,]){re.escape(wanted)}", re.sub(r"(?<=\d),(?=\d)|\s", "", text)) is not None
        )
    # 날짜: 공백을 지우고 비교하고, 앞에 숫자가 없어야 한다 ("1월 2일"이 "11월 2일" 안에서 맞지 않게).
    if kind == "date":
        wanted = re.escape(re.sub(r"\s", "", entity))
        return re.search(rf"(?<!\d){wanted}", re.sub(r"\s", "", text)) is not None
    # email, id: the very string, not inside a longer one
    return (
        re.search(rf"(?<![0-9A-Za-z._-]){re.escape(entity)}(?![0-9A-Za-z])", text, re.IGNORECASE) is not None
    )


# 여러 에피소드의 음성 기록(발화별 said, spoken, heard, text, 시간)을 한 목록으로 편다. 텍스트 실행(V0)에는
# voice 항목이 없어 빈 목록이 된다.
def utterances(episodes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [u for e in episodes for u in e.get("voice") or []]


# 한 실행의 채널 요약: 발화 수, 그대로 전달된 수, 인식 안 된 수, 글자 오류율, 음성 길이 합, 합성·인식 시간
# p50, 엔티티 종류별 (남은 수, 전체 수).
# 비교 대상은 고객이 쓴 글(said)과 에이전트가 실제로 받은 글(text: V2면 정규화 뒤)이다.
def channel_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    heard = utterances(episodes)
    errors = sum(edit_distance(squeeze(u["said"]), squeeze(u["text"])) for u in heard)
    length = sum(len(squeeze(u["said"])) for u in heard)
    by_kind: dict[str, list[bool]] = {}
    for u in heard:
        for kind, entity in entities(u["said"]):
            by_kind.setdefault(kind, []).append(survived(kind, entity, u["text"]))
    # 시간 p50은 캐시에서 꺼낸 발화를 빼고 계산한다. 캐시된 발화는 이번 실행에서 합성·인식을 하지 않았으므로
    # 그 시간은 이번 실행의 측정이 아니다.
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


# 본인 확인 도구 이름. 5단계까지는 find_customer, 9단계 V4는 발신 번호로 확인하는 verify_caller다.
# analyze.py의 smoke 표도 이 튜플로 '확인 도구를 부르지 않은 에피소드'를 센다.
IDENTITY_TOOLS = ("find_customer", "verify_caller")  # verify_caller: voice V4 (stage 9)


# 에피소드에서 본인 확인 도구가 한 번이라도 성공했는지. 모든 과제가 본인 확인에서 시작하므로, 기준 성공률이
# 낮아 pass^1로 차이가 안 보일 때를 대비한 2차 지표다 (docs/experiments.md 5단계 "지표").
def identified(episode: dict[str, Any]) -> bool:
    """The customer was verified at least once: the first thing every task needs."""
    return any(call["name"] in IDENTITY_TOOLS and call["ok"] for call in episode["tool_calls"])


# 과제별 본인 확인 성공 비율. infra_error 에피소드는 분모에서 뺀다 (제공자 오류·도구 버그는 에이전트의 실패가
# 아니라는 판정 규칙, docs/design.md "에이전트 루프 규칙").
def identified_by_task(episodes: list[dict[str, Any]]) -> dict[str, float]:
    by_task: dict[str, list[bool]] = {}
    for e in episodes:
        if e["status"] != "infra_error":
            by_task.setdefault(e["task_id"], []).append(identified(e))
    return {task: sum(v) / len(v) for task, v in by_task.items()}


# 두 실행의 과제별 비율 차이(other − base)의 평균과 95% 부트스트랩 구간. 과제를 다시 뽑을 때 같은 과제의 두
# 값이 짝으로 함께 뽑히므로, 과제 난이도 차이가 구간을 넓히지 않는다.
# 두 실행의 과제 집합이 다르면 짝을 지을 수 없어 ValueError를 낸다.
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


# docs/experiments.md에 붙일 마크다운 표 세 개(채널 요약, 엔티티 생존, 본인 확인 성공과 기준과의 차이)를
# 만든다. 첫 실행이 짝지은 차이의 기준이다.
# 호출 위치: analyze.py의 voice 명령 (analyze voice <텍스트 실행> <음성 실행> ...).
def voice_table(runs: dict[str, list[dict[str, Any]]]) -> str:
    """Markdown tables for docs/experiments.md. The first run is the base of the paired differences."""
    names = list(runs)
    lines = [
        "| 실행 | 발화 | 그대로 전달 | 인식 안 됨 | 글자 오류율 | 음성 길이 합 | 합성 p50 | 인식 p50 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        s = channel_summary(runs[name])
        # 음성 기록이 없는 실행(V0, 텍스트)은 채널 수치 없이 이름만 적는다.
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

    # 본인 확인 성공률 표: 기준 실행과 과제 단위로 짝지어 차이와 구간을 적는다.
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


# 엔티티를 잃은 발화를 (같은 글은 한 번만) 최대 limit개 보여 준다. V2 정규화 규칙(normalize.py)은 개발용 V1
# 기록에서 이 목록만 보고 만들었다 (docs/experiments.md 5단계 "V2의 규칙은 개발용 기록만 보고 만든다").
# 호출 위치: analyze.py의 voice-worst 명령. 시험용 실행이면 그쪽에서 거부한다 (시험용 기록을 보고 규칙을
# 만들지 않으려는 장치).
def worst(episodes: list[dict[str, Any]], limit: int = 40) -> str:
    """Utterances that lost an entity: the material the normaliser is built from (development tasks only)."""
    out, seen = [], set()
    for u in utterances(episodes):
        lost = [entity for kind, entity in entities(u["said"]) if not survived(kind, entity, u["text"])]
        if lost and u["said"] not in seen:
            seen.add(u["said"])
            out.append(f"- said : {u['said']}\n  heard: {u['heard']}\n  lost : {lost}")
    return "\n".join(out[:limit]) + f"\n({len(out)} distinct utterances lost an entity)"
