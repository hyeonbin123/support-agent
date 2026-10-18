"""README의 대표 수치를 커밋된 reports/ 기록에서 다시 계산해, README 글과 같은지 확인한다.

모델을 부르지 않는다. 기록만 읽어 analyze.py·voice/metrics.py·sim_report.py·voice_real.py와 같은 계산으로
수치를 내고, README 표의 행(또는 문장 조각)이 그 값으로 쓰여 있는지 본다.
"""

# [테스트 묶음] README "이 저장소의 수치는 모두 reports/의 기록에서 다시 계산된다"를 CI가 지키게 한다.
# 단계별 대표 표(2·5·7·8·9·11·12단계)의 pass^1·pass^4, 과제 단위로 짝지은 차이와 95% 구간, 정답에 없는 쓰기,
# 통과된 규정 위반, 거짓 완료 전달 수, 본인 확인 성공률, 음성 인식 지표, 시뮬레이터 계수를 다시 계산해
# README의 행 글자와 맞춰 본다. 기록과 공개 문장이 어긋나면(수치를 고치고 문장을 빠뜨리거나 그 반대) 여기서
# 실패한다. [왜 필요한가] 측정 뒤 수치·문장 바로잡음이 여러 번 있었다(docs/experiments.md의 '바로잡음' 절들).
# 사람이 대조하던 일을 테스트로 옮긴다. README 문장을 바꾸면 이 테스트도 바꿔야 한다(의도된 마찰).
# [주의] 시험용 기록은 수치만 다시 계산하고 대화 내용을 보지 않는다(시험용 에피소드를 보며 무엇을 고치지
# 않는다는 규칙과 어긋나지 않는다). 기록 하나를 읽는 데 수백 ms라 실행마다 한 번만 읽어 둔다.
# [함께 확인하는 것] 기록된 어느 실행에서도 Ollama가 프롬프트를 조용히 잘라 답한 흔적이 없다(같은 에피소드
# 안에서 같은 역할의 프롬프트 길이가 줄어든 호출 0). 2026-10-15부터 요청에 넣은 truncate: false가 기록된
# 응답을 바꾸지 않았을 것이라는 근거다(docs/experiments.md 공통 규칙의 2026-10-15 메모).

from __future__ import annotations

import json
import re
from collections import Counter
from functools import cache
from typing import Any

import pytest

from support_agent import analyze, sim_report, voice_real
from support_agent.paths import REPORTS, ROOT
from support_agent.voice import metrics as voice_metrics

README = (ROOT / "README.md").read_text(encoding="utf-8")

# 단계별 기록 이름 (reports/ 아래 폴더). 실행 이름의 시각은 UTC다.
STAGE1_BASELINE = "20260919-102552-qwen2.5-7b-instruct-P0-R0-baseline2"
STAGE2_BASE = "20260920-203406-qwen2.5-7b-instruct-P0-R0-G0-F0-baseline3"
STAGE2_CANDIDATES = {
    "규정을 도구에서도 막음": "20260920-190957-qwen2.5-7b-instruct-P1-R0-G0-F0-p1",
    "think 도구": "20260920-193325-qwen2.5-7b-instruct-P0-R1-G0-F0-r1",
    "멈춤 가드": "20260920-195534-qwen2.5-7b-instruct-P0-R0-G1-F0-g1",
    "본문에 샌 도구 호출 구제": "20260920-201406-qwen2.5-7b-instruct-P0-R0-G0-F1-f1",
    "qwen2.5 14B": "20260920-205943-qwen2.5-14b-instruct-P0-R0-G0-F0-m1",
}
TEST_BASELINE = "20260921-011946-qwen2.5-7b-instruct-P0-R0-G0-F0-test-baseline"
TEST_M1 = "20260921-022839-qwen2.5-14b-instruct-P0-R0-G0-F0-test-m1-reference"
TEST_V1 = "20260921-155250-qwen2.5-7b-instruct-P0-R0-G0-F0-V1-test-v1"
TEST_V2 = "20260921-193331-qwen2.5-7b-instruct-P0-R0-G0-F0-V2-test-v2"
DEV_V1 = "20260921-125755-qwen2.5-7b-instruct-P0-R0-G0-F0-V1-v1"
DEV_V2 = "20260921-145634-qwen2.5-7b-instruct-P0-R0-G0-F0-V2-v2"
STAGE7_BASE = "20260921-142456-qwen2.5-7b-instruct-P0-R0-G0-F0-v0-stage5"
STAGE7_C1 = "20260921-235926-qwen2.5-7b-instruct-P0-R0-G0-F0-C1-c1"
STAGE7_C2 = "20260922-002141-qwen2.5-7b-instruct-P0-R0-G0-F0-C2-c2"
DEV_P1_7B = "20261003-185751-qwen2.5-7b-instruct-P1-R0-G0-F0-p1-stage8"
DEV_M2A = "20261003-194425-qwen3-4b-instruct-2507-q4_K_M-P1-R0-G0-F0-m2a"
DEV_M2B = "20261003-200306-qwen3.5-4b-P1-R0-G0-F0-m2b"
TEST_P1_7B = "20261003-202615-qwen2.5-7b-instruct-P1-R0-G0-F0-test-p1-stage8"
TEST_P0_7B = "20261003-210648-qwen2.5-7b-instruct-P0-R0-G0-F0-test-p0-stage8"
TEST_M2B = "20261003-214557-qwen3.5-4b-P1-R0-G0-F0-test-m2b"
DEV_V2_4B = "20261004-131932-qwen3.5-4b-P0-R0-G0-F0-V2-v2-4b-stage9-respelled"
DEV_V4 = "20261004-154016-qwen3.5-4b-P0-R0-G0-F0-V4-v4-k0-respelled"
TEST_V2_4B = "20261009-101009-qwen3.5-4b-P0-R0-G0-F0-V2-test-v2-4b-stage9-respelled"
TEST_V4 = "20261009-132408-qwen3.5-4b-P0-R0-G0-F0-V4-test-v4-k0"
STAGE11 = "20261014-033627-v4r-a-dev-respelled"
DEV_U1 = "20261008-193054-qwen3.5-4b-P1-R0-G0-F0-u1-sim"


@cache
def episodes(name: str) -> list[dict[str, Any]]:
    return analyze.load_episodes(REPORTS / name)


def by_task(name: str) -> dict[str, list[bool]]:
    return analyze.successes_by_task(episodes(name))


# README의 수 표기: 비율은 소수 한 자리 퍼센트, 부호 있는 차이는 음수에 U+2212(−)를 쓴다.
def pct(value: float) -> str:
    return f"{value:.1%}"


def signed(value: float) -> str:
    return format(value, "+.1%").rstrip("%").replace("-", "−")


def interval(diff: float, low: float, high: float) -> str:
    return f"{signed(diff)}%p [{signed(low)}, {signed(high)}]"


def pass_k(name: str, k: int) -> str:
    return pct(analyze.pass_k(by_task(name), k))


def paired(base: str, other: str, k: int = 1) -> str:
    return interval(*analyze.paired_difference(by_task(base), by_task(other), k))


def judged(name: str) -> list[dict[str, Any]]:
    return [e for e in episodes(name) if e["verdict"]]


def unexpected_writes(name: str) -> int:
    return sum(e["verdict"]["unexpected_writes"] for e in judged(name))


def passed_violations(name: str) -> int:
    return sum(len(e["verdict"]["policy_violations"]) for e in judged(name))


def format_errors(name: str) -> int:
    calls = [c for e in judged(name) for c in e["llm_calls"] if c["who"] == "agent"]
    return sum(c["format_error"] not in analyze._NOT_A_FORMAT_ERROR for c in calls)


def assert_in_readme(text: str) -> None:
    assert text in README, f"README에 이 글이 없다(기록에서 다시 계산한 값): {text!r}"


# ---------------------------------------------------------------- 1·2단계


def test_stage1_and_stage2_baselines():
    assert_in_readme(f"개발용 pass^1 {pass_k(STAGE1_BASELINE, 1)}, pass^4 {pass_k(STAGE1_BASELINE, 4)}")
    assert_in_readme(f"(pass^1 {pass_k(STAGE2_BASE, 1)}, pass^4 {pass_k(STAGE2_BASE, 4)})")


def test_stage2_development_table():
    assert_in_readme(
        f"| 기준: qwen2.5 7B, 규정은 프롬프트에만 | {pass_k(STAGE2_BASE, 1)} | "
        f"| {unexpected_writes(STAGE2_BASE)} | {passed_violations(STAGE2_BASE)} |"
    )
    for label, run in STAGE2_CANDIDATES.items():
        assert_in_readme(
            f"| {label} | {pass_k(run, 1)} | {paired(STAGE2_BASE, run)} "
            f"| {unexpected_writes(run)} | {passed_violations(run)} |"
        )


def test_stage2_test_numbers():
    low, high = analyze.bootstrap_interval(by_task(TEST_BASELINE), 1)
    assert_in_readme(
        f"pass^1 {pass_k(TEST_BASELINE, 1)} [{low * 100:.1f}, {high * 100:.1f}], "
        f"pass^4 {pass_k(TEST_BASELINE, 4)}"
    )
    assert_in_readme(f"{pass_k(TEST_M1, 1)}({paired(TEST_BASELINE, TEST_M1)})")
    assert_in_readme(f"({passed_violations(TEST_BASELINE)} → {passed_violations(TEST_M1)}건)")


# ---------------------------------------------------------------- 5단계 (음성 채널, 시험용)


def identity(name: str) -> float:
    rates = voice_metrics.identified_by_task(episodes(name))
    return sum(rates.values()) / len(rates)


def entity(name: str, kind: str) -> tuple[int, int]:
    return voice_metrics.channel_summary(episodes(name))["entities"][kind]


def test_stage5_test_table():
    assert_in_readme(f"| 글 (V0) | {pass_k(TEST_BASELINE, 1)} | | {pct(identity(TEST_BASELINE))} | | | |")
    ids, emails, phones = (entity(TEST_V1, kind) for kind in ("id", "email", "phone"))
    assert_in_readme(
        f"| 음성 채널 (V1) | {pass_k(TEST_V1, 1)} | V0보다 {paired(TEST_BASELINE, TEST_V1)} "
        f"| {pct(identity(TEST_V1))} | {ids[0]}/{ids[1]} | {emails[0]}/{emails[1]} "
        f"| {phones[0] / phones[1]:.0%} |"
    )
    ids, emails, phones = (entity(TEST_V2, kind) for kind in ("id", "email", "phone"))
    assert_in_readme(
        f"| 음성 채널 + 정규화 (V2) | {pass_k(TEST_V2, 1)} | V1보다 {paired(TEST_V1, TEST_V2)} "
        f"| {pct(identity(TEST_V2))} | {ids[0] / ids[1]:.0%} | {emails[0] / emails[1]:.0%} "
        f"| {phones[0] / phones[1]:.0%} |"
    )


def test_stage5_sentences():
    cer = voice_metrics.channel_summary(episodes(TEST_V1))["cer"]
    assert_in_readme(f"인식 결과의 글자 오류율은 {cer:.1%}인데")
    diff, _, _ = analyze.paired_difference(by_task(DEV_V1), by_task(DEV_V2), 1)
    test_diff, _, _ = analyze.paired_difference(by_task(TEST_V1), by_task(TEST_V2), 1)
    assert_in_readme(f"개발용({signed(diff)}%p)과 시험용({signed(test_diff)}%p)에서 같은 방향이다")


# ---------------------------------------------------------------- 7단계 (완료 주장 가드, 개발용)


def claims(name: str) -> tuple[int, int, int]:
    """(고객에게 전달된 거짓 완료 응답, 가드가 돌려보낸 응답, 돌려보낸 뒤 도구 호출)."""
    counted = [e for e in episodes(name) if e["status"] != "infra_error"]
    told = sum(len(analyze.delivered_claims(e)) for e in counted)
    held = sum(c["format_error"] == "unbacked_claim" for e in counted for c in e["llm_calls"])
    after = Counter(step for e in counted for step in analyze.after_a_held_claim(e))
    return told, held, after["write"] + after["write_failed"] + after["read"]


def test_stage7_development_table():
    told, _, _ = claims(STAGE7_BASE)
    assert_in_readme(f"| 기준 | {pass_k(STAGE7_BASE, 1)} | | {unexpected_writes(STAGE7_BASE)} | {told} | |")
    for label, run in (
        ("C1: 돌려보내고 다시 시킴", STAGE7_C1),
        ("C2: 다시 시킬 때 temperature 0.7", STAGE7_C2),
    ):
        told, held, tools = claims(run)
        assert_in_readme(
            f"| {label} | {pass_k(run, 1)} | {paired(STAGE7_BASE, run)} | {unexpected_writes(run)} "
            f"| {told} | {tools} / {held} |"
        )


# ---------------------------------------------------------------- 8단계 (새 세대 소형 모델)


def agent_call_ms(name: str) -> str:
    calls = [c for e in episodes(name) if e["status"] != "infra_error" for c in e["llm_calls"]]
    agent = [c["wall_ms"] for c in calls if c["who"] == "agent"]
    return f"{round(sum(agent) / len(agent), 1):.0f} ms"


def other_script(name: str) -> str:
    found, total = analyze._other_script(episodes(name))
    return pct(found / total)


def test_stage8_development_table():
    assert_in_readme(
        f"| 기준: qwen2.5 7B | {pass_k(DEV_P1_7B, 1)} | | {pass_k(DEV_P1_7B, 4)} "
        f"| {unexpected_writes(DEV_P1_7B)} | {format_errors(DEV_P1_7B)} | {agent_call_ms(DEV_P1_7B)} |"
    )
    for label, run in (("`qwen3:4b-instruct-2507`", DEV_M2A), ("`qwen3.5:4b` (비사고 모드)", DEV_M2B)):
        assert_in_readme(
            f"| {label} | {pass_k(run, 1)} | {paired(DEV_P1_7B, run)} | {pass_k(run, 4)} "
            f"| {unexpected_writes(run)} | {format_errors(run)} | {agent_call_ms(run)} |"
        )


def test_stage8_test_table():
    for label, run in (
        ("기준: qwen2.5 7B, P1", TEST_P1_7B),
        ("참조: qwen2.5 7B, P0 (규정은 프롬프트에만)", TEST_P0_7B),
    ):
        assert_in_readme(
            f"| {label} | {pass_k(run, 1)} | | {pass_k(run, 4)} | {unexpected_writes(run)} "
            f"| {passed_violations(run)} | {other_script(run)} |"
        )
    assert_in_readme(
        f"| `qwen3.5:4b`, P1 | **{pass_k(TEST_M2B, 1)}** | **{paired(TEST_P1_7B, TEST_M2B)}** "
        f"| {pass_k(TEST_M2B, 4)} | {unexpected_writes(TEST_M2B)} | {passed_violations(TEST_M2B)} "
        f"| {other_script(TEST_M2B)} |"
    )


# ---------------------------------------------------------------- 9단계 (음성 V4, 송장 번호를 고친 기록)


def identity_difference(base: str, other: str) -> str:
    rates = (voice_metrics.identified_by_task(episodes(run)) for run in (base, other))
    return interval(*voice_metrics.paired_mean_difference(*rates))


@pytest.mark.parametrize(
    ("v2", "v4", "bold_pass1"), [(DEV_V2_4B, DEV_V4, False), (TEST_V2_4B, TEST_V4, True)]
)
def test_stage9_tables(v2, v4, bold_pass1):
    assert_in_readme(
        f"| V2: 받아 적은 이름과 연락처 | {pct(identity(v2))} | | {pass_k(v2, 1)} | "
        f"| {unexpected_writes(v2)} | {passed_violations(v2)} |"
    )
    pass1, diff = pass_k(v4, 1), paired(v2, v4)
    if bold_pass1:
        pass1, diff = f"**{pass1}**", f"**{diff}**"
    assert_in_readme(
        f"| V4: 발신 번호 + 이름 정확히 | **{pct(identity(v4))}** | **{identity_difference(v2, v4)}** "
        f"| {pass1} | {diff} | {unexpected_writes(v4)} | {passed_violations(v4)} |"
    )


# ---------------------------------------------------------------- 11단계 (전화 통화 조건, 인식만)


def stage11_rows() -> dict[str, tuple[str, str, str, str]]:
    """voice_real.report 표의 (지표 이름 → 깨끗한 칸, 열화한 칸, 차이, 구간) 그대로."""
    path = REPORTS / STAGE11 / "utterances.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = {}
    for line in voice_real.report(records).splitlines():
        found = re.match(
            r"\| (.+?) \| (.+?) \| (.+?) \| ([+-][\d.]+)%p \[([+-][\d.]+), ([+-][\d.]+)\] \|$", line
        )
        if found:
            rows[found.group(1)] = found.groups()[1:]
    return rows


def test_stage11_table():
    rows = stage11_rows()

    def spread(row: tuple[str, ...]) -> str:  # 보고서의 자릿수 그대로, 음수 기호만 README 표기(−)로
        diff, low, high = (x.replace("-", "−") for x in row[2:])
        return f"{diff}%p [{low}, {high}]"

    def share_first(cell: str) -> str:  # "99/165 (60.0%)" → "60.0% (99/165)"
        counts, share = re.match(r"(\d+/\d+) \((.+)\)", cell).groups()
        return f"{share} ({counts})"

    cer = rows["글자 오류율"]
    assert_in_readme(f"| 글자 오류율 | {cer[0]} | {cer[1]} | {spread(cer)} |")
    name = rows["이름 (정확히, k = 0)"]
    assert_in_readme(
        f"| 이름 정확 생존 (V4 본인 확인의 기준) | {share_first(name[0])} | {share_first(name[1])} "
        f"| {spread(name)} |"
    )
    ids = rows["주문·접수 번호"]
    assert_in_readme(
        f"| 주문·접수 번호 생존 | {share_first(ids[0])} | {share_first(ids[1])} | {spread(ids)} |"
    )


# ---------------------------------------------------------------- 12단계 (시뮬레이터 민감도)


def test_stage12_table():
    for label, run, base in (
        ("고객 qwen2.5 7B (8단계 기록)", DEV_M2B, None),
        ("고객 `qwen3:4b-instruct-2507`", DEV_U1, DEV_M2B),
    ):
        counts = sim_report.run_counts(REPORTS / run)
        endings = Counter(e["termination"] for e in episodes(run))
        diff = f" {paired(base, run)} " if base else " "
        assert_in_readme(
            f"| {label} | {pass_k(run, 1)} |{diff}| {pass_k(run, 4)} | {endings['user_stop']} "
            f"| {endings['max_user_turns']} | {counts['stop_before_values']}/{counts['values_never_heard']} "
            f"| {counts['invented_utterances']} |"
        )


# ---------------------------------------------------------------- 13단계 (4B 에이전트의 가드 네 팔)

STAGE13_DEV = {
    "C1: 근거 없는 완료 주장을 돌려보냄": "20261018-021911-qwen3.5-4b-P1-R0-G0-F0-C1-s13-c1",
    'G1: "확인해 보겠습니다"만 하는 답을 돌려보냄': "20261018-024318-qwen3.5-4b-P1-R0-G1-F0-s13-g1",
    "E1: 규정 거절 오류에 검사가 쓴 사실을 붙임": "20261018-030602-qwen3.5-4b-P1-R0-G0-F0-E1-s13-e1",
    "D1: 바로 앞 답의 되풀이를 한 번 다시 시킴": "20261018-032917-qwen3.5-4b-P1-R0-G0-F0-D1-s13-d1",
}
STAGE13_TEST = "20261018-035645-qwen3.5-4b-P1-R0-G0-F0-E1-s13-test"


# 개발용 표: pass^1, 8단계 M2b와 짝지은 차이, 정답에 없는 쓰기, 고객에게 간 거짓 완료, 선택 조건(13단계 규칙,
# guard_report.dev_conditions)을 다시 계산한다.
def test_stage13_development_table():
    from support_agent import guard_report

    base = guard_report.run_stats(REPORTS / DEV_M2B)
    assert_in_readme(
        f"| 기준: 8단계 `qwen3.5:4b` | {pass_k(DEV_M2B, 1)} | | {unexpected_writes(DEV_M2B)} "
        f"| {claims(DEV_M2B)[0]} | |"
    )
    for label, run in STAGE13_DEV.items():
        checks = guard_report.dev_conditions(base, guard_report.run_stats(REPORTS / run), label[:2])
        verdict = "만족" if all(ok for _, ok in checks) else "불만족"
        assert_in_readme(
            f"| {label} | {pass_k(run, 1)} | {paired(DEV_M2B, run)} | {unexpected_writes(run)} "
            f"| {claims(run)[0]} | {verdict} |"
        )


# 시험용 표와 판정: 고른 구성(E1) 한 번. 거절 유형과 그 밖 세 유형의 pass^1을 따로 낸다.
def test_stage13_test_row_and_verdict():
    from support_agent import guard_report

    stats = {run: guard_report.run_stats(REPORTS / run) for run in (TEST_M2B, STAGE13_TEST)}
    for label, run in (("기준: 8단계 `qwen3.5:4b`", TEST_M2B), ("E1", STAGE13_TEST)):
        diff = f" {paired(TEST_M2B, run)} " if run != TEST_M2B else " "
        assert_in_readme(
            f"| {label} | {pass_k(run, 1)} |{diff}| {pct(stats[run]['by_kind']['refusal'])} "
            f"| {pct(stats[run]['not_refusal'])} | {unexpected_writes(run)} |"
        )
    _, result = guard_report.verdict(REPORTS / TEST_M2B, REPORTS / STAGE13_TEST)
    assert_in_readme(f"**판정: {result}.**")


# ---------------------------------------------------------------- 기록된 프롬프트가 잘린 흔적이 없다


# 잘림이 있었다면 Ollama가 잘린 길이를 prompt_eval_count로 보고해, 같은 에피소드 안에서 같은 역할의 프롬프트
# 길이가 앞 호출보다 줄어든다(대화는 덧붙기만 하므로 정상이면 줄지 않는다). 또 기록된 프롬프트는 모두 그
# 실행의 num_ctx보다 짧다. 그래서 truncate: false를 보냈더라도 기록된 응답은 같았다.
@pytest.mark.parametrize("run", sorted(p.name for p in REPORTS.iterdir() if (p / "episodes.jsonl").exists()))
def test_no_recorded_prompt_was_silently_truncated(run):
    config = json.loads((REPORTS / run / "manifest.json").read_text(encoding="utf-8"))["config"]
    limits = {"agent": config["num_ctx"], "user": config.get("user_num_ctx") or config["num_ctx"]}
    for episode in episodes(run):
        longest = {"agent": 0, "user": 0}
        for call in episode["llm_calls"]:
            tokens, who = call["prompt_tokens"] or 0, call["who"]
            if not tokens:
                continue
            assert tokens >= longest[who], (episode["task_id"], episode["trial"], who, longest[who], tokens)
            assert tokens < limits[who]
            longest[who] = tokens
