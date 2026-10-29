"""README와 docs/experiments.md의 기록 수치를 커밋된 reports/ 기록에서 다시 계산해, 글과 같은지 확인한다.

모델을 부르지 않는다. 기록만 읽어 analyze.py·voice/metrics.py·sim_report.py·voice_real.py·guard_report.py와
같은 계산으로 수치를 내고, 표의 행(또는 문장 조각)이 그 값으로 쓰여 있는지 본다.
"""

# [테스트 묶음] README "이 저장소의 수치는 모두 reports/의 기록에서 다시 계산된다"를 CI가 지키게 한다.
# 단계별 대표 표(2·5·7·8·9·11·12·13단계)의 pass^1·pass^4, 과제 단위로 짝지은 차이와 95% 구간, 정답에
# 없는 쓰기, 통과된 규정 위반, 거짓 완료 전달 수, 본인 확인 성공률, 음성 인식 지표, 시뮬레이터 계수와,
# 표 아래 문장 가운데 고른 것(1·2단계 기준선, 2단계 시험용, 5단계, 13단계 B·판정·D1)의 수를 다시 계산해
# README의 글자와 맞춰 본다. docs/experiments.md에서는 13단계 결과의 보고만 하는 지표 표와 D1 '읽는 법'의
# 수를 맞춰 본다. 기록과 공개 문장이 어긋나면(수치를 고치고 문장을 빠뜨리거나 그 반대) 여기서 실패한다.
# 여기서 맞춰 보지 않는 문장 속 수치는 CI가 지키지 않는다(README "다시 재는 방법"에 그 범위를 적었다).
# [왜 필요한가] 측정 뒤 수치·문장 바로잡음이 여러 번 있었다(docs/experiments.md의 '바로잡음' 절들).
# 사람이 대조하던 일을 테스트로 옮긴다. README 문장을 바꾸면 이 테스트도 바꿔야 한다(의도된 마찰).
# [주의] 시험용 기록은 수치만 다시 계산하고 대화 내용을 보지 않는다(시험용 에피소드를 보며 무엇을 고치지
# 않는다는 규칙과 어긋나지 않는다). 기록 하나를 읽는 데 수백 ms라 실행마다 한 번만 읽어 둔다.
# [함께 확인하는 것] 요청에 truncate 필드가 없던 2026-10-15 전 실행에서 Ollama가 프롬프트를 조용히
# 잘랐을 수 있는 호출을 센다(근거는 맨 아래 절). num_ctx 8192로 잰 2단계 14B 두 기록 말고는 잘림이 있을
# 수 없다.

from __future__ import annotations

import json
import re
from collections import Counter
from functools import cache
from itertools import pairwise
from typing import Any

import pytest

from support_agent import analyze, sim_report, voice_real
from support_agent.paths import REPORTS, ROOT
from support_agent.voice import metrics as voice_metrics

README = (ROOT / "README.md").read_text(encoding="utf-8")
# 측정 규칙과 결과의 원본. 13단계 보고 지표 표처럼 README에 옮기지 않은 기록 수치를 여기서 맞춰 본다.
EXPERIMENTS = (ROOT / "docs" / "experiments.md").read_text(encoding="utf-8")

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


def assert_in_experiments(text: str) -> None:
    assert text in EXPERIMENTS, f"docs/experiments.md에 이 글이 없다(기록에서 다시 계산한 값): {text!r}"


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


# 13단계 B(보고만): 7B 기준을 qwen3 4B 고객으로 한 번. 7B가 고객 모델에 기대는 차이와, 그 고객에서
# qwen3.5:4b − 7B의 차이를 README 12단계 절의 문장과 맞춘다.
def test_stage13b_sentence():
    b = "20261018-051104-qwen2.5-7b-instruct-P1-R0-G0-F0-p1-7b-u1"
    assert_in_readme(f"7B는 {pass_k(DEV_P1_7B, 1)} → {pass_k(b, 1)} ({paired(DEV_P1_7B, b)})")
    assert_in_readme(f"`qwen3.5:4b` − 7B는 {paired(b, DEV_U1)}이다")


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


def held_repeats_answered_again(name: str) -> tuple[int, int]:
    """(다음 에이전트 호출이 이어진 붙잡은 되풀이 수, 그 다음 호출이 도구 호출 없이 같은 글을 낸 수)."""
    followed = same = 0
    for episode in episodes(name):
        if episode["status"] == "infra_error":
            continue
        calls = [c for c in episode["llm_calls"] if c["who"] == "agent"]
        for held, after in pairwise(calls):
            if held["format_error"] == "repeat":
                followed += 1
                if not after["tool_calls"] and after["text"] == held["text"]:
                    same += 1
    return followed, same


# D1(바로 앞 답의 되풀이를 한 번 다시 시킴) 문장의 세 수. README 13단계 절과 docs/experiments.md
# 13단계 '읽는 법'에 같은 수가 있다.
# [무엇을 세나] 붙잡은 답(format_error "repeat") 가운데 다음 에이전트 호출이 이어진 것. 붙잡은 호출이
# 에피소드의 마지막 호출이면 셀 다음이 없다(보고 표의 '붙잡은 되풀이'는 그것까지 센 수라 이것보다 크다).
# 그 다음 호출이 도구 호출 없이 글자까지 같은 글을 낸 것(이 기록에서는 공백만 빼고 비교하는 가드의
# is_repeat로 세도 같다). LLM 호출 한도(max_agent_calls)로 끝난 에피소드(기준 8단계 M2b와 함께).
# infra_error 에피소드는 세지 않는다.
def test_stage13_d1_sentences():
    d1 = STAGE13_DEV["D1: 바로 앞 답의 되풀이를 한 번 다시 시킴"]
    followed, same = held_repeats_answered_again(d1)
    limit = analyze.load_manifest(REPORTS / d1)["config"]["max_agent_calls"]  # 에피소드당 LLM 호출 한도
    capped = {
        run: sum(e["termination"] == "max_agent_calls" for e in episodes(run) if e["status"] != "infra_error")
        for run in (DEV_M2B, d1)
    }
    assert_in_readme(
        f"붙잡힌 답 {followed}번 중 {same}번 같은 답을 다시 냈고, "
        f"그만큼 호출이 늘어 {capped[d1]}개 에피소드가 LLM 호출 한도로 끝났다"
    )
    assert_in_experiments(f"붙잡은 답 {followed}번 뒤 다시 낸 답이 {same}번 글자까지 같았다")
    assert_in_experiments(
        f"{capped[d1]}개 에피소드가 에피소드당 LLM 호출 한도({limit})에 닿아 "
        f"`max_agent_calls`(실패)로 끝났다 (기준 {capped[DEV_M2B]})"
    )
    assert_in_experiments(f"(D1 {same}/{followed})")


# docs/experiments.md 13단계 결과의 보고만 하는 지표 표. 개발용 기준·네 팔과 시험용 기준·E1의 일곱 행이고
# 열은 전달된 거짓 완료, 같은 답 되풀이로 끝남, 전달된 멈춤 답, 붙잡은 완료 주장·멈춤·되풀이, E1 사실을 붙인
# 거절, 에피소드당 초, 에이전트 호출 ms, 한자 섞인 답, infra다. 이 표는 analyze guards·guards-verdict가 낸
# 것을 옮겼다고 적혀 있으므로, 같은 셈(guard_report.run_stats)으로 행 전체를 다시 만들어 그 글자를 찾는다.
@pytest.mark.parametrize("run", [DEV_M2B, *STAGE13_DEV.values(), TEST_M2B, STAGE13_TEST])
def test_stage13_reported_only_rows(run):
    from support_agent import guard_report

    s = guard_report.run_stats(REPORTS / run)
    assert_in_experiments(
        f"| {run} | {s['false_claims']} | {s['loops']} | {s['stall_replies']} | {s['held_claims']} "
        f"| {s['held_stalls']} | {s['held_repeats']} | {s['e1_facts']} | {s['seconds']:.1f} "
        f"| {s['agent_ms']:.0f} | {s['other_script']} | {s['infra']} |"
    )


# ---------------------------------------------------------------- 기록된 프롬프트가 조용히 잘렸을 수 있는가

# [잘림 모형] 2026-10-15 전의 실행은 Ollama 요청에 truncate 필드가 없었다. 그러면 Ollama 0.35.1
# (server/prompt.go의 chatPrompt)은 프롬프트가 num_ctx를 넘을 때 오류 없이, 시스템 메시지를 두고 앞
# 메시지부터 빼서 num_ctx 안에 드는 가장 긴 메시지 접미사만 모델에 넣는다. 그래서 잘린 호출의 보고 길이
# (prompt_eval_count)는 num_ctx 가까이(빠진 메시지 하나의 길이 안쪽) 남고, 앞 호출보다 짧아진다는 보장이
# 없다. 같은 역할의 길이가 줄지 않았다는 것만으로는 잘림을 배제하지 못한다(docs/experiments.md 2026-10-15
# 메모의 2026-10-29 바로잡음).
# [근거] 대화는 덧붙기만 하므로 어느 호출이 잘리려면 바로 앞 호출의 길이 + 한 번에 늘어난 길이가 num_ctx를
# 넘어야 한다. 한 번에 늘어난 길이는 이 기록들에서 최대 MAX_STEP이므로(아래 테스트가 그 전제를 확인한다),
# 앞 호출이 num_ctx − MAX_STEP 이하이면 다음 호출은 잘릴 수 없다. 에이전트는 프롬프트가 num_ctx의 95%를 넘은
# 호출에서 에피소드를 끝내므로(agent.py의 context_limit), 잘림을 배제하지 못하는 것은 길이가
# (num_ctx − MAX_STEP, 0.95·num_ctx] 띠에 든 호출의 바로 다음 호출뿐이다. 95%를 넘은 호출 자체는 잘렸든
# 아니든 truncate: false에서도 context_limit 종료라 결과가 같다. 시뮬레이터에는 95% 검사가 없으므로
# 시뮬레이터의 띠는 (한도 − MAX_STEP, 한도)다.
# [범위] 2026-10-15부터의 실행은 truncate: false를 보냈고(manifest의 제공자 설명에 남음), Ollama는 넘친
# 요청을 잘라 답하지 않고 HTTP 400으로 거절하므로 띠 셈이 필요 없다. 그 전 실행 36개 가운데 띠에 호출이 든
# 것은 num_ctx 8192로 잰 2단계 14B 두 기록뿐이고, 그 수를 기록값으로 고정한다. 두 기록은 판정을 가르지
# 않았다: 개발용 m1은 호출당 시간 조건(7.8배, 3배 이하)만으로도 채택하지 않는 2단계 후보이고, 시험용은
# 판정과 무관한 참고 측정이다.

# 기록에서 본, 같은 에피소드·같은 역할의 한 호출에서 다음 호출로 늘어난 프롬프트 길이의 최대값(토큰).
MAX_STEP = {"agent": 1075, "user": 494}
# 실행 → (띠에 든 에이전트 호출 수, 띠 호출 다음에 같은 역할의 호출이 이어진 에피소드 수).
NOT_EXCLUDED = {STAGE2_CANDIDATES["qwen2.5 14B"]: (6, 2), TEST_M1: (20, 6)}


@pytest.mark.parametrize("run", sorted(p.name for p in REPORTS.iterdir() if (p / "episodes.jsonl").exists()))
def test_no_recorded_prompt_was_silently_truncated(run):
    manifest = json.loads((REPORTS / run / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    limits = {"agent": config["num_ctx"], "user": config.get("user_num_ctx") or config["num_ctx"]}
    # 두 제공자 모두 truncate: false를 보낸 실행: 넘친 요청은 잘리지 않고 거절됐다(띠 셈을 하지 않음).
    sides = ("agent_provider", "user_provider")
    refused = all((manifest.get(side) or {}).get("truncate") is False for side in sides)
    # 역할 → 길이가 한도 − MAX_STEP을 넘은 호출 수(에이전트는 95%를 넘은 호출도 든다).
    near: Counter[str] = Counter()
    band = 0  # 띠 (num_ctx − MAX_STEP, 0.95·num_ctx]에 든 에이전트 호출 수
    # 한도 − MAX_STEP을 넘은 호출 다음에 같은 역할의 호출이 이어진 에피소드 (과제, 시도).
    exposed: set[tuple[str, int]] = set()
    for episode in episodes(run):
        key = (episode["task_id"], episode["trial"])
        last = {"agent": 0, "user": 0}
        for call in episode["llm_calls"]:
            tokens, who = call["prompt_tokens"] or 0, call["who"]
            if not tokens:  # 길이를 보고하지 않은 호출은 건너뛴다(증가는 그 앞뒤 호출 사이로 잰다)
                continue
            # 모든 실행의 기록 점검: 같은 역할의 보고 길이는 줄지 않고 한도보다 짧다(이것만으로는 잘림을
            # 배제하지 못한다).
            assert last[who] <= tokens < limits[who], (*key, who, last[who], tokens)
            if not refused:
                margin = limits[who] - MAX_STEP[who]
                assert not last[who] or tokens - last[who] <= MAX_STEP[who], (*key, who, last[who], tokens)
                if last[who] > margin:
                    exposed.add(key)
                if tokens > margin:
                    near[who] += 1
                    if who == "agent" and tokens <= 0.95 * limits["agent"]:
                        band += 1
            last[who] = tokens
    if refused:
        return
    assert near["user"] == 0, near
    assert (band, len(exposed)) == NOT_EXCLUDED.get(run, (0, 0))
    if run not in NOT_EXCLUDED:  # 띠 위(95% 초과)의 에이전트 호출도 없어야 한다
        assert near["agent"] == 0, near
