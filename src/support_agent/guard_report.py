"""13단계(4B 에이전트의 남은 실패를 겨냥한 가드) 분석: 개발용 선택과 시험용 판정. 모델을 부르지 않는다.

사용법:
    uv run python -m support_agent.analyze guards reports/<M2b 개발용> reports/<팔 개발용> [...]
    uv run python -m support_agent.analyze guards-verdict reports/<M2b 시험용> reports/<고른 구성 시험용>
"""

# [파일 역할] 13단계 규칙(docs/experiments.md 13단계, 측정 전에 커밋)을 그대로 코드로 옮긴 분석. 팔(C1, G1,
# E1, D1, 합친 팔)의 개발용 기록을 8단계 qwen3.5:4b 개발용 기록(기준)과 과제 단위로 짝지어 선택 조건을
# 계산하고, 고른 구성의 시험용 기록으로 판정을 낸다. 보고만 하는 지표(유형별 pass^1, 전달된 거짓 완료, 같은 답
# 되풀이로 끝난 에피소드, 전달된 멈춤 답, 가드 발동 수, 시간)도 함께 낸다.
# [설계 이유] 문턱값과 판정표를 측정 전에 상수로 고정해, 결과를 본 뒤 기준을 바꿀 수 없게 한다(이 프로젝트의
# 측정 방식). 짝짓기 검사는 analyze.setup_differences를 그대로 쓰고 가드 축 넷만 달라도 되게 한다.
# [주의] E1은 거절 과제의 판정 값(수령일·출고일 같은 날짜)과 같은 종류의 사실을 도구 오류 글에 붙인다. 그래서
# 거절 유형은 따로 보고하고, 거절이 아닌 세 유형의 pass^1이 기준보다 낮으면(한 에피소드라도) E1을 고르지
# 않는다. 시험용 기록으로는 판정만 하고 무엇도 고치지 않는다.
# [연결] analyze.py의 guards·guards-verdict 명령이 부른다. 이 모듈은 analyze.py의 함수를 가져다 쓰므로
# analyze.py는 이 모듈을 함수 안에서 가져온다.

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from support_agent.agent import is_stall
from support_agent.analyze import (
    _other_script,
    delivered_claims,
    delivered_replies,
    load_episodes,
    load_manifest,
    load_task_types,
    paired_difference,
    pass_k,
    setup_differences,
    successes_by_task,
)
from support_agent.config import FIRST_AGENT_MESSAGE, RunConfig

# 13단계가 바꾸는 축(설정 이름)과 기본값. 이 넷 말고는 기준과 같아야 짝지을 수 있다.
GUARD_AXES = {"claims": "C0", "guard": "G0", "errors": "E0", "repeat": "D0"}
# 합친 팔에서 같은 성공률일 때 고르는 순서(작은 변경부터): 이미 있던 가드(C1, G1), 새 루프 가드(D1), 도구 글을
# 바꾸는 E1(판정 값과 같은 종류의 사실을 주므로 마지막).
TIE_ORDER = ("C1", "G1", "D1", "E1")
# 개발용 선택 조건(13단계 규칙): pass^1 차이 점추정 +3.1%p(96 에피소드 중 3) 이상, 정답에 없는 쓰기가 기준 + 2
# 이하, 통과된 규정 위반 0, 형식 오류·근거 없는 완료 주장으로 끝난 에피소드 합 2 이하.
MIN_GAIN = 3 / 96
DEV_WRITE_SLACK = 2
MAX_FORMAT_AND_CLAIM_ENDINGS = 2
# 시험용 판정: 정답에 없는 쓰기가 기준 + 3 이하(8단계 시험용 13 → 16).
TEST_WRITE_SLACK = 3
# 서비스의 CLAIMS=C1 기본값 조건(7단계 규칙을 M2b로 다시): pass^1 차이 −5%p 이상, 정답에 없는 쓰기 기준 이하.
C1_SERVICE_MIN_DIFF = -0.05
# E1이 사실을 붙이는 규정 오류 코드(tools.py의 고정 목록). 가드 발동 수를 셀 때 쓴다.
E1_CODES = frozenset(
    {
        "return_window_expired",
        "exchange_window_expired",
        "return_not_delivered",
        "exchange_not_delivered",
        "cancel_not_allowed_status",
        "address_change_not_allowed_status",
        "coupon_already_issued_for_order",
        "coupon_limit_exceeded",
        "coupon_not_eligible",
    }
)
_KINDS = ("lookup", "action", "refusal", "composite")
_FLOAT_SLACK = 1e-9  # 3/96 같은 경계값을 부동소수 오차로 놓치지 않게


# 실행의 팔 이름: 기본값이 아닌 가드 축 코드를 축 순서(C, G, E, D)로 이어 쓴 것(예: "C1", "C1+D1"). 모두
# 기본값이면 "기준".
def arm_of(run_dir: Path) -> str:
    config = {**RunConfig().to_dict(), **load_manifest(run_dir)["config"]}
    return arm_name([config[axis] for axis, default in GUARD_AXES.items() if config[axis] != default])


# 축 코드 목록을 축 순서(C, G, E, D)로 이어 쓴 팔 이름. 빈 목록이면 "기준".
def arm_name(codes: list[str]) -> str:
    order = "CGED"
    return "+".join(sorted(codes, key=lambda code: order.index(code[0]))) or "기준"


# 기준과 팔이 가드 축 넷 말고 같은 설정인가(analyze same-setup과 같은 검사). 다르면 그 이유들.
def pairing_problems(base_dir: Path, cand_dir: Path) -> list[str]:
    return setup_differences(base_dir, cand_dir, ignore=tuple(GUARD_AXES))


# 같은 답 되풀이로 끝난 에피소드: 고객에게 전달된 상담원 답(고정 첫 인사 제외)의 마지막 셋이 공백만 빼고 같다.
# 감사에서 M2b 개발용 실패 40개 중 15개가 이 꼴이었다(13단계 "왜 하는가").
def ends_in_a_loop(episode: dict[str, Any]) -> bool:
    replies = [" ".join(r.split()) for r in delivered_replies(episode) if r != FIRST_AGENT_MESSAGE]
    return len(replies) >= 3 and len(set(replies[-3:])) == 1


# 기록 하나의 보고용 수치. 판정된 에피소드(infra 제외)만 센다.
def run_stats(run_dir: Path) -> dict[str, Any]:
    episodes = [e for e in load_episodes(run_dir) if e["status"] != "infra_error"]
    types = load_task_types()
    by_task = successes_by_task(episodes)
    agent_calls = [c for e in episodes for c in e["llm_calls"] if c["who"] == "agent"]
    errors_on = load_manifest(run_dir)["config"].get("errors") == "E1"
    endings = Counter(e["termination"] for e in episodes)
    found, replies = _other_script(episodes)
    return {
        "by_task": by_task,
        "pass1": pass_k(by_task, 1),
        "pass4": pass_k(by_task, 4),
        "by_kind": {
            kind: pass_k({t: v for t, v in by_task.items() if types.get(t) == kind}, 1) for kind in _KINDS
        },
        "not_refusal": pass_k({t: v for t, v in by_task.items() if types.get(t) != "refusal"}, 1),
        "unexpected_writes": sum(e["verdict"]["unexpected_writes"] for e in episodes if e["verdict"]),
        "violations": sum(len(e["verdict"]["policy_violations"]) for e in episodes if e["verdict"]),
        "endings": endings,
        "format_and_claim_endings": endings["agent_format_error"] + endings["unbacked_claim"],
        "false_claims": sum(len(delivered_claims(e)) for e in episodes),
        "loops": sum(ends_in_a_loop(e) for e in episodes),
        "stall_replies": sum(is_stall(r) for e in episodes for r in delivered_replies(e)),
        "held_claims": sum(c["format_error"] == "unbacked_claim" for c in agent_calls),
        "held_stalls": sum(c["format_error"] == "stall" for c in agent_calls),
        "held_repeats": sum(c["format_error"] == "repeat" for c in agent_calls),
        "e1_facts": sum(
            bool(t["policy_blocked"] and t["error_code"] in E1_CODES)
            for e in episodes
            for t in e["tool_calls"]
        )
        if errors_on
        else 0,
        "seconds": sum(e["wall_seconds"] for e in episodes) / len(episodes) if episodes else 0.0,
        "agent_ms": sum(c["wall_ms"] for c in agent_calls) / len(agent_calls) if agent_calls else 0.0,
        "other_script": f"{found}/{replies}",
        "infra": sum(e["status"] == "infra_error" for e in load_episodes(run_dir)),
    }


def _pp(value: float) -> str:
    return f"{value:+.1%}p"


# 기준과 짝지은 pass^1 차이(점추정, 95% 구간 아래·위).
def _difference(base: dict[str, Any], cand: dict[str, Any]) -> tuple[float, float, float]:
    return paired_difference(base["by_task"], cand["by_task"], 1)


# 개발용 선택 조건 다섯(E1이 든 팔만 다섯째). (조건 이름, 만족 여부) 목록.
def dev_conditions(base: dict[str, Any], cand: dict[str, Any], arm: str) -> list[tuple[str, bool]]:
    diff, _, _ = _difference(base, cand)
    checks = [
        (f"pass^1 차이 ≥ {_pp(MIN_GAIN)}", diff >= MIN_GAIN - _FLOAT_SLACK),
        (
            f"정답에 없는 쓰기 ≤ {base['unexpected_writes'] + DEV_WRITE_SLACK}",
            cand["unexpected_writes"] <= base["unexpected_writes"] + DEV_WRITE_SLACK,
        ),
        ("통과된 규정 위반 0", cand["violations"] == 0),
        (
            f"형식 오류·완료 주장 종료 ≤ {MAX_FORMAT_AND_CLAIM_ENDINGS}",
            cand["format_and_claim_endings"] <= MAX_FORMAT_AND_CLAIM_ENDINGS,
        ),
    ]
    if "E1" in arm.split("+"):
        checks.append(
            ("거절이 아닌 유형 pass^1 ≥ 기준", cand["not_refusal"] >= base["not_refusal"] - _FLOAT_SLACK)
        )
    return checks


# 보고 표 두 개: 수치 표와 보고만 하는 지표 표.
def _tables(names: list[str], arms: dict[str, str], stats: dict[str, dict[str, Any]]) -> list[str]:
    base = stats[names[0]]
    lines = [
        "| 실행 | 팔 | pass^1 | 기준과의 차이 [95% 구간] | pass^4 | 조회 | 처리 | 거절 | 복합 "
        "| 거절 아닌 유형 | 정답에 없는 쓰기 | 통과된 위반 | 형식 오류·완료 주장 종료 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        s = stats[name]
        diff = "" if name == names[0] else "{} [{}, {}]".format(*(_pp(x) for x in _difference(base, s)))
        kinds = " | ".join(f"{s['by_kind'][k]:.1%}" for k in _KINDS)
        lines.append(
            f"| {name} | {arms[name]} | {s['pass1']:.1%} | {diff} | {s['pass4']:.1%} | {kinds} "
            f"| {s['not_refusal']:.1%} | {s['unexpected_writes']} | {s['violations']} "
            f"| {s['format_and_claim_endings']} |"
        )
    lines += [
        "",
        "| 실행 | 전달된 거짓 완료 | 같은 답 되풀이로 끝남 | 전달된 멈춤 답 | 붙잡은 완료 주장 | 붙잡은 멈춤 "
        "| 붙잡은 되풀이 | E1 사실을 붙인 거절 | 에피소드당 초 | 에이전트 호출 ms | 한자 섞인 답 | infra |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        s = stats[name]
        lines.append(
            f"| {name} | {s['false_claims']} | {s['loops']} | {s['stall_replies']} | {s['held_claims']} "
            f"| {s['held_stalls']} | {s['held_repeats']} | {s['e1_facts']} | {s['seconds']:.1f} "
            f"| {s['agent_ms']:.0f} | {s['other_script']} | {s['infra']} |"
        )
    return lines


# 개발용 선택(13단계 규칙). 첫 실행이 기준(8단계 M2b 개발용), 나머지가 팔. 단일 팔(가드 축 하나)과 합친
# 팔(둘 이상)을 함께 받는다. 돌려주는 것: 보고서 글과 고른 구성(없으면 None, 합친 팔이 필요한데 없으면
# "need:<팔>").
def select(base_dir: Path, arm_dirs: list[Path]) -> tuple[str, str | None]:
    names = [base_dir.name, *(d.name for d in arm_dirs)]
    arms = {base_dir.name: arm_of(base_dir), **{d.name: arm_of(d) for d in arm_dirs}}
    problems = [f"- {d.name}: {p}" for d in arm_dirs for p in pairing_problems(base_dir, d)]
    if problems:
        return "짝지을 수 없음 (가드 축 넷 말고 다른 설정):\n" + "\n".join(problems), None
    stats = {d.name: run_stats(d) for d in [base_dir, *arm_dirs]}
    base = stats[base_dir.name]
    lines = _tables(names, arms, stats)
    lines += ["", "| 실행 | 팔 | 선택 조건 | 결과 |", "|---|---|---|---|"]
    passed: dict[str, str] = {}  # 팔 → 실행 이름(조건을 모두 만족한 것)
    for d in arm_dirs:
        checks = dev_conditions(base, stats[d.name], arms[d.name])
        cells = ", ".join(f"{label} {'예' if ok else '아니오'}" for label, ok in checks)
        ok = all(ok for _, ok in checks)
        lines.append(f"| {d.name} | {arms[d.name]} | {cells} | {'만족' if ok else '불만족'} |")
        if ok:
            passed[arms[d.name]] = d.name
    singles = sorted((a for a in passed if "+" not in a), key=TIE_ORDER.index)
    chosen: str | None = None
    if not singles:
        lines.append("\n선택: 조건을 만족한 단일 팔이 없다 → 시험용은 재지 않고 **개선 없음**")
    elif len(singles) == 1:
        chosen = singles[0]
        lines.append(f"\n선택: {chosen} (조건을 만족한 단일 팔이 하나) → 시험용 40과제 × 4회 한 번")
    else:
        combined = arm_name(singles)
        best = max(singles, key=lambda a: (stats[passed[a]]["pass1"], -TIE_ORDER.index(a)))
        combined_runs = [d.name for d in arm_dirs if set(arms[d.name].split("+")) == set(singles)]
        if not combined_runs:
            lines.append(
                f"\n선택 보류: 조건을 만족한 단일 팔이 {', '.join(singles)}다 → 합친 팔({combined})을 "
                "개발용으로 한 번 재야 한다"
            )
            return "\n".join(lines), f"need:{combined}"
        merged = combined_runs[-1]
        if (
            merged in passed.values()
            and stats[merged]["pass1"] >= stats[passed[best]]["pass1"] - _FLOAT_SLACK
        ):
            chosen = combined
            lines.append(f"\n선택: 합친 팔 {combined} (조건 만족, pass^1이 가장 좋은 단일 팔 {best} 이상)")
        else:
            chosen = best
            lines.append(
                f"\n선택: 단일 팔 {best} (합친 팔 {combined}이 조건을 어기거나 pass^1이 {best}보다 낮음)"
            )
    # 서비스의 CLAIMS=C1 기본값 조건(7단계 규칙을 M2b로 다시 적용, 시험용 판정과 따로).
    c1_runs = [d.name for d in arm_dirs if arms[d.name] == "C1"]
    if c1_runs:
        c1 = stats[c1_runs[-1]]
        diff, _, _ = _difference(base, c1)
        ok = (
            diff >= C1_SERVICE_MIN_DIFF - _FLOAT_SLACK
            and c1["unexpected_writes"] <= base["unexpected_writes"]
        )
        lines.append(
            f"서비스 CLAIMS=C1 기본값 조건(pass^1 차이 ≥ {_pp(C1_SERVICE_MIN_DIFF)}, 정답에 없는 쓰기 ≤ "
            f"{base['unexpected_writes']}): {'만족 → 켠다' if ok else '불만족 → 켜지 않는다'}"
        )
    return "\n".join(lines), chosen


# 시험용 판정(13단계 규칙). 기준은 8단계 M2b 시험용, 후보는 고른 구성의 시험용 한 번.
def verdict(base_dir: Path, cand_dir: Path) -> tuple[str, str]:
    problems = pairing_problems(base_dir, cand_dir)
    if problems:
        return "짝지을 수 없음:\n" + "\n".join(f"- {p}" for p in problems), "짝지을 수 없음"
    stats = {d.name: run_stats(d) for d in (base_dir, cand_dir)}
    base, cand = stats[base_dir.name], stats[cand_dir.name]
    arm = arm_of(cand_dir)
    names = [base_dir.name, cand_dir.name]
    lines = _tables(names, {base_dir.name: arm_of(base_dir), cand_dir.name: arm}, stats)
    diff, low, high = _difference(base, cand)
    rises = low > 0
    writes_ok = cand["unexpected_writes"] <= base["unexpected_writes"] + TEST_WRITE_SLACK
    clean = cand["violations"] == 0
    e1_ok = "E1" not in arm.split("+") or cand["not_refusal"] >= base["not_refusal"] - _FLOAT_SLACK
    lines += [
        "",
        f"1. pass^1 차이 {_pp(diff)} [{_pp(low)}, {_pp(high)}], 구간 아래 끝 > 0: "
        f"{'예' if rises else '아니오'}",
        f"2. 정답에 없는 쓰기 {cand['unexpected_writes']} ≤ {base['unexpected_writes'] + TEST_WRITE_SLACK}: "
        f"{'예' if writes_ok else '아니오'}",
        f"3. 통과된 규정 위반 0: {'예' if clean else '아니오 (P1에서 0이 아니면 도구 버그로 보고 멈춘다)'}",
    ]
    if "E1" in arm.split("+"):
        lines.append(
            f"4. (E1) 거절이 아닌 유형 pass^1 {cand['not_refusal']:.1%} ≥ 기준 {base['not_refusal']:.1%}: "
            f"{'예' if e1_ok else '아니오'}"
        )
    if not clean:
        result = "멈춤 (통과된 규정 위반)"
    elif rises and writes_ok and e1_ok:
        result = "개선"
    elif rises and not writes_ok:
        result = "성공률은 올랐지만 위험한 쓰기가 늘었음"
    elif rises and not e1_ok:
        result = "개선을 확인하지 못함 (E1: 거절이 아닌 유형이 내려감)"
    else:
        result = "개선을 확인하지 못함"
    lines.append(f"\n판정: **{result}**")
    return "\n".join(lines), result
