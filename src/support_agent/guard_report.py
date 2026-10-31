"""13단계(4B 에이전트의 남은 실패를 겨냥한 가드)와 15단계(읽기 도구 출력 X1) 분석: 개발용 선택과 시험용
판정, 그리고 모든 팔 비교에 붙이는 음성 대조(장치가 한 번도 작동하지 않은 에피소드로 본 실행 잡음)와 다시 돈
실행 맞대기. 모델을 부르지 않는다.

사용법:
    uv run python -m support_agent.analyze guards reports/<M2b 개발용> reports/<팔 개발용> [...]
    uv run python -m support_agent.analyze guards-verdict reports/<M2b 시험용> reports/<고른 구성 시험용>
    uv run python -m support_agent.analyze replay reports/<기록> reports/<같은 설정으로 다시 돈 실행>
"""

# [파일 역할] 13단계 규칙(docs/experiments.md 13단계, 측정 전에 커밋)을 그대로 코드로 옮긴 분석. 팔(C1, G1,
# E1, D1, 합친 팔)의 개발용 기록을 8단계 qwen3.5:4b 개발용 기록(기준)과 과제 단위로 짝지어 선택 조건을
# 계산하고, 고른 구성의 시험용 기록으로 판정을 낸다. 보고만 하는 지표(유형별 pass^1, 전달된 거짓 완료, 같은 답
# 되풀이로 끝난 에피소드, 전달된 멈춤 답, 가드 발동 수, 시간)도 함께 낸다. 15단계(같은 문서 15단계)는 같은
# 선택·판정 규칙을 읽기 도구 출력 축 X1(get_order에 배송 시각) 한 팔에 쓰고, X1 팔이 있을 때만 읽기 도구 보고
# 표(get_order·track_shipment 호출, 배송 시각을 본 거절 에피소드, 프롬프트 토큰, get_order 호출 시간)를 더
# 낸다.
# [설계 이유] 문턱값과 판정표를 측정 전에 상수로 고정해, 결과를 본 뒤 기준을 바꿀 수 없게 한다(이 프로젝트의
# 측정 방식). 짝짓기 검사는 analyze.setup_differences를 그대로 쓰고 가드 축 넷과 읽기 축(reads)만 달라도 되게
# 한다.
# [주의] E1은 거절 과제의 판정 값(수령일·출고일 같은 날짜)과 같은 종류의 사실을 도구 오류 글에 붙이고, X1은
# 같은 종류의 사실을 get_order 결과에 붙인다. 그래서 거절 유형은 따로 보고하고, 거절이 아닌 세 유형의 pass^1이
# 기준보다 낮으면(한 에피소드라도) 그 팔을 고르지 않는다(FACT_ARMS). 시험용 기록으로는 판정만 하고 무엇도
# 고치지 않는다.
# [음성 대조] 기준과 장치 축(가드·도구 글)만 다른 팔에서, 다른 장치가 두 에피소드 어느 쪽에서도 작동하지 않은
# (과제, 시도) 짝은 장치가 대화를 바꿀 수 없었던 짝이다. 그 짝이 기준과 갈리면 실행 잡음이다. guards·
# guards-verdict 보고서 끝과 analyze compare의 순열 p 줄 아래에 이 표를 붙인다(docs/experiments.md 공통 규칙
# "음성 대조", 2026-10-30). 기존 표의 행·열은 그대로다. replay는 일부 과제·에피소드를 같은 설정으로 다시 돈
# 실행을 기록과 에피소드별로 맞대어, 갈림이 같은 순서에서 되풀이되는지 본다(13단계 끝 "원인 확인").
# [연결] analyze.py의 guards·guards-verdict·replay 명령과 compare 명령의 출력이 부른다. 이 모듈은 analyze.py의
# 함수를 가져다 쓰므로 analyze.py는 이 모듈을 함수 안에서 가져온다.

from __future__ import annotations

# json: 도구 호출 인자를 키 순서와 상관없이 비교하려고 정렬한 JSON 글로 바꾼다. re: E1 사실 괄호를 알아본다.
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from support_agent.agent import is_stall
from support_agent.analyze import (
    _other_script,
    delivered_claims,
    delivered_replies,
    is_success,
    load_episodes,
    load_manifest,
    load_task_types,
    paired_difference,
    pass_k,
    setup_differences,
    successes_by_task,
)
from support_agent.config import FIRST_AGENT_MESSAGE, RunConfig

# 13단계가 바꾸는 가드 축 넷과 15단계의 읽기 도구 출력 축(reads), 그리고 기본값. 이 축들 말고는 기준과 같아야
# 짝지을 수 있다. reads를 무시할 때 도구 목록 해시는 analyze.setup_differences가 두 실행의 X0 판으로 맞댄다.
GUARD_AXES = {"claims": "C0", "guard": "G0", "errors": "E0", "repeat": "D0", "reads": "X0"}
# 합친 팔에서 같은 성공률일 때 고르는 순서(작은 변경부터): 이미 있던 가드(C1, G1), 새 루프 가드(D1), 도구 글을
# 바꾸는 E1(판정 값과 같은 종류의 사실을 주므로 13단계에서 마지막), 그 뒤에 15단계의 X1(읽기 도구 결과와
# 설명을 바꾸고 E1처럼 판정 값과 같은 종류의 사실을 준다). 15단계는 X1 한 팔이라 순서가 쓰일 일은 없지만, 고른
# 단일 팔을 이 순서로 정렬하므로 모든 팔 이름이 들어 있어야 한다.
TIE_ORDER = ("C1", "G1", "D1", "E1", "X1")
# 판정 값(수령일·출고일 같은 날짜)과 같은 종류의 사실을 모델에 주는 팔. 이 팔이 들면 개발용 선택 조건 5와
# 시험용 판정 4(거절이 아닌 세 유형의 pass^1이 기준보다 낮지 않다)를 더한다(13단계 E1, 15단계 X1 규칙).
FACT_ARMS = ("E1", "X1")
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


# 실행의 팔 이름: 기본값이 아닌 가드·읽기 축 코드를 축 순서(C, G, E, D, X)로 이어 쓴 것(예: "C1", "C1+D1",
# "X1"). 모두 기본값이면 "기준".
def arm_of(run_dir: Path) -> str:
    config = {**RunConfig().to_dict(), **load_manifest(run_dir)["config"]}
    return arm_name([config[axis] for axis, default in GUARD_AXES.items() if config[axis] != default])


# 축 코드 목록을 축 순서(C, G, E, D, X)로 이어 쓴 팔 이름. 빈 목록이면 "기준".
def arm_name(codes: list[str]) -> str:
    order = "CGEDX"
    return "+".join(sorted(codes, key=lambda code: order.index(code[0]))) or "기준"


# 기준과 팔이 가드 축 넷과 읽기 축 말고 같은 설정인가(analyze same-setup과 같은 검사). 다르면 그 이유들.
def pairing_problems(base_dir: Path, cand_dir: Path) -> list[str]:
    return setup_differences(base_dir, cand_dir, ignore=tuple(GUARD_AXES))


# 같은 답 되풀이로 끝난 에피소드: 고객에게 전달된 상담원 답(고정 첫 인사 제외)의 마지막 셋이 공백만 빼고 같다.
# 감사에서 M2b 개발용 실패 40개 중 15개가 이 꼴이었다(13단계 "왜 하는가").
def ends_in_a_loop(episode: dict[str, Any]) -> bool:
    replies = [" ".join(r.split()) for r in delivered_replies(episode) if r != FIRST_AGENT_MESSAGE]
    return len(replies) >= 3 and len(set(replies[-3:])) == 1


# 에이전트가 배송 시각(출고·수령 시각, 도착 예정일)을 본 에피소드인가: 성공한 track_shipment가 있거나, 성공한
# get_order 결과에 비어 있지 않은 shipment가 있다(15단계 X1. X0의 get_order 결과에는 이 키가 없다). 15단계
# 보고 표의 '배송 시각을 본 거절 에피소드'가 쓴다. 기록의 도구 결과 글(JSON)만 읽고 대화 글은 보지 않는다.
def saw_shipment_times(episode: dict[str, Any]) -> bool:
    for tool in episode.get("tool_calls") or ():
        if not tool.get("ok"):
            continue
        if tool.get("name") == "track_shipment":
            return True
        if tool.get("name") == "get_order":
            try:
                result = json.loads(tool.get("content") or "")
            except ValueError:
                continue
            if isinstance(result, dict) and result.get("shipment"):
                return True
    return False


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# 기록 하나의 보고용 수치. 판정된 에피소드(infra 제외)만 센다.
def run_stats(run_dir: Path) -> dict[str, Any]:
    episodes = [e for e in load_episodes(run_dir) if e["status"] != "infra_error"]
    types = load_task_types()
    by_task = successes_by_task(episodes)
    agent_calls = [c for e in episodes for c in e["llm_calls"] if c["who"] == "agent"]
    errors_on = load_manifest(run_dir)["config"].get("errors") == "E1"
    endings = Counter(e["termination"] for e in episodes)
    found, replies = _other_script(episodes)
    tool_calls = [t for e in episodes for t in e["tool_calls"]]
    refusals = [e for e in episodes if types.get(e["task_id"]) == "refusal"]
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
        # 15단계 읽기 도구 보고(reads_table): 두 도구의 호출 수와 성공 수, 배송 시각을 본 거절 에피소드와 거절
        # 에피소드 수, 에이전트 호출당 프롬프트 토큰(서버가 알려 준 값이 있는 호출만), get_order 호출당
        # ms(도구 실행 시간, 기록의 ms).
        "get_order_calls": sum(t["name"] == "get_order" for t in tool_calls),
        "get_order_ok": sum(t["name"] == "get_order" and bool(t["ok"]) for t in tool_calls),
        "track_shipment_calls": sum(t["name"] == "track_shipment" for t in tool_calls),
        "track_shipment_ok": sum(t["name"] == "track_shipment" and bool(t["ok"]) for t in tool_calls),
        "refusal_saw_times": sum(saw_shipment_times(e) for e in refusals),
        "refusal_episodes": len(refusals),
        "prompt_tokens": _mean(
            [c["prompt_tokens"] for c in agent_calls if c.get("prompt_tokens") is not None]
        ),
        "get_order_ms": _mean([t["ms"] for t in tool_calls if t["name"] == "get_order" and "ms" in t]),
    }


def _pp(value: float) -> str:
    return f"{value:+.1%}p"


# 기준과 짝지은 pass^1 차이(점추정, 95% 구간 아래·위).
def _difference(base: dict[str, Any], cand: dict[str, Any]) -> tuple[float, float, float]:
    return paired_difference(base["by_task"], cand["by_task"], 1)


# 개발용 선택 조건 다섯(E1·X1이 든 팔만 다섯째). (조건 이름, 만족 여부) 목록.
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
    if any(code in FACT_ARMS for code in arm.split("+")):
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


# 15단계 읽기 도구 보고 표(보고만, 판정에 넣지 않음). X1 팔이 있는 보고서에만 붙인다(13단계 보고서는 그대로).
# 열: get_order·track_shipment 호출 수(괄호 안은 성공 수), 배송 시각을 본 거절 에피소드(saw_shipment_times /
# 판정된 거절 에피소드), 에이전트 호출당 프롬프트 토큰(X1은 get_order 결과가 길어져 는다), get_order 호출당
# ms.
def reads_table(names: list[str], arms: dict[str, str], stats: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "읽기 도구 (15단계, 보고만. 배송 시각을 본 = 성공한 track_shipment 또는 shipment가 있는 "
        "get_order 결과)",
        "",
        "| 실행 | 팔 | get_order 호출 (성공) | track_shipment 호출 (성공) | 배송 시각을 본 거절 에피소드 "
        "| 에이전트 호출당 프롬프트 토큰 | get_order 호출당 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in names:
        s = stats[name]
        lines.append(
            f"| {name} | {arms[name]} | {s['get_order_calls']} ({s['get_order_ok']}) "
            f"| {s['track_shipment_calls']} ({s['track_shipment_ok']}) "
            f"| {s['refusal_saw_times']}/{s['refusal_episodes']} | {s['prompt_tokens']:.0f} "
            f"| {s['get_order_ms']:.1f} |"
        )
    return lines


# 보고서에 15단계 읽기 도구 표를 붙이는가: 실행 가운데 X1 팔(이름에 X1)이 있을 때만.
def _has_reads_arm(arms: dict[str, str]) -> bool:
    return any("X1" in arm.split("+") for arm in arms.values())


# 개발용 선택(13단계 규칙, 15단계도 같은 규칙). 첫 실행이 기준(8단계 M2b 개발용), 나머지가 팔. 단일
# 팔(가드·읽기 축 하나)과 합친 팔(둘 이상)을 함께 받는다. 돌려주는 것: 보고서 글과 고른 구성(없으면 None, 합친
# 팔이 필요한데 없으면 "need:<팔>"). X1 팔이 있으면 두 보고 표 뒤에 15단계 읽기 도구 표를 붙인다.
def select(base_dir: Path, arm_dirs: list[Path]) -> tuple[str, str | None]:
    names = [base_dir.name, *(d.name for d in arm_dirs)]
    arms = {base_dir.name: arm_of(base_dir), **{d.name: arm_of(d) for d in arm_dirs}}
    problems = [f"- {d.name}: {p}" for d in arm_dirs for p in pairing_problems(base_dir, d)]
    if problems:
        return "짝지을 수 없음 (가드·읽기 축 말고 다른 설정):\n" + "\n".join(problems), None
    stats = {d.name: run_stats(d) for d in [base_dir, *arm_dirs]}
    base = stats[base_dir.name]
    # 음성 대조 표(보고만)는 보고서 맨 끝에 붙인다. 미발동 몫이 선택 문턱(MIN_GAIN) 크기면 그 줄도 낸다(선택
    # 조건은 그대로, docs/experiments.md 공통 규칙 "음성 대조").
    negative = ["", *negative_control_lines(base_dir, arm_dirs, threshold=MIN_GAIN)]
    lines = _tables(names, arms, stats)
    if _has_reads_arm(arms):
        lines += ["", *reads_table(names, arms, stats)]
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
            return "\n".join(lines + negative), f"need:{combined}"
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
    return "\n".join(lines + negative), chosen


# 시험용 판정(13단계 규칙, 15단계도 같은 규칙). 기준은 8단계 M2b 시험용, 후보는 고른 구성의 시험용 한 번. 판정
# 4(거절이 아닌 세 유형이 기준보다 낮지 않다)는 고른 구성에 E1·X1(FACT_ARMS)이 있을 때만 보고 그 팔 이름을
# 적는다.
def verdict(base_dir: Path, cand_dir: Path) -> tuple[str, str]:
    problems = pairing_problems(base_dir, cand_dir)
    if problems:
        return "짝지을 수 없음:\n" + "\n".join(f"- {p}" for p in problems), "짝지을 수 없음"
    stats = {d.name: run_stats(d) for d in (base_dir, cand_dir)}
    base, cand = stats[base_dir.name], stats[cand_dir.name]
    arm = arm_of(cand_dir)
    names = [base_dir.name, cand_dir.name]
    arms = {base_dir.name: arm_of(base_dir), cand_dir.name: arm}
    lines = _tables(names, arms, stats)
    if _has_reads_arm(arms):
        lines += ["", *reads_table(names, arms, stats)]
    diff, low, high = _difference(base, cand)
    rises = low > 0
    writes_ok = cand["unexpected_writes"] <= base["unexpected_writes"] + TEST_WRITE_SLACK
    clean = cand["violations"] == 0
    facts = [code for code in arm.split("+") if code in FACT_ARMS]  # 판정 4를 보는 팔 (E1, X1)
    named = "·".join(facts)
    facts_ok = not facts or cand["not_refusal"] >= base["not_refusal"] - _FLOAT_SLACK
    lines += [
        "",
        f"1. pass^1 차이 {_pp(diff)} [{_pp(low)}, {_pp(high)}], 구간 아래 끝 > 0: "
        f"{'예' if rises else '아니오'}",
        f"2. 정답에 없는 쓰기 {cand['unexpected_writes']} ≤ {base['unexpected_writes'] + TEST_WRITE_SLACK}: "
        f"{'예' if writes_ok else '아니오'}",
        f"3. 통과된 규정 위반 0: {'예' if clean else '아니오 (P1에서 0이 아니면 도구 버그로 보고 멈춘다)'}",
    ]
    if facts:
        lines.append(
            f"4. ({named}) 거절이 아닌 유형 pass^1 {cand['not_refusal']:.1%} "
            f"≥ 기준 {base['not_refusal']:.1%}: {'예' if facts_ok else '아니오'}"
        )
    if not clean:
        result = "멈춤 (통과된 규정 위반)"
    elif rises and writes_ok and facts_ok:
        result = "개선"
    elif rises and not writes_ok:
        result = "성공률은 올랐지만 위험한 쓰기가 늘었음"
    elif rises and not facts_ok:
        result = f"개선을 확인하지 못함 ({named}: 거절이 아닌 유형이 내려감)"
    else:
        result = "개선을 확인하지 못함"
    lines.append(f"\n판정: **{result}**")
    # 음성 대조(보고만): 시험용 판정에는 선택 문턱이 없으므로 문턱 줄을 내지 않는다. 시험용 기록은 대화가
    # 같은지 수만 센다(글을 보여 주지 않는다).
    lines += ["", *negative_control_lines(base_dir, [cand_dir])]
    return "\n".join(lines), result


# ==================================================================== 음성 대조 (장치 미발동 에피소드)
# [무엇] 기준과 장치 축만 다른 팔을 (과제, 시도)로 짝지을 때, 다른 장치가 두 에피소드 어느 쪽에서도 한 번도
# 작동하지 않은 짝(미발동)은 장치가 대화를 바꿀 수 없었던 짝이다. 그 짝의 대화가 기준과 갈리면 장치의 효과가
# 아니라 실행 잡음(같은 입력에 다른 출력 등)이다. 미발동 짝의 성공 차이(미발동 몫)와 장치가 작동한 짝의 성공
# 차이(작동 몫)를 나눠 내면, 두 몫의 합이 기준과의 pass^1 차이다.
# [왜] 13단계 개발용 D1 팔은 미발동 63개 가운데 38개가 기준과 갈려 순 −3 에피소드(−3.1%p)가 생겼다. 팔의
# 차이를 장치 몫과 잡음 몫으로 가르지 않으면 선택 문턱(+3.1%p) 크기의 잡음을 장치의 효과로 읽을 수 있다. 같은
# 설정을 한 번 더 재는 대조는 실행기 상태가 바뀌지 않으면 글자까지 같게 재현돼 잡음을 0으로 잘못 보일 수
# 있어, 이미 있는 기록 안의 미발동 짝을 쓴다(docs/experiments.md 공통 규칙 "음성 대조").
# [발동 판정] 축(설정 이름)마다 판정 함수를 FIRED에 등록한다. 판정 함수가 없는 축이 다르면
# NegativeControlError로 멈춘다: 새 축을 더하면서 판정을 빠뜨리면 모든 짝이 '미발동'으로 세여 장치의 효과가
# 잡음으로 보이기 때문이다(15단계 읽기 도구 출력 축 reads는 x1_fired). 장치 축 말고 다른 설정이 달라도 짝짓지
# 않는다(same-setup과 같은 검사).


class NegativeControlError(ValueError):
    """음성 대조를 낼 수 없다: 발동 판정 함수가 없는 축이 다르거나, 장치 축 말고 다른 설정이 다르다."""


# 가드가 붙잡은 답의 표시(에이전트 호출 기록의 format_error)로 그 가드가 작동했는지 보는 판정 함수를 만든다.
# C(완료 주장)는 unbacked_claim, G(멈춤)는 stall, D(되풀이)는 repeat다(agent.py). 붙잡은 답이 하나라도 있으면
# 작동이다. 붙잡지 않고 전달한 답은 가드가 대화를 바꾸지 않았으므로 작동이 아니다.
def _held(kind: str) -> Callable[[dict[str, Any]], bool]:
    def fired(episode: dict[str, Any]) -> bool:
        return any(call.get("format_error") == kind for call in episode.get("llm_calls") or ())

    return fired


# E1이 규정 거절 오류 글 끝에 붙이는 사실 괄호의 모양. tools.py의 _status_facts·_window_facts·_coupon_facts·
# _delay_facts가 만드는 글이고 머리말은 13단계 규칙의 고정 목록이다. 괄호 안에는 다른 괄호가 없다(시각·상태·
# 쿠폰 번호뿐). 쿠폰 한도의 기간(30일)은 rules.py 상수라 숫자로 받는다.
_E1_FACTS = re.compile(
    r" \((?:(?:수령|주문 상태|이 주문의 보상 쿠폰|최근 \d+일 보상 쿠폰|도착 예정일): [^()]*"
    r"|배송 정보 없음)\)$"
)


def e1_fired(episode: dict[str, Any]) -> bool:
    """E1이 작동한 에피소드: P1이 막은 도구 호출 가운데 E1 코드 목록의 오류 글 끝에 사실 괄호가 붙은 것이
    있다.

    run_stats의 e1_facts(막힌 E1 코드의 수)와 달리 실제로 붙은 괄호를 본다. 같은 코드라도 사실을 붙이지 않는
    거절(불량 사유의 coupon_not_eligible)이 있고, E0·P0의 기록에는 괄호가 없다.
    """
    return any(
        bool(tool.get("policy_blocked"))
        and tool.get("error_code") in E1_CODES
        and _E1_FACTS.search(tool.get("content") or "") is not None
        for tool in episode.get("tool_calls") or ()
    )


def x1_fired(episode: dict[str, Any]) -> bool:
    """X1(15단계)이 작동한 에피소드: 에이전트 호출이 한 번이라도 있다.

    X1은 get_order의 도구 설명을 바꾸고, 도구 목록은 모든 에이전트 호출의 입력에 들어간다(agent.py가
    호출마다 visible_tools를 보낸다). 그래서 설명을 본 모델의 출력은 첫 호출부터 달라질 수 있고, get_order를
    부르지 않은 에피소드도 장치가 대화를 바꿀 수 있었던 짝이다. 에이전트 호출 없이 끝난 에피소드(고객이 첫
    말에서 끝냄)만 미발동이다. 이 정의에서는 X1 팔의 음성 대조에 미발동 짝이 거의 없어 실행 잡음을 기록
    안에서 가를 수 없고, 그것을 15단계 규칙의 한계에 적었다. get_order 결과의 shipment를 실제로 본
    에피소드는 saw_shipment_times로 따로 센다.
    """
    return any(call.get("who") == "agent" for call in episode.get("llm_calls") or ())


# 축(manifest config의 이름) → 그 축의 장치가 에피소드에서 한 번이라도 작동했는가. 13단계 가드 축 넷과 15단계
# 읽기 축이다(GUARD_AXES와 같은 이름, 테스트가 빠짐없이 있는지 본다).
FIRED: dict[str, Callable[[dict[str, Any]], bool]] = {
    "claims": _held("unbacked_claim"),
    "guard": _held("stall"),
    "errors": e1_fired,
    "repeat": _held("repeat"),
    "reads": x1_fired,
}


def device_axes(base_dir: Path, other_dir: Path) -> list[str]:
    """두 실행에서 값이 다른 설정 축(이름순). 발동 판정 함수가 없는 축이 다르면 NegativeControlError.

    옛 manifest에 없는 설정은 기본값으로 본다(setup_differences와 같다). 빈 목록이면 같은 설정이고, 그때는
    모든 짝이 미발동이다(같은 설정을 다시 잰 실행의 비교).
    """
    defaults = RunConfig().to_dict()
    a, b = load_manifest(base_dir)["config"], load_manifest(other_dir)["config"]
    differing = sorted(
        key
        for key in set(a) | set(b) | set(defaults)
        if a.get(key, defaults.get(key)) != b.get(key, defaults.get(key))
    )
    unknown = [key for key in differing if key not in FIRED]
    if unknown:
        raise NegativeControlError(f"발동 판정 함수가 없는 축: {', '.join(unknown)}")
    return differing


# 도구 호출 목록을 비교용 값으로: (이름, 키를 정렬한 인자 JSON). 호출 id는 실행마다 새로 붙으므로 뺀다.
def _calls(calls: Any) -> tuple[tuple[str, str], ...]:
    return tuple(
        (call["name"], json.dumps(call.get("arguments"), ensure_ascii=False, sort_keys=True))
        for call in calls or ()
    )


# 대화 메시지 하나의 비교용 값: 역할, 글, 도구 이름, 전달 여부, 하니스 안내 여부, 도구 호출(id 제외). 도구
# 결과의 tool_call_id도 실행마다 다른 id라 뺀다.
def _message(message: dict[str, Any]) -> tuple[Any, ...]:
    return (
        message["role"],
        message.get("content") or "",
        message.get("tool_name"),
        message.get("delivered", True),
        message.get("harness", False),
        _calls(message.get("tool_calls")),
    )


# LLM 호출 하나의 출력: 누구의 호출, 본문, 도구 호출(id 제외). 붙잡은 표시(format_error)와 시간은 출력이
# 아니다.
def _output(call: dict[str, Any]) -> tuple[Any, ...]:
    return (call["who"], call.get("text") or "", _calls(call.get("tool_calls")))


def first_agent_output(episode: dict[str, Any]) -> tuple[Any, ...] | None:
    """첫 에이전트 호출의 출력(누구, 본문, 도구 호출). 에이전트 호출이 없으면 None."""
    return next((_output(call) for call in episode.get("llm_calls") or () if call["who"] == "agent"), None)


def first_divergence(a: dict[str, Any], b: dict[str, Any]) -> tuple[int, str] | None:
    """두 에피소드의 대화(메시지)가 처음 갈린 자리와 그쪽. 대화가 같으면 None.

    쪽은 셋이다. "agent": 처음 다른 메시지가 상담원 글이고, 순서대로 맞댄 에이전트 호출 가운데 출력이 처음
    다른 것의 프롬프트 토큰 수가 같다(같은 입력에 다른 출력). "user": 처음 다른 메시지가 고객
    말(시뮬레이터 출력)이다. 한쪽 대화가 먼저 끝났으면 긴 쪽의 다음 메시지로 본다(짧은 쪽 시뮬레이터가 그
    자리에서 대화를 끝냄). "other": 그 밖(도구 결과, 하니스 안내, 상담원 글인데 입력 길이가 다름 등).
    [주의] llm_calls는 에이전트 호출을 모두 적은 뒤 시뮬레이터 호출을 잇는 목록이라(episode.py) 호출 순번은
    시간 순서가 아니다. 그래서 갈린 자리는 시간 순서인 메시지로 정하고, 에이전트 호출은 그 역할 안에서만
    맞댄다.
    """
    left = [_message(m) for m in a["messages"]]
    right = [_message(m) for m in b["messages"]]
    if left == right:
        return None
    index = next(
        (i for i, (x, y) in enumerate(zip(left, right, strict=False)) if x != y), min(len(left), len(right))
    )
    kinds = {("harness" if m[4] else m[0]) for m in left[index : index + 1] + right[index : index + 1]}
    if kinds == {"user"}:
        return index, "user"
    if kinds == {"assistant"}:
        agent_a = [c for c in a["llm_calls"] if c["who"] == "agent"]
        agent_b = [c for c in b["llm_calls"] if c["who"] == "agent"]
        pair = next(
            ((x, y) for x, y in zip(agent_a, agent_b, strict=False) if _output(x) != _output(y)), None
        )
        if pair is not None and pair[0].get("prompt_tokens") is not None:
            if pair[0].get("prompt_tokens") == pair[1].get("prompt_tokens"):
                return index, "agent"
    return index, "other"


# 판정된(infra_error가 아닌) 에피소드를 (과제, 시도)로.
def _judged_by_key(run_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    return {(e["task_id"], e["trial"]): e for e in load_episodes(run_dir) if e["status"] != "infra_error"}


def negative_control(base_dir: Path, arm_dir: Path) -> dict[str, Any]:
    """기준과 팔의 음성 대조. 짝지을 수 없으면 NegativeControlError.

    돌려주는 것: axes(다른 장치 축), paired(짝지은 판정된 에피소드), fired(어느 쪽에서든 장치가 작동한 짝),
    clean(미발동 짝), diverged(미발동 중 대화가 갈린 짝), first_agent·first_user·first_other(갈린 짝의 첫 갈림
    쪽, first_divergence), lost·gained(미발동 짝의 성공→실패·실패→성공), clean_net·fired_net(팔 성공 수 − 기준
    성공 수), clean_share·fired_share(pass^1 몫: 에피소드마다 (팔 − 기준) / (과제 수 × 그 과제의 판정된 시도
    수)를 더한 값. 두 몫의 합이 pass^1 차이다. 과제마다 시도 수가 같으면 순 에피소드 수 / 짝지은 수).
    성공은 analyze.is_success(지금 규칙으로 다시 계산한 성공)다.
    """
    axes = device_axes(base_dir, arm_dir)
    problems = setup_differences(base_dir, arm_dir, ignore=tuple(axes))
    if problems:
        shown = "; ".join(problems[:3]) + (f" 외 {len(problems) - 3}개" if len(problems) > 3 else "")
        raise NegativeControlError(f"장치 축 말고도 설정이 다르다: {shown}")
    checks = [FIRED[axis] for axis in axes]
    # setup_differences가 과제별로 판정된 시도 번호가 같은지 봤으므로 두 실행의 키는 같다.
    base, arm = _judged_by_key(base_dir), _judged_by_key(arm_dir)
    trials = Counter(task for task, _ in base)
    counts: Counter[str] = Counter()
    shares = {"clean": 0.0, "fired": 0.0}
    for key in sorted(base):
        before, after = base[key], arm[key]
        delta = int(is_success(after)) - int(is_success(before))
        side = "fired" if any(check(e) for check in checks for e in (before, after)) else "clean"
        counts[side] += 1
        counts[f"{side}_net"] += delta
        shares[side] += delta / (len(trials) * trials[key[0]])
        if side == "fired":
            continue
        counts["lost"] += delta < 0
        counts["gained"] += delta > 0
        split = first_divergence(after, before)
        if split is not None:
            counts["diverged"] += 1
            counts[f"first_{split[1]}"] += 1
    keys = ("fired", "clean", "diverged", "first_agent", "first_user", "first_other", "lost", "gained")
    return {
        "axes": axes,
        "paired": len(base),
        **{key: counts[key] for key in keys},
        "clean_net": counts["clean_net"],
        "fired_net": counts["fired_net"],
        "clean_share": shares["clean"],
        "fired_share": shares["fired"],
    }


# 실행의 다른 장치 축 값을 팔 이름처럼(C, G, E, D 순, 예: "C1+D1"). 다른 축이 없으면 "없음".
def _axes_label(run_dir: Path, axes: list[str]) -> str:
    config = {**RunConfig().to_dict(), **load_manifest(run_dir)["config"]}
    return arm_name([str(config[axis]) for axis in axes]) if axes else "없음"


def negative_control_lines(
    base_dir: Path, run_dirs: list[Path], *, threshold: float | None = None
) -> list[str]:
    """음성 대조 표(보고만)의 줄들. 짝지을 수 없는 실행은 이유 한 줄로 낸다(계산하지 않음).

    threshold(개발용 선택 문턱, pass^1 몫)를 주면 미발동 몫의 절댓값이 그 이상인 실행마다 "음성 대조 잡음이
    문턱 크기" 줄을 더 낸다. 선택 조건은 바꾸지 않는다(docs/experiments.md 공통 규칙 "음성 대조").
    """
    lines = [
        "음성 대조 (다른 장치가 두 에피소드 어느 쪽에서도 한 번도 작동하지 않은 짝 = 미발동. 보고만, "
        'docs/experiments.md 공통 규칙 "음성 대조")'
    ]
    rows, flags, notes = [], [], []
    for run_dir in run_dirs:
        try:
            nc = negative_control(base_dir, run_dir)
        except NegativeControlError as error:
            notes.append(f"- {run_dir.name}: 계산하지 않음 ({error})")
            continue
        label = _axes_label(run_dir, nc["axes"])
        rows.append(
            f"| {run_dir.name} | {label} | {nc['paired']} | {nc['fired']} | {nc['clean']} "
            f"| {nc['diverged']} | {nc['first_agent']} / {nc['first_user']} / {nc['first_other']} "
            f"| {nc['lost']} | {nc['gained']} | {nc['clean_net']:+d} ({_pp(nc['clean_share'])}) "
            f"| {nc['fired_net']:+d} ({_pp(nc['fired_share'])}) |"
        )
        if threshold is not None and abs(nc["clean_share"]) >= threshold - _FLOAT_SLACK:
            flags.append(
                f"- 음성 대조 잡음이 문턱 크기: {run_dir.name} ({label}) 미발동 몫 {_pp(nc['clean_share'])}, "
                f"절댓값이 선택 문턱 {_pp(threshold)} 이상 → 선택 문장에 적는다 (선택 조건은 그대로)"
            )
    if rows:
        lines += [
            "",
            "| 실행 | 다른 축 | 짝지은 에피소드 | 장치 작동 | 미발동 | 미발동 중 대화가 갈림 "
            "| 첫 갈림: 상담원(같은 입력) / 고객 / 그 밖 | 미발동 성공→실패 | 미발동 실패→성공 | 미발동 몫 "
            "| 작동 몫 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
            *rows,
            "",
            "몫 = (팔 성공 − 기준 성공)을 에피소드마다 1 / (과제 수 × 그 과제의 판정된 시도 수)로 더한 "
            "pass^1 몫 (과제마다 시도 수가 같으면 순 에피소드 수 / 짝지은 에피소드 수). 미발동 몫 + 작동 몫 "
            "= 기준과의 pass^1 차이. 대화 = 메시지의 역할·글·도구 호출 이름과 인자(도구 호출 id 제외)",
        ]
    return lines + flags + notes


# ==================================================================== 다시 돈 실행을 기록과 맞대기
# [무엇] 같은 설정으로 일부 과제(--task-id)나 일부 에피소드(--episode)를 다시 돈 실행을, 기록의 같은 (과제,
# 시도)와 에피소드마다 맞댄다. 지표는 두 역할 모든 LLM 호출 출력(누구, 본문, 도구 호출 이름과 인자. id
# 제외)의 글자 일치와 첫 상담원 답(첫 에이전트 호출의 출력)의 일치다. 13단계 끝 "원인 확인"의 (A)·(B)를
# 읽는 데 쓴다.
# [짝짓기] 설정·프롬프트·하니스 글·제공자 설명·seed가 같아야 한다(same-setup과 같은 검사). 다른 점은 둘만
# 허용한다: (1) 다시 돈 실행이 기록의 일부 과제·에피소드만 돈 것(과제 해시는 돈 과제마다 기록과 같아야 함),
# (2) 발동 판정 함수가 있는 장치 축(기준 기록과 맞댈 때). (2)이면 그 장치가 어느 쪽에서든 작동했는지 함께
# 적는다.


def replay_problems(record_dir: Path, rerun_dir: Path) -> list[str]:
    """다시 돈 실행을 기록과 에피소드마다 맞댈 수 없는 이유들. 빈 목록이면 맞댈 수 있다."""
    try:
        axes = device_axes(record_dir, rerun_dir)
    except NegativeControlError as error:
        return [str(error)]
    # 일부만 다시 돌면 과제 해시 목록(--task-id)과 과제별 판정된 시도가 기록과 다르다. 그 둘은 아래에서
    # '기록의 부분 집합인가'로 따로 본다.
    problems = [
        problem
        for problem in setup_differences(record_dir, rerun_dir, ignore=tuple(axes))
        if not problem.startswith(("task_sha256 differs", "judged trials of "))
    ]
    recorded = load_manifest(record_dir)["task_sha256"]
    for task, digest in sorted(load_manifest(rerun_dir)["task_sha256"].items()):
        if task not in recorded:
            problems.append(f"과제 {task}: 기록에 없다")
        elif recorded[task] != digest:
            problems.append(f"과제 {task}: 과제 해시가 기록과 다르다")
    keys = {(e["task_id"], e["trial"]) for e in load_episodes(record_dir)}
    problems += [
        f"에피소드 {e['task_id']}:{e['trial']}: 기록에 없다"
        for e in load_episodes(rerun_dir)
        if (e["task_id"], e["trial"]) not in keys
    ]
    return problems


_SIDE = {"agent": "상담원, 같은 입력", "user": "고객", "other": "그 밖"}


def replay(record_dir: Path, rerun_dir: Path) -> tuple[str, bool]:
    """(보고서 글, 맞댈 수 있었는가). 다시 돈 실행의 순서대로 에피소드마다 한 행이다.

    열: 순서(다시 돈 실행 안), 에피소드, 기록의 순서, LLM 출력(같음·다름·infra), 첫 갈린 메시지(대화의 첫
    차이 자리와 쪽. 출력만 다르고 대화가 같으면 "- (대화는 같음)", 예: 도구 호출과 함께 온 본문은 대화
    기록에서 빠진다), 첫 상담원 답, 장치 작동(장치 축이 다를 때만 예·아니오, 같은 설정이면 "-"). 끝 줄에 같은
    수와 처음 다른 에피소드.
    """
    problems = replay_problems(record_dir, rerun_dir)
    if problems:
        return "짝지을 수 없음:\n" + "\n".join(f"- {problem}" for problem in problems), False
    axes = device_axes(record_dir, rerun_dir)
    checks = [FIRED[axis] for axis in axes]
    recorded = load_episodes(record_dir)
    position = {(e["task_id"], e["trial"]): i for i, e in enumerate(recorded)}
    by_key = {(e["task_id"], e["trial"]): e for e in recorded}
    reruns = load_episodes(rerun_dir)
    rows: list[str] = []
    same_outputs = same_first = 0
    first_different: str | None = None
    for i, again in enumerate(reruns):
        key = (again["task_id"], again["trial"])
        before = by_key[key]
        name = f"{key[0]}:{key[1]}"
        fired = (
            "-" if not checks else ("예" if any(c(e) for c in checks for e in (again, before)) else "아니오")
        )
        if "infra_error" in (again["status"], before["status"]):
            rows.append(f"| {i} | {name} | {position[key]} | infra | - | - | {fired} |")
            first_different = first_different or f"순서 {i} {name}"
            continue
        outputs = [_output(c) for c in again["llm_calls"]] == [_output(c) for c in before["llm_calls"]]
        first = first_agent_output(again)
        first_same = first is not None and first == first_agent_output(before)
        split = first_divergence(again, before)
        if split is not None:
            where = f"{split[0]} ({_SIDE[split[1]]})"
        else:
            where = "-" if outputs else "- (대화는 같음)"
        rows.append(
            f"| {i} | {name} | {position[key]} | {'같음' if outputs else '다름'} | {where} "
            f"| {'같음' if first_same else '다름'} | {fired} |"
        )
        same_outputs += outputs
        same_first += first_same
        if not outputs and first_different is None:
            first_different = f"순서 {i} {name}"
    device = f" 다른 장치 축: {', '.join(axes)} (장치 작동 = 그 장치가 어느 쪽에서든 작동함)." if axes else ""
    lines = [
        f"기록 {record_dir.name} ↔ 다시 돈 실행 {rerun_dir.name}: 같은 (과제, 시도)끼리, 다시 돈 실행의 "
        f"순서대로 맞댄다 (에피소드 {len(reruns)}개). LLM 출력 = 두 역할 모든 호출의 누구·본문·도구 "
        "호출(이름과 인자, id 제외). 첫 상담원 답 = 첫 에이전트 호출의 출력." + device,
        "",
        "| 순서 | 에피소드 | 기록의 순서 | LLM 출력 | 첫 갈린 메시지 | 첫 상담원 답 | 장치 작동 |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
        f"LLM 출력이 모두 같은 에피소드 {same_outputs}/{len(reruns)}, 첫 상담원 답이 같은 에피소드 "
        f"{same_first}/{len(reruns)}, 처음 다른 에피소드: {first_different or '없음'}",
    ]
    return "\n".join(lines), True
