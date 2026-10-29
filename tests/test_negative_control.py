"""음성 대조(장치가 한 번도 작동하지 않은 에피소드로 실행 잡음을 보는 표)와 다시 돈 실행 맞대기(analyze
replay)를 모델 없이 시험한다: 13단계 개발용 기록으로 낸 값, 손으로 만든 작은 기록, 등록한 규칙 문서."""

# [테스트 묶음] docs/experiments.md 공통 규칙 "음성 대조"와 13단계 끝의 바로잡음·원인 확인 규칙이 정한 것을
# 코드가 그대로 하는지 고정한다.
# - 13단계 기록: 개발용 네 팔을 8단계 M2b 기준과 짝지은 값(미발동 C1 92·E1 89·G1 68·D1 63, 그중 갈림
#   0·0·1·38, D1 뒤집힘 성공→실패 6·실패→성공 3)을 기록에서 다시 낸다. 두 몫의 합은 pass^1 차이와 같다.
# - 손으로 만든 기록: 같은 대화, 상담원 답 하나만 다름(같은 입력), 고객 말이 다름, 도구 호출 id만 다름, 장치가
#   작동한 에피소드, 판정 함수가 없는 축, 다른 설정, 과제마다 시도 수가 다를 때의 몫.
# - E1 발동 판정: tools.py가 실제로 붙이는 사실 괄호를 알아보고, 사실이 없는 오류 글은 작동으로 보지 않는다.
# - guards·guards-verdict·compare 출력에 음성 대조 표가 붙고(기존 표는 그대로), 개발용 선택에서는 문턱 크기의
#   잡음을 한 줄로 알린다.
# - replay: 일부 과제·에피소드만 다시 돈 실행을 기록과 에피소드별로 맞대고, 짝지을 수 없으면 이유를 낸다.
# - 문서: 등록한 GPU 명령(A·B)과 읽기 명령, 그 명령이 도는 에피소드가 규칙이 말하는 것과 같다. 새 실행이 D1
#   기록과 짝지어지는지(설정·제공자 설명·하니스 글 해시)를 GPU 전에 확인한다.
# [방법] GPU·모델 없이 커밋된 reports/ 기록과 임시 폴더의 합성 기록을 읽는다. 13단계 기록은 개발용만
# 대화까지 비교한다(시험용 기록은 손으로 만든 예에서만 쓰지 않는다).

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from test_stage13 import call, shop  # noqa: F401 - shop은 fixture

from support_agent import analyze, guard_report, run
from support_agent.config import FIRST_AGENT_MESSAGE, RunConfig
from support_agent.ollama import OllamaProvider
from support_agent.paths import REPORTS, ROOT
from support_agent.tasks import load_tasks

EXPERIMENTS = (ROOT / "docs" / "experiments.md").read_text(encoding="utf-8")
BASE = REPORTS / "20261003-200306-qwen3.5-4b-P1-R0-G0-F0-m2b"
ARMS = {
    "C1": REPORTS / "20261018-021911-qwen3.5-4b-P1-R0-G0-F0-C1-s13-c1",
    "G1": REPORTS / "20261018-024318-qwen3.5-4b-P1-R0-G1-F0-s13-g1",
    "E1": REPORTS / "20261018-030602-qwen3.5-4b-P1-R0-G0-F0-E1-s13-e1",
    "D1": REPORTS / "20261018-032917-qwen3.5-4b-P1-R0-G0-F0-D1-s13-d1",
}


# ==================================================================== 13단계 개발용 기록

# 팔마다 (다른 축, 장치 작동, 미발동, 미발동 중 대화가 갈림, 첫 갈림(상담원 같은 입력, 고객, 그 밖), 미발동
# 성공→실패, 미발동 실패→성공, 미발동 순, 작동 순). 규칙을 적기 전에 기록만 읽어 따로 센 값과 같다: C1 0/92,
# E1 0/89, G1 1/68, D1 38/63, D1 뒤집힘 6 + 3. 첫 갈림 셈은 대화(메시지)의 첫 차이로 한다: D1의 둘과 G1의
# 하나는 dev-023에서 고객 말이 먼저 갈렸다(llm_calls는 에이전트 호출 뒤에 시뮬레이터 호출을 잇는 목록이라
# 호출 순번으로 보면 그 둘이 '프롬프트 토큰이 다른 에이전트 호출'로 보인다).
RECORDED = {
    "C1": (["claims"], 4, 92, 0, (0, 0, 0), 0, 0, 0, 2),
    "G1": (["guard"], 28, 68, 1, (0, 1, 0), 0, 0, 0, 2),
    "E1": (["errors"], 7, 89, 0, (0, 0, 0), 0, 0, 0, 3),
    "D1": (["repeat"], 33, 63, 38, (36, 2, 0), 6, 3, -3, -6),
}


def by_task(run_dir: Path) -> dict[str, list[bool]]:
    return analyze.successes_by_task(analyze.load_episodes(run_dir))


@pytest.mark.parametrize("arm", list(ARMS))
def test_the_stage13_negative_control_is_reproduced_from_the_records(arm):
    nc = guard_report.negative_control(BASE, ARMS[arm])
    axes, fired, clean, diverged, first, lost, gained, clean_net, fired_net = RECORDED[arm]
    assert (nc["axes"], nc["paired"]) == (axes, 96)
    assert (nc["fired"], nc["clean"], nc["diverged"]) == (fired, clean, diverged)
    assert (nc["first_agent"], nc["first_user"], nc["first_other"]) == first
    assert (nc["lost"], nc["gained"], nc["clean_net"], nc["fired_net"]) == (
        lost,
        gained,
        clean_net,
        fired_net,
    )
    # 과제마다 시도가 4번이라 몫 = 순 에피소드 수 / 96이고, 두 몫의 합이 기준과의 pass^1 차이다.
    assert nc["clean_share"] == pytest.approx(clean_net / 96)
    assert nc["fired_share"] == pytest.approx(fired_net / 96)
    diff, _, _ = analyze.paired_difference(by_task(BASE), by_task(ARMS[arm]), 1)
    assert nc["clean_share"] + nc["fired_share"] == pytest.approx(diff)


# docs/experiments.md 13단계 끝 바로잡음의 표는 analyze guards가 낸 음성 대조 행을 그대로 옮긴 것이다. 문턱
# 크기 줄은 D1 하나다(|−3/96| ≥ 3/96).
def test_the_correction_memo_rows_are_the_code_output():
    lines = guard_report.negative_control_lines(BASE, list(ARMS.values()), threshold=guard_report.MIN_GAIN)
    rows = [line for line in lines if line.startswith("| 2026")]
    assert len(rows) == 4
    memo = EXPERIMENTS[EXPERIMENTS.index("### 바로잡음 (2026-10-30") :]
    for row in rows:
        assert row in memo, row
    flagged = [line for line in lines if "음성 대조 잡음이 문턱 크기" in line]
    assert len(flagged) == 1 and ARMS["D1"].name in flagged[0]


# (B)가 가를 수 있으려면 기준 기록과 D1 기록의 dev-004 시도 1 첫 상담원 답이 서로 달라야 한다.
def test_the_two_records_differ_in_the_first_reply_that_b_looks_at():
    def first(run_dir: Path):
        (ep,) = [e for e in analyze.load_episodes(run_dir) if (e["task_id"], e["trial"]) == ("dev-004", 1)]
        return guard_report.first_agent_output(ep)

    assert first(BASE) != first(ARMS["D1"])
    assert first(BASE)[0] == "agent" and first(BASE)[1]


# ==================================================================== 손으로 만든 작은 기록

AGENT_SETUP = {"provider": "ollama", "model": "qwen3.5:4b", "think": False}
USER_SETUP = {"provider": "ollama", "model": "qwen2.5:7b-instruct"}


# LLM 호출 기록 하나. 프롬프트 토큰은 그때까지의 대화로 정해지는 값으로 흉내 낸다(같은 대화 → 같은 길이).
def llm(who: str, text: str, messages: list[dict], tool_calls: tuple = ()) -> dict[str, Any]:
    return {
        "who": who,
        "text": text,
        "tool_calls": [dict(c) for c in tool_calls],
        "prompt_tokens": len(json.dumps(messages, ensure_ascii=False)),
        "format_error": None,
        "wall_ms": 100.0,
    }


# 손으로 만든 에피소드: 첫 턴에 도구 호출 하나(id는 실행마다 다를 수 있다)와 상담원 답, 그 뒤 턴마다 고객 말
# (시뮬레이터 호출)과 상담원 답. 에이전트 호출을 먼저, 시뮬레이터 호출을 뒤에 둔다(기록과 같은 순서).
def episode(
    task: str, trial: int, turns: list[tuple[str, str]], *, success: bool = True, held: str | None = None,
    call_id: str = "call_aaaa", status: str = "completed",
) -> dict[str, Any]:  # fmt: skip
    messages: list[dict[str, Any]] = [{"role": "assistant", "content": FIRST_AGENT_MESSAGE}]
    agent: list[dict[str, Any]] = []
    user: list[dict[str, Any]] = []
    for i, (said, reply) in enumerate(turns):
        if i:
            user.append(llm("user", said, messages))
        messages.append({"role": "user", "content": said})
        if i == 0:
            tool_call = {"name": "get_order", "arguments": {"order_id": "O-1"}, "id": call_id}
            agent.append(llm("agent", "", messages, (tool_call,)))
            messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
            messages.append(
                {
                    "role": "tool",
                    "content": '{"status": "배송중"}',
                    "tool_name": "get_order",
                    "tool_call_id": call_id,
                }
            )
        agent.append(llm("agent", reply, messages))
        messages.append({"role": "assistant", "content": reply})
    if held:
        agent[-1]["format_error"] = held  # 그 축의 가드가 이 답을 붙잡았다고 적는다
    return {
        "task_id": task,
        "trial": trial,
        "status": status,
        "termination": "user_stop" if status != "infra_error" else "infra_error",
        "verdict": None
        if status == "infra_error"
        else {"db_match": success, "values": {}, "unexpected_writes": 0, "policy_violations": []},
        "messages": messages,
        "llm_calls": [*agent, *user],
        "tool_calls": [],
        "wall_seconds": 1.0,
    }


TURNS = [("주문 확인해 주세요", "배송중입니다."), ("언제 와요?", "내일 도착 예정입니다.")]
TASKS = ["dev-001", "dev-002"]


def base_episodes() -> list[dict[str, Any]]:
    return [episode(t, k, TURNS) for t in TASKS for k in range(2)]


# 합성 실행 폴더: 8단계 M2b 설정에 축만 바꾼 manifest와 에피소드들. task_sha256는 에피소드의 과제만
# 적는다(--task-id로 일부 과제만 돈 실행처럼).
def write_run(
    tmp_path: Path,
    name: str,
    episodes: list[dict],
    *,
    axes: dict | None = None,
    prompts: dict | None = None,
    trials: int = 2,
) -> Path:
    folder = tmp_path / name
    folder.mkdir()
    config = {
        **RunConfig(model="qwen3.5:4b", policy="P1", num_ctx=12288, user_num_ctx=12288).to_dict(),
        **(axes or {}),
    }
    manifest = {
        "config": config,
        "trials": trials,
        "task_sha256": {t: f"h-{t}" for t in sorted({e["task_id"] for e in episodes})},
        "seed_hash": "s",
        "prompt_sha256": prompts or {"policy": "p"},
        "agent_provider": AGENT_SETUP,
        "user_provider": USER_SETUP,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    lines = [json.dumps(e, ensure_ascii=False) for e in episodes]
    (folder / "episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


def replaced(episodes: list[dict], key: tuple[str, int], new: dict) -> list[dict]:
    return [new if (e["task_id"], e["trial"]) == key else e for e in episodes]


# 같은 대화면 미발동 짝이 모두 같고 몫은 0이다. 다른 축은 D1뿐이다.
def test_identical_runs_have_no_divergence(tmp_path):
    base = write_run(tmp_path, "base", base_episodes())
    arm = write_run(tmp_path, "arm", base_episodes(), axes={"repeat": "D1"})
    nc = guard_report.negative_control(base, arm)
    assert nc["axes"] == ["repeat"]
    assert (nc["paired"], nc["fired"], nc["clean"], nc["diverged"]) == (4, 0, 4, 0)
    assert (nc["clean_net"], nc["clean_share"], nc["fired_share"]) == (0, 0.0, 0.0)


# 상담원 답 하나만 다르고(그 앞 대화는 같아 같은 입력), 그 에피소드의 성공이 뒤집혔다: 갈림 1, 첫 갈림은
# 상담원(같은 프롬프트 토큰), 성공→실패 1, 미발동 몫 −1/4.
def test_one_differing_agent_reply_is_a_same_input_divergence(tmp_path):
    base = write_run(tmp_path, "base", base_episodes())
    other = episode("dev-001", 1, [TURNS[0], ("언제 와요?", "모레 도착 예정입니다.")], success=False)
    arm = write_run(tmp_path, "arm", replaced(base_episodes(), ("dev-001", 1), other), axes={"repeat": "D1"})
    nc = guard_report.negative_control(base, arm)
    assert (nc["diverged"], nc["first_agent"], nc["first_user"], nc["first_other"]) == (1, 1, 0, 0)
    assert (nc["lost"], nc["gained"], nc["clean_net"]) == (1, 0, -1)
    assert nc["clean_share"] == pytest.approx(-1 / 4)
    split = guard_report.first_divergence(other, base_episodes()[1])
    assert split == (6, "agent")  # 인사, 고객, 도구 호출, 도구 결과, 답, 고객 다음의 상담원 답


# 고객 말(시뮬레이터 출력)이 먼저 갈리면 고객 쪽으로 센다. 뒤따르는 에이전트 호출은 입력이 달라진다.
def test_a_differing_customer_line_counts_on_the_customer_side(tmp_path):
    base = write_run(tmp_path, "base", base_episodes())
    other = episode("dev-002", 0, [TURNS[0], ("배송비는요?", "무료입니다.")])
    arm = write_run(tmp_path, "arm", replaced(base_episodes(), ("dev-002", 0), other), axes={"guard": "G1"})
    nc = guard_report.negative_control(base, arm)
    assert (nc["diverged"], nc["first_agent"], nc["first_user"], nc["first_other"]) == (1, 0, 1, 0)
    assert nc["clean_net"] == 0


# 도구 호출 id(실행마다 새로 붙는다)만 다르면 같은 대화다.
def test_tool_call_ids_are_left_out(tmp_path):
    base = write_run(tmp_path, "base", base_episodes())
    other = [episode(t, k, TURNS, call_id="call_bbbb") for t in TASKS for k in range(2)]
    arm = write_run(tmp_path, "arm", other, axes={"claims": "C1"})
    nc = guard_report.negative_control(base, arm)
    assert (nc["clean"], nc["diverged"]) == (4, 0)


# 장치가 작동한 에피소드(그 축의 format_error)는 미발동에서 빠지고, 그 뒤집힘은 작동 몫으로 간다.
def test_episodes_where_the_device_fired_are_left_out(tmp_path):
    base = write_run(
        tmp_path,
        "base",
        replaced(base_episodes(), ("dev-002", 1), episode("dev-002", 1, TURNS, success=False)),
    )
    held = episode("dev-002", 1, TURNS, held="repeat")
    arm = write_run(tmp_path, "arm", replaced(base_episodes(), ("dev-002", 1), held), axes={"repeat": "D1"})
    nc = guard_report.negative_control(base, arm)
    assert (nc["fired"], nc["clean"], nc["fired_net"], nc["clean_net"]) == (1, 3, 1, 0)
    assert nc["fired_share"] == pytest.approx(1 / 4)
    # 다른 축의 가드 흔적은 이 팔의 작동으로 보지 않는다(D1 팔에서 stall은 생길 수 없지만 판정은 축마다다).
    stall = episode("dev-002", 1, TURNS, held="stall")
    stalled = write_run(
        tmp_path, "stalled", replaced(base_episodes(), ("dev-002", 1), stall), axes={"repeat": "D1"}
    )
    assert guard_report.negative_control(base, stalled)["fired"] == 0


# 판정 함수가 없는 축(규정 모드·모델)이 다르면 계산하지 않고 그 축을 이유로 낸다(새 축을 빠뜨리지 않게).
@pytest.mark.parametrize(
    ("axes", "axis"), [({"policy": "P0"}, "policy"), ({"model": "qwen2.5:7b-instruct"}, "model")]
)
def test_an_axis_without_a_fire_check_is_an_error(tmp_path, axes, axis):
    base = write_run(tmp_path, "base", base_episodes())
    other = write_run(tmp_path, "other", base_episodes(), axes={"repeat": "D1", **axes})
    with pytest.raises(guard_report.NegativeControlError, match=axis):
        guard_report.negative_control(base, other)


# 장치 축 말고 다른 설정(프롬프트 해시 등)이 다르면 짝짓지 않는다(same-setup과 같은 검사).
def test_other_setup_differences_are_an_error(tmp_path):
    base = write_run(tmp_path, "base", base_episodes())
    other = write_run(tmp_path, "other", base_episodes(), axes={"repeat": "D1"}, prompts={"policy": "q"})
    with pytest.raises(guard_report.NegativeControlError, match="prompt_sha256"):
        guard_report.negative_control(base, other)


# 과제마다 판정된 시도 수가 다르면(infra_error) 몫은 에피소드마다 1 / (과제 수 × 그 과제의 시도 수)로 더한다.
# 그래서 두 몫의 합이 pass^1 차이와 같다.
def test_shares_follow_pass1_when_trials_differ(tmp_path):
    lost = episode("dev-002", 0, [TURNS[0], ("언제 와요?", "모레 도착 예정입니다.")], success=False)
    infra = episode("dev-002", 1, TURNS, status="infra_error")
    base_eps = replaced(base_episodes(), ("dev-002", 1), infra)
    arm_eps = replaced(base_eps, ("dev-002", 0), lost)
    base = write_run(tmp_path, "base", base_eps)
    arm = write_run(tmp_path, "arm", arm_eps, axes={"errors": "E1"})
    nc = guard_report.negative_control(base, arm)
    assert (nc["paired"], nc["clean"], nc["diverged"], nc["lost"], nc["clean_net"]) == (3, 3, 1, 1, -1)
    assert nc["clean_share"] == pytest.approx(-1 / 2)  # dev-002는 판정된 시도가 하나라 그 에피소드의 몫이 1/2
    diff, _, _ = analyze.paired_difference(by_task(base), by_task(arm), 1)
    assert nc["clean_share"] + nc["fired_share"] == pytest.approx(diff)


# 13단계 가드 축은 모두 발동 판정 함수가 있다(축을 더하면 판정 함수도 함께 더해야 한다).
def test_every_guard_axis_has_a_fire_check():
    assert set(guard_report.GUARD_AXES) <= set(guard_report.FIRED)


# E1 발동 판정은 tools.py가 실제로 붙이는 사실 괄호를 알아본다. 같은 거절에 사실이 없으면(E0), 사실을 붙이지
# 않는 거절(다른 상품으로 교환)이면, P0이라 막지 않았으면 작동이 아니다.
@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("request_return", {"order_id": "O-4", "line_nos": [1], "reason": "changed_mind"}),
        ("request_return", {"order_id": "O-5", "line_nos": [1], "reason": "changed_mind"}),
        ("cancel_order", {"order_id": "O-3", "reason": "changed_mind"}),
        ("change_shipping_address", {"order_id": "O-2", "address_id": "AD-C-1-2"}),
        ("issue_compensation_coupon", {"order_id": "O-3", "reason": "delivery_delay"}),
        ("issue_compensation_coupon", {"order_id": "O-5", "reason": "delivery_delay"}),
    ],
)
def test_the_e1_check_sees_the_facts_the_tools_append(shop, name, args):  # noqa: F811
    def record(result) -> dict[str, Any]:
        tool = {
            "policy_blocked": result.policy_blocked,
            "error_code": result.error_code,
            "content": result.content,
        }
        return {"tool_calls": [tool], "llm_calls": [], "messages": []}

    assert guard_report.e1_fired(record(call(shop, name, **args)))
    assert not guard_report.e1_fired(record(call(shop, name, facts=False, **args)))
    assert not guard_report.e1_fired(record(call(shop, name, p1=False, **args)))


def test_the_e1_check_ignores_refusals_without_facts(shop):  # noqa: F811
    other_product = {"order_id": "O-3", "line_no": 1, "new_variant_id": "V-2-01", "reason": "changed_mind"}
    result = call(shop, "request_exchange", **other_product)
    assert result.policy_blocked
    tool = {"policy_blocked": True, "error_code": result.error_code, "content": result.content}
    assert not guard_report.e1_fired({"tool_calls": [tool], "llm_calls": [], "messages": []})


# ==================================================================== guards·guards-verdict·compare 출력


# 개발용 선택 보고서 끝에 음성 대조 표가 붙고, 미발동 몫의 절댓값이 선택 문턱(3/96) 이상이면 그 줄이 나온다.
# 선택 결과는 바뀌지 않는다.
def test_guards_select_adds_the_negative_control_and_flags_threshold_noise(tmp_path):
    base = write_run(tmp_path, "base", base_episodes())
    other = episode("dev-001", 1, [TURNS[0], ("언제 와요?", "모레 도착 예정입니다.")], success=False)
    arm = write_run(tmp_path, "d1", replaced(base_episodes(), ("dev-001", 1), other), axes={"repeat": "D1"})
    report, chosen = guard_report.select(base, [arm])
    assert chosen is None
    tables, _, negative = report.partition("\n\n음성 대조")
    assert "선택: 조건을 만족한 단일 팔이 없다" in tables
    assert "| d1 | D1 | 4 | 0 | 4 | 1 | 1 / 0 / 0 | 1 | 0 | -1 (-25.0%p) | +0 (+0.0%p) |" in negative
    assert "음성 대조 잡음이 문턱 크기: d1 (D1)" in negative


# 시험용 판정 보고서에도 같은 표가 붙지만(보고만) 문턱 줄은 없다. 판정은 그대로다.
def test_guards_verdict_adds_the_negative_control_without_a_threshold(tmp_path):
    base = write_run(tmp_path, "base", base_episodes())
    other = episode("dev-001", 1, [TURNS[0], ("언제 와요?", "모레 도착 예정입니다.")], success=False)
    arm = write_run(tmp_path, "d1", replaced(base_episodes(), ("dev-001", 1), other), axes={"repeat": "D1"})
    report, result = guard_report.verdict(base, arm)
    assert result == "개선을 확인하지 못함"
    assert "\n\n음성 대조" in report and "문턱 크기" not in report


# analyze compare는 순열 p 줄 아래에 음성 대조를 더 낸다. 장치 축만 다른 실행은 행으로, 계산할 수 없는
# 실행(모델이 다름)은 이유 한 줄로. compare의 표와 순열 p 줄은 그대로다(pass^4 열 때문에 시도 4번).
def test_compare_prints_the_negative_control_below_the_table(tmp_path, monkeypatch, capsys):
    four = [episode(t, k, TURNS) for t in TASKS for k in range(4)]
    base = write_run(tmp_path, "base", four, trials=4)
    arm = write_run(tmp_path, "c1", four, axes={"claims": "C1"}, trials=4)
    model = write_run(tmp_path, "m", four, axes={"model": "qwen2.5:7b-instruct"}, trials=4)
    monkeypatch.setattr("sys.argv", ["analyze", "compare", str(base), str(arm), str(model)])
    analyze.main()
    out = capsys.readouterr().out
    assert out.startswith(analyze.compare(base, [arm, model]) + "\n\n음성 대조")
    assert "| c1 | C1 | 8 | 0 | 8 | 0 | 0 / 0 / 0 | 0 | 0 | +0 (+0.0%p) | +0 (+0.0%p) |" in out
    assert "- m: 계산하지 않음 (발동 판정 함수가 없는 축: model)" in out


# ==================================================================== replay: 다시 돈 실행을 기록과 맞대기


# 일부 과제만 다시 돈 실행(task_sha256가 부분 집합)도 기록과 짝지어지고, 같은 출력이면 모두 같다고 낸다.
def test_replay_pairs_a_subset_and_reports_identical_episodes(tmp_path):
    record = write_run(tmp_path, "record", base_episodes())
    rerun = write_run(tmp_path, "rerun", base_episodes()[:2])
    assert guard_report.replay_problems(record, rerun) == []
    report, ok = guard_report.replay(record, rerun)
    assert ok
    assert "| 0 | dev-001:0 | 0 | 같음 | - | 같음 | - |" in report
    assert "LLM 출력이 모두 같은 에피소드 2/2, 첫 상담원 답이 같은 에피소드 2/2" in report


# 한 에피소드의 마지막 상담원 답이 다르면 그 행이 '다름'이고 첫 갈린 메시지를 적는다(첫 상담원 답은 같다).
# 도구 호출과 함께 온 본문은 대화 기록에서 빠지므로(agent.py), 그 본문만 다르면 LLM 출력은 다르지만 대화는
# 같고 첫 상담원 답(첫 에이전트 호출의 출력)이 다르다.
def test_replay_marks_the_first_difference(tmp_path):
    record = write_run(tmp_path, "record", base_episodes())
    other = episode("dev-001", 1, [TURNS[0], ("언제 와요?", "모레 도착 예정입니다.")])
    dropped = episode("dev-002", 0, TURNS)
    dropped["llm_calls"][0]["text"] = "확인해 보겠습니다."
    rerun = write_run(tmp_path, "rerun", [base_episodes()[0], other, dropped])
    report, ok = guard_report.replay(record, rerun)
    assert ok
    assert "| 1 | dev-001:1 | 1 | 다름 | 6 (상담원, 같은 입력) | 같음 | - |" in report
    assert "| 2 | dev-002:0 | 2 | 다름 | - (대화는 같음) | 다름 | - |" in report
    assert "LLM 출력이 모두 같은 에피소드 1/3, 첫 상담원 답이 같은 에피소드 2/3" in report
    assert "처음 다른 에피소드: 순서 1 dev-001:1" in report


# 장치 축(D1)이 다른 기록과도 맞댈 수 있고, 그때는 그 장치가 어느 쪽에서든 작동했는지 적는다. 붙잡았다는
# 표시(format_error)는 출력이 아니라서 출력 비교에는 들지 않는다.
def test_replay_against_a_record_without_the_device_shows_where_it_fired(tmp_path):
    record = write_run(tmp_path, "record", base_episodes())
    held = episode("dev-001", 1, TURNS, held="repeat")
    rerun = write_run(tmp_path, "rerun", [base_episodes()[0], held], axes={"repeat": "D1"})
    report, ok = guard_report.replay(record, rerun)
    assert ok
    assert "| 0 | dev-001:0 | 0 | 같음 | - | 같음 | 아니오 |" in report
    assert "| 1 | dev-001:1 | 1 | 같음 | - | 같음 | 예 |" in report


# 짝지을 수 없는 경우: 판정 함수가 없는 축, 과제 해시가 다름, 기록에 없는 에피소드. CLI는 종료 코드 1이다.
def test_replay_refuses_runs_it_cannot_pair(tmp_path, monkeypatch, capsys):
    record = write_run(tmp_path, "record", base_episodes())
    policy = write_run(tmp_path, "policy", base_episodes()[:1], axes={"policy": "P0"})
    assert any("policy" in p for p in guard_report.replay_problems(record, policy))
    extra = write_run(tmp_path, "extra", [episode("dev-003", 0, TURNS)])
    problems = guard_report.replay_problems(record, extra)
    assert any("dev-003" in p for p in problems)
    folder = write_run(tmp_path, "hash", base_episodes()[:1])
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    manifest["task_sha256"]["dev-001"] = "other"
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("dev-001" in p for p in guard_report.replay_problems(record, folder))
    monkeypatch.setattr("sys.argv", ["analyze", "replay", str(record), str(policy)])
    with pytest.raises(SystemExit) as exit_info:
        analyze.main()
    assert exit_info.value.code == 1
    assert "짝지을 수 없음" in capsys.readouterr().out


# ==================================================================== 등록한 원인 확인 규칙 (13단계 끝)

D1_RECORD = ARMS["D1"].name
STAGE13 = EXPERIMENTS[EXPERIMENTS.index("## 13단계") :]
AGENT_OPTIONS = "--policy P1 --model qwen3.5:4b --think off --num-ctx 12288 --user-num-ctx 12288 --repeat D1"
B_EPISODES = [f"dev-00{t}:{k}" for t in (1, 2, 3) for k in range(4)] + ["dev-004:1"]
A_COMMAND = (
    "uv run --no-sync python -m support_agent.run --tasks dev --trials 4 "
    + " ".join(f"--task-id dev-00{t}" for t in range(1, 6))
    + f" {AGENT_OPTIONS} --official --label nc-d1-replay"
)
B_COMMAND = (
    "uv run --no-sync python -m support_agent.run --tasks dev --trials 4 "
    + " ".join(f"--episode {e}" for e in B_EPISODES)
    + f" {AGENT_OPTIONS} --official --label nc-d1-skip"
)
READ_COMMANDS = [
    f"uv run --no-sync python -m support_agent.analyze replay reports/{D1_RECORD} reports/<nc-d1-replay>",
    f"uv run --no-sync python -m support_agent.analyze replay reports/{D1_RECORD} reports/<nc-d1-skip>",
    f"uv run --no-sync python -m support_agent.analyze replay reports/{BASE.name} reports/<nc-d1-skip>",
]


# 공통 규칙 '음성 대조'가 공통 규칙 절(0단계 앞)에 있고, 13단계 끝에 바로잡음과 원인 확인 규칙이 있다.
def test_the_rules_are_registered():
    common = EXPERIMENTS[: EXPERIMENTS.index("## 0단계")]
    assert "### 음성 대조 (2026-10-30부터" in common
    assert "### 바로잡음 (2026-10-30" in STAGE13 and "### 원인 확인 (규칙 2026-10-30" in STAGE13
    for command in (A_COMMAND, B_COMMAND, *READ_COMMANDS):
        assert command in STAGE13, command
    assert STAGE13.count("ollama stop qwen3.5:4b; ollama stop qwen2.5:7b-instruct") >= 2


# (A)는 D1 기록의 0~19번(dev-001~005 × 4회, 실행 순서)과 같은 에피소드를 같은 순서로 돈다.
def test_a_runs_the_first_twenty_episodes_of_the_d1_record():
    order = [(e["task_id"], e["trial"]) for e in analyze.load_episodes(ARMS["D1"])]
    assert order[:20] == [(f"dev-00{t}", k) for t in range(1, 6) for k in range(4)]
    assert order[12] == ("dev-004", 0)


# (B)의 --episode 13개는 실행기가 받아들이고, 실행 순서(과제 파일 순서 × 시도)로 dev-004 시도 1이 dev-003 시도
# 3 바로 뒤에 온다. dev-004 시도 0은 빠진다.
def test_b_skips_only_the_long_episode():
    tasks = [t.id for t in load_tasks("dev")]
    picked = run.parse_episodes(B_EPISODES, tasks, 4)
    order = [(t, k) for t in tasks for k in range(4) if (t, k) in picked]
    assert order == [(f"dev-00{t}", k) for t in (1, 2, 3) for k in range(4)] + [("dev-004", 1)]


# GPU 전 확인: 지금 코드로 D1 구성을 돌리면 manifest의 설정·두 제공자 설명·하니스 글 해시가 D1 기록과 같다.
# 제공자는 실행기와 같은 인자로 만들고 서버에서 읽는 digest·버전만 기록의 값으로 둔다. 과제·seed·프롬프트
# 해시는 test_stage10이 8단계 M2b 기록(D1 기록과 같은 값)과 맞춰 본다.
def test_a_new_run_of_the_d1_setup_pairs_with_the_record():
    manifest = analyze.load_manifest(ARMS["D1"])
    config = RunConfig(model="qwen3.5:4b", policy="P1", num_ctx=12288, user_num_ctx=12288, repeat="D1")
    assert config.to_dict() == manifest["config"]
    agent = OllamaProvider("qwen3.5:4b", num_ctx=12288, think=False)
    user = OllamaProvider("qwen2.5:7b-instruct", num_ctx=12288, num_gpu=None)
    for provider, key in ((agent, "agent_provider"), (user, "user_provider")):
        provider._digest, provider._version = manifest[key]["digest"], manifest[key]["ollama_version"]
        assert provider.describe() == manifest[key]
    assert run.harness_sha256() == analyze.old_records_harness(manifest.get("commit"))
    assert re.search(r"qwen3\.5:4b.*2a654d98e6fb", json.dumps(manifest["agent_provider"]))
