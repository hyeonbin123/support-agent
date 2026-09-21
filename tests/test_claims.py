"""The claim guard (C1): a reply that says work is done which no tool result of the conversation shows."""

from __future__ import annotations

import json

import pytest
from conftest import NOW
from toy_tools import TOY_REGISTRY

from support_agent import analyze, db
from support_agent.agent import CLAIM_NOTICE, agent_turn, new_state
from support_agent.chat import Message, ScriptedProvider, ToolCall
from support_agent.claims import unbacked_claim
from support_agent.config import JUDGED, RunConfig
from support_agent.toolkit import ToolContext, execute


def shown(tool: str, **result) -> Message:
    return Message("tool", json.dumps(result, ensure_ascii=False), tool_name=tool)


def refused(tool: str) -> Message:
    return Message("tool", "Error: [not_verified] 본인 확인이 필요합니다.", tool_name=tool)


# ------------------------------------------------------------------------------------------ the detector


@pytest.mark.parametrize(
    ("text", "kinds"),
    [
        ("취소가 완료되었습니다.", ("cancel",)),
        ("주문 번호 O-91020을 취소하고 결제 금액 전액을 환불 처리했습니다.", ("cancel", "refund")),
        ("교환 요청이 접수되었습니다.", ("exchange",)),
        ("주문 번호 O-91019의 청바지 30인치를 32인치로 교환 접수하였습니다.", ("exchange",)),
        ("반품 접수를 완료했습니다.", ("return",)),
        ("주문 번호 O-91021의 배송지를 회사 주소로 변경했습니다.", ("address",)),
        ("쿠폰 발급이 완료되었습니다.", ("coupon",)),
        ("보상 쿠폰을 발급해 드렸어요.", ("coupon",)),
        ("티켓이 제출되었습니다.", ("ticket",)),
        ("담당 부서가 확인하도록 상담 티켓을 남겼습니다.", ("ticket",)),
        ("표준영님, 사람 상담원과 연결되었습니다.", ("handoff",)),
        ("주문 취소가 완료되었고, 환불은 3-5일 내에 이루어질 것입니다.", ("cancel", "refund")),
    ],
)
def test_saying_that_work_is_done_with_no_tool_result_is_an_unbacked_claim(text, kinds):
    claim = unbacked_claim(text, [Message("user", "취소해 주세요.")])
    assert claim is not None and claim.kinds == kinds and claim.sentence == text


@pytest.mark.parametrize(
    "text",
    [
        "본인 확인이 완료되었습니다.",
        "취소 처리해 드리겠습니다.",
        "환불이 완료되면 알려 드리겠습니다.",
        "O-1 주문을 단순 변심으로 취소할까요?",
        "이미 배송이 완료되었기 때문에 취소할 수 없습니다.",
        "배송지는 변경되지 않았습니다.",
        "9월 3일에 배송이 완료되었습니다.",
        "수령 후 7일이 지나 반품이 어렵습니다.",
        "주문이 정상적으로 접수되었습니다.",
        "반품 접수하셨다면 접수 번호를 알려 주세요.",
    ],
)
def test_promises_questions_refusals_and_other_news_are_not_claims(text):
    assert unbacked_claim(text, []) is None


def test_a_report_of_what_a_tool_showed_passes():
    text = "주문 O-6은 9월 1일에 단순 변심으로 이미 취소되었습니다."
    looked_up = [shown("get_order", order_id="O-6", status="cancelled", payment={"refund_won": 38900})]
    assert unbacked_claim(text, looked_up) is None
    assert unbacked_claim(text, []).kinds == ("cancel",)
    assert unbacked_claim(text, [refused("get_order")]) is not None
    assert unbacked_claim(text, [shown("get_order", order_id="O-1", status="paid")]) is not None


def test_a_changed_address_is_shown_only_by_the_write():
    text = "배송지가 회사 주소로 변경되었습니다."
    read = shown("get_order", order_id="O-1", shipping={"address_id": "AD-C-1-2", "address": "회사"})
    assert unbacked_claim(text, [read]) is not None
    assert unbacked_claim(text, [refused("change_shipping_address")]) is not None
    assert unbacked_claim(text, [shown("change_shipping_address", order_id="O-1")]) is None


def test_a_refund_is_shown_by_an_amount_or_by_the_cancelled_order():
    text = "네, 정확히 38,900원이 환불되었습니다."
    listed = shown("list_orders", orders=[{"order_id": "O-6", "status": "cancelled"}])
    assert unbacked_claim(text, [listed]) is None
    assert unbacked_claim(text, [shown("get_order", payment={"status": "paid", "refund_won": 38900})]) is None
    assert unbacked_claim(text, [shown("get_order", payment={"status": "paid", "refund_won": 0})]) is not None


def test_a_coupon_that_a_read_lists_may_be_reported():
    text = "하도경님, 이미 2,000원 보상 쿠폰이 발급되었습니다."
    listed = shown("get_order", compensation_coupons=[{"coupon_id": "CP-O-1-1", "amount_won": 2000}])
    assert unbacked_claim(text, [listed]) is None
    assert unbacked_claim(text, [shown("get_order", compensation_coupons=[])]) is not None


def test_one_kind_with_evidence_is_enough_and_the_first_bare_sentence_is_reported():
    cancelled = [shown("cancel_order", order_id="O-1", status="cancelled", refund_won=38900)]
    assert unbacked_claim("주문이 취소되어 쿠폰도 함께 처리되었습니다.", cancelled) is None
    text = "본인 확인이 완료되었습니다. 교환 접수를 완료했습니다. 취소도 했습니다."
    claim = unbacked_claim(text, cancelled)
    assert (claim.sentence, claim.kinds) == ("교환 접수를 완료했습니다.", ("exchange",))
    assert (claim.label, claim.tools) == ("교환 접수", "request_exchange")


# ------------------------------------------------------------------------------------------ in the loop

CLAIM = "주문 취소가 완료되었습니다."


def turn(tiny_engine, script, **config):
    engine = db.memory_engine(tiny_engine)
    ctx = ToolContext(now=NOW)
    state = new_state("system prompt")
    provider = ScriptedProvider(script)
    result = agent_turn(
        state,
        "O-1 주문을 취소해 주세요.",
        provider=provider,
        registry=TOY_REGISTRY,
        run_tool=lambda name, args: execute(TOY_REGISTRY, engine, ctx, name, args),
        config=RunConfig(**config),
    )
    turn.provider = provider
    return result, state, engine


def test_c1_holds_the_claim_back_and_the_retry_does_the_work(tiny_engine):
    script = [CLAIM, ToolCall("toy_cancel_order", {"order_id": "O-1"}), CLAIM]
    result, state, engine = turn(tiny_engine, script, claims="C1")
    assert result.reply == CLAIM and result.stop is None  # now the tool result shows it
    assert (state.held_claims, state.format_errors, state.stalls) == (1, 0, 0)
    held, notice = state.messages[3], state.messages[4]
    assert (held.content, held.delivered) == (CLAIM, False)
    expected = CLAIM_NOTICE.format(label="주문 취소", tools="cancel_order")
    assert (notice.harness, notice.content) == (True, expected)
    assert [log.format_error for log in state.llm_log] == ["unbacked_claim", None, None]
    assert db.dump_db(engine)["orders"][0]["status"] == "cancelled"


def test_c0_delivers_the_same_claim(tiny_engine):
    result, state, engine = turn(tiny_engine, [CLAIM])
    assert result.reply == CLAIM and state.held_claims == 0 and RunConfig().claims == "C0"
    assert db.dump_db(engine)["orders"][0]["status"] == "paid"  # the customer was told otherwise


def test_c1_ends_the_episode_rather_than_deliver_the_claim(tiny_engine):
    result, state, engine = turn(tiny_engine, [CLAIM] * 3, claims="C1")
    assert (result.reply, result.stop) == (None, "unbacked_claim") and "unbacked_claim" not in JUDGED
    assert state.held_claims == 3 and sum(m.harness for m in state.messages) == 2
    assert not [m for m in state.messages if m.role == "assistant" and m.delivered and m.content == CLAIM]
    assert db.dump_db(engine)["orders"][0]["status"] == "paid"


def test_c1_lets_an_honest_answer_through_after_the_notice(tiny_engine):
    honest = "아직 취소 처리를 하지 못했습니다. 취소 사유를 알려 주시겠어요?"
    result, state, _ = turn(tiny_engine, [CLAIM, honest], claims="C1")
    assert result.reply == honest and state.held_claims == 1


def test_c1_does_not_hold_a_report_of_a_lookup(tiny_engine):
    script = [ToolCall("toy_cancel_order", {"order_id": "O-1"}), "주문 O-1은 취소되었습니다."]
    _, _, engine = turn(tiny_engine, script)  # cancelled in an earlier conversation, so to speak
    state = new_state("system prompt")
    ctx = ToolContext(now=NOW)
    script = [ToolCall("toy_get_order", {"order_id": "O-1"}), "주문 O-1은 이미 취소되었습니다."]
    result = agent_turn(
        state,
        "O-1 주문 상태를 알려 주세요.",
        provider=ScriptedProvider(script),
        registry=TOY_REGISTRY,
        run_tool=lambda name, args: execute(TOY_REGISTRY, engine, ctx, name, args),
        config=RunConfig(claims="C1"),
    )
    assert result.reply == "주문 O-1은 이미 취소되었습니다." and state.held_claims == 0


def test_c2_samples_the_retry_but_not_the_first_attempt(tiny_engine):
    script = [CLAIM, ToolCall("toy_cancel_order", {"order_id": "O-1"}), CLAIM]
    turn(tiny_engine, script, claims="C2")
    assert [request["temperature"] for request in turn.provider.requests] == [0.0, 0.7, 0.7]
    turn(tiny_engine, script, claims="C1")
    assert [request["temperature"] for request in turn.provider.requests] == [0.0, 0.0, 0.0]


# ------------------------------------------------------------------------------------------ in the records


def llm(index, format_error=None, tool=None):
    calls = [tool] if tool else []
    return {"who": "agent", "index": index, "format_error": format_error, "tool_calls": calls}


def episode(messages, llm_calls, tool_calls=(), termination="user_stop", ok=True):
    return {
        "task_id": "dev-001",
        "trial": 0,
        "status": "completed",
        "termination": termination,
        "verdict": {"db_match": ok, "values": {}, "unexpected_writes": 0},
        "messages": messages,
        "llm_calls": llm_calls,
        "tool_calls": list(tool_calls),
    }


def test_delivered_claims_reads_any_record_and_skips_what_was_held():
    messages = [
        {"role": "user", "content": "취소해 주세요."},
        {"role": "assistant", "content": CLAIM, "delivered": False},
        {"role": "user", "content": "안내문", "harness": True},
        {"role": "assistant", "content": "교환 접수를 완료했습니다. 더 도와드릴까요?"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "cancel_order", "arguments": {}}]},
        {"role": "tool", "content": '{"status": "cancelled"}', "tool_name": "cancel_order"},
        {"role": "assistant", "content": CLAIM},
    ]
    assert analyze.delivered_claims(episode(messages, [])) == ["교환 접수를 완료했습니다."]


def test_after_a_held_claim_names_the_next_step():
    calls = [
        llm(0, "unbacked_claim"),
        llm(1, "unbacked_claim"),
        llm(2, tool={"name": "cancel_order"}),
        llm(3),
        llm(4, "unbacked_claim"),
        llm(5, tool={"name": "get_order"}),
        llm(6, "unbacked_claim"),
        llm(7, tool={"name": "cancel_order"}),
        llm(8, "unbacked_claim"),
        llm(9, "leaked_tool_call"),
        llm(10, "unbacked_claim"),
        llm(11),
        llm(12, "unbacked_claim"),
    ]
    tools = [
        {"agent_call": 2, "name": "cancel_order", "ok": True, "write": True},
        {"agent_call": 5, "name": "get_order", "ok": True, "write": False},
        {"agent_call": 7, "name": "cancel_order", "ok": False, "write": True},
    ]
    steps = analyze.after_a_held_claim(episode([], calls, tools))
    assert steps == ["claim_again", "write", "read", "write_failed", "format_error", "reply", "nothing"]


def test_claims_table_counts_per_run(tmp_path):
    told = episode([{"role": "assistant", "content": CLAIM}], [llm(0)], ok=False)
    guarded = episode(
        [{"role": "assistant", "content": CLAIM, "delivered": False}],
        [llm(0, "unbacked_claim")],
        termination="unbacked_claim",
        ok=False,
    )
    for name, rows in (("base", [told, told]), ("c1", [guarded])):
        (tmp_path / name).mkdir()
        lines = chr(10).join(json.dumps(row, ensure_ascii=False) for row in rows)
        (tmp_path / name / "episodes.jsonl").write_text(lines, encoding="utf-8")
    table = analyze.claims_table([tmp_path / "base", tmp_path / "c1"]).splitlines()
    assert table[2].startswith("| base | 2 | 2 | 0 | 2 | 0 |") and table[2].endswith("| 0 |")
    assert table[3].startswith("| c1 | 1 | 0 | 0 | 0 | 1 |") and table[3].endswith("| 1 | 1 |")


def test_a_held_claim_is_not_counted_as_a_format_error(tmp_path):
    row = episode([], [llm(0, "unbacked_claim"), llm(1, "empty"), llm(2, "stall")])
    row["verdict"] |= {"policy_violations": [], "policy_blocks": [], "auth_blocks": 0}
    row["wall_seconds"] = 1.0
    for call in row["llm_calls"]:
        call["dropped_calls"] = 0
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "episodes.jsonl").write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    counts = analyze.table([tmp_path / "run"]).splitlines()[-1]
    assert counts == "| run | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1.0 |"
