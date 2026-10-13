"""The claim guard (C1): a reply that says work is done which no tool result of the conversation shows."""

# [테스트 묶음] 7단계 완료 주장 가드(claims.py, C1·C2)를 세 층에서 확인한다: (1) 검출기 unbacked_claim이 어떤
# 문장을 '근거 없는 완료 안내'로 보는지, (2) 실제 쓰기 도구의 결과가 그 도구의 완료 문장을 뒷받침하는지, (3)
# 에이전트 루프에서 가드가 돌려보내고 다시 시키는지, 그리고 analyze가 기록에서 이것을 세는지.
# [왜 중요한가] 서비스에서 7B가 취소 도구를 부르지 않고 "주문이 취소되었습니다"라고 답한 일이 있었다
# (docs/experiments.md 7단계 "왜 하는가"). 검출기는 정규식이라, 놓치는 표현과 잘못 거는 표현(약속, 질문, 거절,
# 택배 완료 소식)을 사례로 고정해 둬야 측정 뒤에 바뀌지 않는다.

from __future__ import annotations

# functools.cache: 개발용 과제의 정답 재실행 결과를 한 번만 만든다. 루프 시험에는 작은 DB와 장난감 도구를,
# 실제 도구 결과 시험에는 seed DB와 실제 도구 등록부를 쓴다.
import functools
import json

import pytest
from conftest import NOW
from toy_tools import TOY_REGISTRY

from support_agent import analyze, db
from support_agent.agent import CLAIM_NOTICE, agent_turn, new_state
from support_agent.chat import Message, ScriptedProvider, ToolCall
from support_agent.claims import unbacked_claim
from support_agent.config import JUDGED, RunConfig
from support_agent.seed import build_seed_engine
from support_agent.tasks import load_tasks
from support_agent.toolkit import ConversationState, ToolContext, execute
from support_agent.tools import build_registry


# 성공한 도구 결과 메시지를 만든다 (결과 dict를 JSON 문자열로, 도구 이름과 함께).
def shown(tool: str, **result) -> Message:
    return Message("tool", json.dumps(result, ensure_ascii=False), tool_name=tool)


# 거절된 도구 결과 메시지. 'Error: [...]'로 시작하는 결과는 근거가 되지 않는다. 글은 실제 도구가 본인 확인
# 전에 돌려주는 것과 같다(identity_not_verified. 아래
# test_the_refusal_used_above_is_what_a_real_tool_answers가 확인한다).
def refused(tool: str) -> Message:
    return Message("tool", "Error: [identity_not_verified] 본인 확인이 되지 않았습니다.", tool_name=tool)


# ------------------------------------------------------------------------------------------ the detector


# 끝났다는 말 + 무엇에 대한 말(취소, 반품, 교환, 배송지, 쿠폰, 티켓, 상담원, 환불)이 있고 대화에 도구 결과가
# 없으면 걸려야 한다. 조사가 낀 표현('취소가 되었습니다'), '마쳤습니다', '이루어졌습니다', '남겼습니다', 앞
# 절은 본인 확인이고 뒤 절이 주장인 문장까지 사례로 고정한다. 종류(kinds)와 걸린 문장도 맞아야 한다.
@pytest.mark.parametrize(
    ("text", "kinds"),
    [
        ("취소가 완료되었습니다.", ("cancel",)),
        ("주문 번호 O-91020을 취소하고 결제 금액 전액을 환불 처리했습니다.", ("refund",)),
        ("주문 취소가 되었습니다.", ("cancel",)),
        ("취소 처리를 마쳤습니다.", ("cancel",)),
        ("환불이 이루어졌습니다.", ("refund",)),
        ("네, 무드등의 주문이 취소되었습니다.", ("cancel",)),
        ("본인 확인이 완료되었고, 취소 처리도 완료되었습니다.", ("cancel",)),
        ("교환 요청이 접수되었습니다.", ("exchange",)),
        ("주문 번호 O-91019의 청바지 30인치를 32인치로 교환 접수하였습니다.", ("exchange",)),
        ("반품 접수를 완료했습니다.", ("return",)),
        ("주문 번호 O-91021의 배송지를 회사 주소로 변경했습니다.", ("address",)),
        ("쿠폰 발급이 완료되었습니다.", ("coupon",)),
        ("보상 쿠폰을 발급해 드렸어요.", ("coupon",)),
        ("티켓이 제출되었습니다.", ("ticket",)),
        ("담당 부서가 확인하도록 상담 티켓을 남겼습니다.", ("ticket",)),
        ("표준영님, 사람 상담원과 연결되었습니다.", ("handoff",)),
        ("주문 취소가 완료되었고, 환불은 3-5일 내에 이루어질 것입니다.", ("cancel",)),
    ],
)
def test_saying_that_work_is_done_with_no_tool_result_is_an_unbacked_claim(text, kinds):
    claim = unbacked_claim(text, [Message("user", "취소해 주세요.")])
    assert claim is not None and claim.kinds == kinds and claim.sentence == text


# 걸리면 안 되는 문장: 앞으로 하겠다는 약속, 질문, 거절, 본인 확인 완료, 택배의 배송 완료 소식, '~고/~며'로
# 이어진 절에서 주장이 아닌 쪽. 잘못 걸리면 정답이 거절인 과제에서 가드가 쓰기를 부추기거나 에피소드를 실패로
# 끝내므로 사례로 막는다 (docs/experiments.md 7단계 "코드 검토 뒤에 고친 것").
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
        "본인 확인이 완료되었고, O-1 주문은 취소가 가능합니다.",
        "이미 배송이 완료되었으며 취소는 불가능합니다.",
        "본인 확인이 완료되었으며, 주문 O-1의 배송지는 서울특별시 중구입니다.",
        "9월 3일에 배송이 완료되었고 반품 기한은 9월 10일까지입니다.",
        "본인 확인이 완료되었고, 상담원 연결을 원하시면 말씀해 주세요.",
        "주문하신 상품은 등록된 배송지로 배송이 완료되었습니다.",
        "이메일 주소로 본인 확인이 완료되었습니다.",
    ],
)
def test_promises_questions_refusals_and_other_news_are_not_claims(text):
    assert unbacked_claim(text, []) is None


# 조회 결과도 근거가 된다: 이미 취소된 주문을 조회한 뒤 '이미 취소되었습니다'라고 전하는 것은 통과한다. 근거가
# 없거나, 거절된 조회이거나, 조회한 주문이 취소 상태가 아니면 걸린다.
def test_a_report_of_what_a_tool_showed_passes():
    text = "주문 O-6은 9월 1일에 단순 변심으로 이미 취소되었습니다."
    looked_up = [shown("get_order", order_id="O-6", status="cancelled", payment={"refund_won": 38900})]
    assert unbacked_claim(text, looked_up) is None
    assert unbacked_claim(text, []).kinds == ("cancel",)
    assert unbacked_claim(text, [refused("get_order")]) is not None
    assert unbacked_claim(text, [shown("get_order", order_id="O-1", status="paid")]) is not None


# 배송지 변경은 change_shipping_address의 성공만 근거로 본다. 조회 결과는 지금 배송지를 보여 줄 뿐
# '바뀌었다'는 것을 보여 주지 않기 때문이다 (docs/experiments.md 7단계 근거 표).
def test_a_changed_address_is_shown_only_by_the_write():
    text = "배송지가 회사 주소로 변경되었습니다."
    read = shown("get_order", order_id="O-1", shipping={"address_id": "AD-C-1-2", "address": "회사"})
    assert unbacked_claim(text, [read]) is not None
    assert unbacked_claim(text, [refused("change_shipping_address")]) is not None
    assert unbacked_claim(text, [shown("change_shipping_address", order_id="O-1")]) is None


# 환불 주장의 근거: 취소된 주문 상태, 또는 0이 아닌 refund_won. 환불액이 0인 결과는 근거가 아니다.
def test_a_refund_is_shown_by_an_amount_or_by_the_cancelled_order():
    text = "네, 정확히 38,900원이 환불되었습니다."
    listed = shown("list_orders", orders=[{"order_id": "O-6", "status": "cancelled"}])
    assert unbacked_claim(text, [listed]) is None
    assert unbacked_claim(text, [shown("get_order", payment={"status": "paid", "refund_won": 38900})]) is None
    assert unbacked_claim(text, [shown("get_order", payment={"status": "paid", "refund_won": 0})]) is not None


# 조회 결과에 이미 발급된 보상 쿠폰이 있으면 '이미 발급되었습니다'는 통과하고, 쿠폰 목록이 비어 있으면 걸린다.
def test_a_coupon_that_a_read_lists_may_be_reported():
    text = "하도경님, 이미 2,000원 보상 쿠폰이 발급되었습니다."
    listed = shown("get_order", compensation_coupons=[{"coupon_id": "CP-O-1-1", "amount_won": 2000}])
    assert unbacked_claim(text, [listed]) is None
    assert unbacked_claim(text, [shown("get_order", compensation_coupons=[])]) is not None


# 서비스 전용 경우: 큰 환불이 승인 대기열에 올라가면 도구는 approval_required 오류로 답한다. 이 오류는 '요청이
# 접수되었다'는 말의 근거로만 인정하고, '취소가 완료되었다', '환불되었다', 다른 종류의 접수는 여전히 막는다.
# 그냥 거절된 결과는 접수의 근거도 아니다.
def test_a_request_queued_for_approval_counts_as_received():
    queued = Message(
        "tool",
        "Error: [approval_required] 환불액 108,200원은 담당자 승인이 필요해 승인 대기열에 올렸습니다.",
        tool_name="cancel_order",
    )
    assert unbacked_claim("취소 요청이 접수되었으며 담당자 확인 후 처리됩니다.", [queued]) is None
    assert unbacked_claim("취소 요청을 접수했습니다. 확인 후 안내드리겠습니다.", [queued]) is None
    assert unbacked_claim("취소가 완료되었습니다.", [queued]) is not None  # not done, only queued
    assert unbacked_claim("108,200원이 환불되었습니다.", [queued]) is not None
    assert unbacked_claim("쿠폰 요청이 접수되었습니다.", [queued]) is not None  # another kind
    assert unbacked_claim("취소 요청이 접수되었습니다.", [refused("cancel_order")]) is not None


# refused()가 만드는 거절 결과는 실제 도구가 본인 확인 전에 돌려주는 글과 같아야 한다. 오류 코드나 문구가
# 실제와 어긋나면 위 테스트들이 실제로 나오지 않는 결과로 검출기를 보게 된다.
def test_the_refusal_used_above_is_what_a_real_tool_answers():
    engine = db.memory_engine(build_seed_engine())
    real = execute(build_registry(), engine, ToolContext(now=NOW), "get_order", {"order_id": "O-90001"})
    assert not real.ok and real.error_code == "identity_not_verified"
    assert refused("get_order").content == real.content


# 한 문장이 여러 종류에 걸치면 하나만 근거가 있어도 통과한다(관대한 쪽, 7단계 규칙). 걸린 것이 여러 문장이면
# 근거 없는 첫 문장을 보고하고, 안내문에 넣을 이름(교환 접수)과 도구(request_exchange)가 그 종류의 것이어야
# 한다.
def test_one_kind_with_evidence_is_enough_and_the_first_bare_sentence_is_reported():
    cancelled = [shown("cancel_order", order_id="O-1", status="cancelled", refund_won=38900)]
    assert unbacked_claim("주문이 취소되어 쿠폰도 함께 처리되었습니다.", cancelled) is None
    text = "본인 확인이 완료되었습니다. 교환 접수를 완료했습니다. 취소도 했습니다."
    claim = unbacked_claim(text, cancelled)
    assert (claim.sentence, claim.kinds) == ("교환 접수를 완료했습니다.", ("exchange",))
    assert (claim.label, claim.tools) == ("교환 접수", "request_exchange")


# ------------------------------------------------------------------------------------------ real tool answers

# 위 테스트들은 근거 JSON을 직접 만들었다. 아래는 실제 도구가 execute()를 거쳐 쓰는 JSON이 근거 패턴에
# 걸리는지 확인한다. 도구 출력 형식이 바뀌어 가드가 진짜 완료까지 막는 회귀를 잡는다.
# The patterns of claims.KINDS read the JSON that toolkit.execute writes from the answers of tools.py; the
# tests above build their own. One completion sentence per write tool: its real answer must back it.
DONE_BY_TOOL = {
    "cancel_order": "주문이 취소되었습니다.",
    "request_return": "반품 접수를 완료했습니다.",
    "request_exchange": "교환 접수가 완료되었습니다.",
    "change_shipping_address": "배송지를 변경했습니다.",
    "issue_compensation_coupon": "보상 쿠폰을 발급해 드렸습니다.",
    "create_ticket": "상담 티켓을 남겼습니다.",
    "transfer_to_human": "상담원과 연결되었습니다.",
}


# 개발용 과제의 정답 동작을 과제마다 새 seed DB 복사본에서 실행해, 쓰기 도구마다 첫 결과 문자열을 모은다. 정답
# 동작은 모두 성공해야 한다(실패하면 과제 파일이 잘못된 것). 무거우므로 한 번만 계산한다.
@functools.cache
def real_answers() -> dict[str, str]:
    """The first answer of each write tool when the gold actions of the development tasks are replayed."""
    registry, seed, out = build_registry(), build_seed_engine(), {}
    for task in load_tasks("dev"):
        engine = db.memory_engine(seed)
        ctx = ToolContext(
            now=task.now, state=ConversationState(task.customer_id if task.gold_verified else None)
        )
        for action in task.gold_actions:
            result = execute(registry, engine, ctx, action.tool, action.args)
            assert result.ok, (task.id, result.content)
            out.setdefault(action.tool, result.content)
        engine.dispose()
    return out


# 모든 쓰기 도구에 완료 문장이 하나씩 있어야 한다. 새 쓰기 도구를 더하면 이 테스트가 실패해 근거 패턴을 함께
# 정하게 한다.
def test_every_write_tool_has_a_completion_sentence():
    assert sorted(DONE_BY_TOOL) == sorted(name for name, spec in build_registry().items() if spec.write)


# 각 완료 문장은 근거 없이는 주장으로 걸리고, 그 도구의 실제 결과가 있으면 통과해야 한다. 취소와 반품의 결과는
# '환불 처리했습니다'의 근거도 된다.
@pytest.mark.parametrize(
    ("tool", "text"),
    [*DONE_BY_TOOL.items(), ("cancel_order", "환불 처리했습니다."), ("request_return", "환불 처리했습니다.")],
    ids=[*DONE_BY_TOOL, "cancel_order-refund", "request_return-refund"],
)
def test_the_real_answer_of_a_write_tool_backs_its_completion_sentence(tool, text):
    assert unbacked_claim(text, []) is not None  # the sentence is a claim
    assert unbacked_claim(text, [Message("tool", real_answers()[tool], tool_name=tool)]) is None


# ------------------------------------------------------------------------------------------ in the loop

# 루프 시험에 쓰는 근거 없는 완료 문장.
CLAIM = "주문 취소가 완료되었습니다."


# 장난감 도구와 작은 DB 복사본으로 'O-1 취소' 한 턴을 돌린다. 설정(claims 등)은 키워드로 받는다. 마지막
# 제공자를 turn.provider에 남겨 테스트가 요청의 temperature를 볼 수 있게 한다.
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


# C1: 근거 없는 주장은 전달하지 않고(delivered=False) 안내문(종류 이름과 도구 이름을 넣은 CLAIM_NOTICE)을 넣어
# 다시 시킨다. 다시 시킨 뒤 도구로 실제 취소하면 같은 문장이 이제 근거가 있어 전달된다. 돌려보낸 응답은 형식
# 오류나 멈춤으로 세지 않고 held_claims로 따로 센다.
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


# 기본값 C0(가드 끔)에서는 같은 주장이 그대로 고객에게 간다. DB는 그대로인데 고객은 취소되었다고 듣는, 가드가
# 막으려는 상황을 보여 준다.
def test_c0_delivers_the_same_claim(tiny_engine):
    result, state, engine = turn(tiny_engine, [CLAIM])
    assert result.reply == CLAIM and state.held_claims == 0 and RunConfig().claims == "C0"
    assert db.dump_db(engine)["orders"][0]["status"] == "paid"  # the customer was told otherwise


# 세 번째에도 같은 주장이면 전달하지 않고 unbacked_claim으로 끝낸다(판정하는 종료가 아니라 실패로 센다). 멈춤
# 가드와 달리 끝까지 전달하지 않는 이유: '취소되었습니다'를 들은 고객은 더 묻지 않고 떠나기 때문이다
# (docs/experiments.md 7단계 후보).
def test_c1_ends_the_episode_rather_than_deliver_the_claim(tiny_engine):
    result, state, engine = turn(tiny_engine, [CLAIM] * 3, claims="C1")
    assert (result.reply, result.stop) == (None, "unbacked_claim") and "unbacked_claim" not in JUDGED
    assert state.held_claims == 3 and sum(m.harness for m in state.messages) == 2
    assert not [m for m in state.messages if m.role == "assistant" and m.delivered and m.content == CLAIM]
    assert db.dump_db(engine)["orders"][0]["status"] == "paid"


# 안내문 뒤에 사실대로 답하면(아직 처리하지 못했다, 사유를 묻는다) 그대로 전달된다. 처리할 수 없는 일이면
# 사실대로 안내하는 것이 안내문이 열어 둔 두 번째 길이다.
def test_c1_lets_an_honest_answer_through_after_the_notice(tiny_engine):
    honest = "아직 취소 처리를 하지 못했습니다. 취소 사유를 알려 주시겠어요?"
    result, state, _ = turn(tiny_engine, [CLAIM, honest], claims="C1")
    assert result.reply == honest and state.held_claims == 1


# 앞선 대화에서 이미 취소된 주문을 이번 대화에서 조회하고 '이미 취소되었습니다'라고 전하는 것은 근거가
# 있으므로 C1이 막지 않는다.
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


# C2는 돌려보낸 뒤 그 턴의 남은 호출만 temperature 0.7로 뽑고 첫 호출은 0.0이다. C1은 끝까지 0.0이다.
# temperature 0인 모델은 안내문을 받고도 같은 문장을 되풀이하기 쉬워서 둔 후보다.
def test_c2_samples_the_retry_but_not_the_first_attempt(tiny_engine):
    script = [CLAIM, ToolCall("toy_cancel_order", {"order_id": "O-1"}), CLAIM]
    turn(tiny_engine, script, claims="C2")
    assert [request["temperature"] for request in turn.provider.requests] == [0.0, 0.7, 0.7]
    turn(tiny_engine, script, claims="C1")
    assert [request["temperature"] for request in turn.provider.requests] == [0.0, 0.0, 0.0]


# ------------------------------------------------------------------------------------------ in the records


# 기록(episodes.jsonl)의 에이전트 LLM 호출 한 줄을 흉내 낸다.
def llm(index, format_error=None, tool=None):
    calls = [tool] if tool else []
    return {"who": "agent", "index": index, "format_error": format_error, "tool_calls": calls}


# analyze가 읽는 에피소드 기록 한 줄을 최소한의 키로 만든다.
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


# delivered_claims는 고객에게 실제로 전달된 응답만 본다: 가드가 돌려보낸 응답(delivered=False)은 빼고, 각
# 응답은 그보다 앞선 도구 결과만 근거로 본다. 그래서 취소 결과가 나온 뒤의 같은 주장은 걸리지 않고, 근거 없던
# 교환 주장만 남는다. 가드 없이 돈 실행의 기록에도 같은 검출기를 대어 '거짓 완료가 전달된 에피소드'를 셀 수
# 있다.
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


# 돌려보낸 응답마다 다음 호출이 무엇이었는지 분류한다: 같은 주장, 쓰기 성공, 조회, 쓰기 실패, 형식 오류, 다른
# 말, 끝(다음 호출 없음). 7단계 결과표의 '그 뒤' 열이 이 분류다.
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


# claims_table의 한 줄 값을 확인한다: 가드 없는 실행은 거짓 완료가 두 에피소드에 전달되었고, C1 실행은 전달 0,
# 돌려보낸 응답 1, 그 뒤 끝 1, 가드로 끝난 에피소드 1이다.
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


# analyze table의 형식 오류 수에는 돌려보낸 완료 주장과 멈춤이 들어가지 않는다(여기서는 빈 응답 1개만).
# summary.json과 table이 달랐던 것을 같게 맞춘 뒤의 회귀 테스트다 (docs/experiments.md 7단계 "코드 검토 뒤에
# 고친 것").
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
