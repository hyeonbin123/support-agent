"""The 14 domain tools, always through toolkit.execute(). The shop rows are described in factories.py."""

# [파일 역할] 도메인 도구 14개(tools.py)를 항상 toolkit.execute()를 거쳐 검증하는 테스트 묶음이다. 등록부와
# 스키마, 정상 경로의 출력 키, 읽기 도구의 내용, 본인 확인, 존재·소유 검사, 규정 검사(P0/P1), 쓰기의 효과,
# 반복 호출과 결정성, 규정 문서와 코드의 숫자 일치를 차례로 본다.
# [왜 중요한가] 도구는 에이전트 루프·정답 재실행·서비스·MCP 서버가 모두 같은 경로로 부르는 공통 부품이고,
# 판정은 최종 DB 상태로 한다. 출력 키가 docs/design.md의 도구 표와 어긋나거나, 거절된 호출이 DB를 바꾸거나,
# 같은 호출이 다른 상태를 만들면 모든 측정과 서비스가 함께 틀어진다.
# [방법] conftest의 작은 DB에 factories.py가 행을 더한 가게(shop 픽스처: 고객 C-1·C-2, 주문 O-1~O-6·O-9, 상품
# P-1·P-2)를 테스트마다 새 복사본으로 쓴다. 각 주문의 상태와 날짜는 factories.py 머리 docstring에 있다.

import ast
import json
from datetime import timedelta
from pathlib import Path

# factories: 가게 행을 더하고(extend_tiny_rows), 보상 쿠폰 행을 직접 넣고, KST 시각을 만드는 도우미.
import pytest
from conftest import NOW
from factories import add_compensation_coupon, extend_tiny_rows, kst
from sqlalchemy.orm import Session

# support_agent 패키지 경로로 tools.py·rules.py·prompts/policy.md 원문을 읽는다. rules는 규정 숫자(반품 기한,
# 쿠폰 금액 등)의 출처, labels.choices는 enum의 '코드=한국어' 목록 문자열, HANDOFF_MESSAGE는 이관 시 고객에게
# 가는 고정 안내문이다.
import support_agent
from support_agent import db, rules
from support_agent.config import HANDOFF_MESSAGE
from support_agent.labels import choices
from support_agent.toolkit import ConversationState, ToolContext, ToolResult, execute
from support_agent.tools import build_registry

# REGISTRY: 실제 도구 등록부(글 상담용, find_customer 포함). NO_IDENTITY_NEEDED: 본인 확인 전에도 쓸 수 있는
# 도구(docs/design.md의 identity_not_verified 예외 목록에서 음성 V4 전용 verify_caller를 뺀 것). WRITE_TOOLS:
# DB를 바꾸는 도구 7개.
REGISTRY = build_registry()
PACKAGE_DIR = Path(support_agent.__file__).resolve().parent
NO_IDENTITY_NEEDED = {"find_customer", "get_product", "transfer_to_human", "think"}
WRITE_TOOLS = {
    "cancel_order",
    "request_return",
    "request_exchange",
    "change_shipping_address",
    "issue_compensation_coupon",
    "create_ticket",
    "transfer_to_human",
}


# 테스트마다 작은 DB를 복사하고 가게 행을 더한 새 엔진을 준다. 테스트끼리 쓰기가 섞이지 않게 하려고 매번
# 복사한다.
@pytest.fixture
def shop(tiny_engine):
    engine = db.memory_engine(tiny_engine)
    with Session(engine) as session:
        extend_tiny_rows(session)
        session.commit()
    yield engine
    engine.dispose()


# 도구 문맥을 만든다. customer_id가 있으면 이미 본인 확인이 끝난 대화로, None이면 확인 전 대화로 시작한다.
# p1으로 규정 검사 방식(P1 막음/P0 기록 후 통과), now로 고정 시각을 바꾼다.
def make_ctx(customer_id: str | None = "C-1", *, p1: bool = True, now=NOW) -> ToolContext:
    return ToolContext(now=now, enforce_policy=p1, state=ConversationState(verified_customer_id=customer_id))


# 도구 하나를 execute()로 부른다. 핸들러를 직접 부르지 않고 실행 경로 전체(인자 검증, 세션, 롤백)를 지나게
# 한다.
def call(engine, ctx: ToolContext, name: str, /, **args) -> ToolResult:
    return execute(REGISTRY, engine, ctx, name, args)


# 성공해야 하는 호출: 실패하면 도구가 돌려준 오류 문장을 보여 주며 실패하고, 성공하면 JSON 출력을 dict로
# 돌려준다.
def ok(engine, ctx: ToolContext, name: str, /, **args) -> dict:
    result = call(engine, ctx, name, **args)
    assert result.ok, result.content
    return json.loads(result.content)


# 실패해야 하는 호출: 오류 문장이 'Error: [code] '로 시작하는 약속된 형식인지 확인하고 오류 코드를 돌려준다.
def code(engine, ctx: ToolContext, name: str, /, **args) -> str | None:
    result = call(engine, ctx, name, **args)
    assert not result.ok, result.content
    assert result.content.startswith(f"Error: [{result.error_code}] ")
    return result.error_code


# 비교에서 빼는 컬럼까지 모두 담은 덤프(ignore=빈 집합)로 한 테이블의 행을 기본 키 문자열로 찾게 한다. 티켓
# 본문처럼 판정에서 빼는 값도 실제로 저장됐는지 보려고 쓴다.
def rows(engine, table: str) -> dict[str, dict]:
    """Rows of one table keyed by their primary key values joined with '/'."""
    pk = [c.name for c in db.Base.metadata.tables[table].primary_key.columns]
    return {"/".join(str(r[k]) for k in pk): r for r in db.dump_db(engine, ignore=frozenset())[table]}


# 등록부: 도구 목록, 쓰기·종료 표시, 스키마 크기와 모양.
# ------------------------------------------------------------------------------------------------ registry


# 등록부가 docs/design.md의 도구 표와 같은 14개를 같은 순서로 갖고, 쓰기 도구 집합, 대화를 끝내는
# 도구(transfer_to_human 하나), 순서 무관 인자(request_return의 line_nos)가 설계대로인지 본다. 쓰기 표시는
# 서비스의 승인·MCP 쓰기 범위가 기대는 값이다.
def test_the_registry_has_the_14_tools_of_the_design():
    assert list(REGISTRY) == [
        "find_customer",
        "get_customer",
        "list_orders",
        "get_order",
        "get_product",
        "track_shipment",
        "cancel_order",
        "request_return",
        "request_exchange",
        "change_shipping_address",
        "issue_compensation_coupon",
        "create_ticket",
        "transfer_to_human",
        "think",
    ]
    assert {name for name, spec in REGISTRY.items() if spec.write} == WRITE_TOOLS
    assert [name for name, spec in REGISTRY.items() if spec.terminates] == ["transfer_to_human"]
    assert REGISTRY["request_return"].unordered_args == ("line_nos",)


# 도구 스키마 전체가 9,000자 미만인지 본다. 에이전트 기본 컨텍스트(num_ctx 16384)를 규정 문서, 지침, 대화와
# 나눠 써야 하기 때문이다. 길이를 출력해 두어 바뀔 때 눈으로 볼 수 있게 한다.
def test_the_schemas_are_small_enough_for_a_16k_context():
    text = json.dumps([spec.schema() for spec in REGISTRY.values()], ensure_ascii=False)
    print(f"\ntool schemas: {len(text)} characters")
    assert len(text) < 9000


# 모든 도구 설명이 한국어이고, 모든 인자에 설명이 있으며, 인자 타입이 string·integer·array뿐(평평한
# 스키마)인지 본다. enum 인자는 ASCII 코드이고 설명에 '코드=한국어' 목록이 들어가야 한다(docs/design.md
# "도구"). create_ticket의 order_id는 선택(기본 빈 문자열), line_nos의 원소는 1~999 정수로 제한되는지도
# 확인한다.
def test_schemas_are_korean_flat_and_list_the_enum_codes():
    for spec in REGISTRY.values():
        schema = spec.schema()
        assert any("가" <= ch <= "힣" for ch in schema["description"]), spec.name
        for name, prop in schema["parameters"]["properties"].items():
            assert prop.get("description"), f"{spec.name}.{name}"
            assert prop["type"] in {"string", "integer", "array"}
    props = {name: spec.schema()["parameters"] for name, spec in REGISTRY.items()}
    assert props["cancel_order"]["properties"]["reason"]["enum"] == [m.value for m in db.CancelReason]
    for tool_name, field, enum_cls in [
        ("cancel_order", "reason", db.CancelReason),
        ("request_return", "reason", db.RequestReason),
        ("request_exchange", "reason", db.RequestReason),
        ("issue_compensation_coupon", "reason", db.CompensationReason),
        ("create_ticket", "category", db.TicketCategory),
        ("transfer_to_human", "reason", db.HandoffReason),
    ]:
        assert choices(enum_cls) in props[tool_name]["properties"][field]["description"]
    assert props["create_ticket"]["required"] == ["category", "body"]
    assert props["create_ticket"]["properties"]["order_id"]["default"] == ""
    assert props["request_return"]["properties"]["line_nos"]["items"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 999,
    }


# 정상 경로와 출력 키: 도구마다 성공하는 인자 한 벌과 docs/design.md 도구 표의 출력 키 목록.
# ------------------------------------------------------------------------------ happy paths and output keys

# 도구 이름 → (성공하는 인자, 출력 JSON의 키 순서). 다른 테스트들도 이 인자를 '그 도구를 정상적으로 부르는
# 법'으로 재사용한다(본인 확인 전 거절, 반복 쓰기).
HAPPY: dict[str, tuple[dict, list[str]]] = {
    "find_customer": (
        {"name": "김하준", "contact": "010-0000-0001"},
        ["customer_id", "name", "grade", "grade_label"],
    ),
    "get_customer": (
        {"customer_id": "C-1"},
        [
            "customer_id",
            "name",
            "phone",
            "email",
            "grade",
            "grade_label",
            "joined_at",
            "addresses",
            "compensation_coupons",
        ],
    ),
    "list_orders": ({"customer_id": "C-1"}, ["customer_id", "orders"]),
    "get_order": (
        {"order_id": "O-3"},
        [
            "order_id",
            "status",
            "status_label",
            "ordered_at",
            "items",
            "items_won",
            "shipping_fee_won",
            "discount_won",
            "total_won",
            "payment",
            "shipping",
            "requests",
            "compensation_coupons",
            "cancelled_at",
            "cancel_reason",
            "cancel_reason_label",
        ],
    ),
    "get_product": ({"product_id": "P-1"}, ["product_id", "name", "category", "variants"]),
    "track_shipment": (
        {"order_id": "O-4"},
        [
            "order_id",
            "carrier",
            "tracking_no",
            "status",
            "status_label",
            "shipped_at",
            "delivered_at",
            "promised_by",
        ],
    ),
    "cancel_order": (
        {"order_id": "O-1", "reason": "changed_mind"},
        ["order_id", "status", "status_label", "refund_won", "refund_method", "refund_method_label"],
    ),
    "request_return": (
        {"order_id": "O-3", "line_nos": [1], "reason": "changed_mind"},
        ["request_id", "order_id", "line_nos", "refund_won", "return_fee_won"],
    ),
    "request_exchange": (
        {"order_id": "O-3", "line_no": 1, "new_variant_id": "V-1-02", "reason": "changed_mind"},
        ["request_id", "order_id", "line_no", "new_variant_id", "new_option_label"],
    ),
    "change_shipping_address": (
        {"order_id": "O-1", "address_id": "AD-C-1-2"},
        ["order_id", "address_id", "recipient", "postal_code", "address"],
    ),
    "issue_compensation_coupon": (
        {"order_id": "O-4", "reason": "delivery_delay"},
        ["coupon_id", "amount_won", "expires_at"],
    ),
    "create_ticket": (
        {"category": "refund", "body": "환불 입금이 늦어지고 있어 확인을 요청함"},
        ["ticket_id", "category", "category_label"],
    ),
    "transfer_to_human": (
        {"reason": "customer_request", "summary": "고객이 상담원 연결을 요청함"},
        ["handoff_id", "message"],
    ),
    "think": ({"thought": "주문 상태를 먼저 확인해야 한다."}, ["ok"]),
}


# HAPPY 표가 등록부의 모든 도구를 빠짐없이 다루는지 본다. 도구를 새로 넣고 표를 잊으면 여기서 걸린다.
def test_the_happy_table_covers_every_tool():
    assert set(HAPPY) == set(REGISTRY)


# 도구마다 정상 호출의 출력 키가 설계 표와 순서까지 같은지, 그리고 쓰기 도구만 DB를 바꾸고 읽기 도구는 DB를
# 그대로 두는지 본다. 출력 키는 에이전트 프롬프트와 과제의 전달 값 검증이 기대는 계약이다.
@pytest.mark.parametrize("name", list(HAPPY))
def test_happy_path_output_keys_match_the_design_table(shop, name):
    args, keys = HAPPY[name]
    before = db.dump_db(shop)
    out = ok(shop, make_ctx(), name, **args)
    assert list(out) == keys
    assert (db.dump_db(shop) != before) == REGISTRY[name].write


# 중첩된 출력(주소, 주문 목록, 주문 줄, 결제, 배송, 접수 내역, 상품 옵션)의 키도 설계 표와 같은지 본다. 접수
# 내역 키를 보려고 먼저 반품을 하나 접수한다.
def test_nested_output_keys_match_the_design_table(shop):
    ctx = make_ctx()
    ok(shop, ctx, "request_return", order_id="O-3", line_nos=[2], reason="defective")
    customer = ok(shop, ctx, "get_customer", customer_id="C-1")
    assert list(customer["addresses"][0]) == [
        "address_id",
        "label",
        "recipient",
        "postal_code",
        "address",
        "is_default",
    ]
    orders = ok(shop, ctx, "list_orders", customer_id="C-1")["orders"]
    assert list(orders[0]) == [
        "order_id",
        "ordered_at",
        "status",
        "status_label",
        "total_won",
        "item_summary",
    ]
    order = ok(shop, ctx, "get_order", order_id="O-3")
    assert list(order["items"][0]) == [
        "line_no",
        "product_id",
        "variant_id",
        "product_name",
        "option_label",
        "quantity",
        "unit_price_won",
        "status",
        "status_label",
    ]
    assert list(order["payment"]) == [
        "method",
        "method_label",
        "amount_won",
        "status",
        "status_label",
        "refund_won",
    ]
    assert list(order["shipping"]) == ["address_id", "recipient", "postal_code", "address"]
    assert list(order["requests"][0]) == [
        "request_id",
        "kind",
        "kind_label",
        "reason",
        "reason_label",
        "line_nos",
        "refund_won",
        "return_fee_won",
        "created_at",
    ]
    product = ok(shop, ctx, "get_product", product_id="P-1")
    assert list(product["variants"][0]) == ["variant_id", "option_label", "price_won", "stock"]


# 읽기 도구의 내용: 정렬, 코드와 한국어 표기, KST 시각, 판단이 아닌 사실만.
# ------------------------------------------------------------------------------------------- read contents


# list_orders가 최근 주문부터, 같은 시각이면 주문 번호로 정렬하는지, 상품 요약이 '무선 이어폰 외 1건'처럼
# 나오는지, 한 항목의 키·값이 정확한지 본다. 순서가 실행마다 바뀌면 같은 입력에 다른 도구 출력이 나와 재현성이
# 깨진다.
def test_list_orders_is_newest_first_with_the_id_as_tie_break(shop):
    orders = ok(shop, make_ctx(), "list_orders", customer_id="C-1")["orders"]
    assert [o["order_id"] for o in orders] == ["O-5", "O-1", "O-2", "O-6", "O-3", "O-4"]
    by_id = {o["order_id"]: o for o in orders}
    assert by_id["O-3"]["item_summary"] == "무선 이어폰 외 1건"
    assert by_id["O-4"]["item_summary"] == "무선 이어폰"
    assert by_id["O-5"] == {
        "order_id": "O-5",
        "ordered_at": "2026-09-12 09:00",
        "status": "preparing",
        "status_label": "배송준비중",
        "total_won": 38900,
        "item_summary": "무선 이어폰",
    }


# get_order가 금액 내역(상품 68,300 + 배송비 3,000 = 71,300원), 상태 코드와 한국어 표기, KST 시각,
# 결제·배송·접수 내역, 취소 정보(없으면 None)를 주는지 본다. 취소된 주문(O-6)은 취소 시각·사유와 환불 대기
# 결제 상태를 보여 줘야 한다.
def test_get_order_gives_codes_labels_and_kst_times(shop):
    ctx = make_ctx()
    order = ok(shop, ctx, "get_order", order_id="O-3")
    assert (order["items_won"], order["shipping_fee_won"], order["discount_won"]) == (68300, 3000, 0)
    assert order["total_won"] == 71300 and order["ordered_at"] == "2026-09-05 09:00"
    assert (order["status"], order["status_label"]) == ("delivered", "배송완료")
    assert order["items"][1]["product_id"] == "P-2" and order["items"][1]["quantity"] == 2
    assert order["payment"]["method_label"] == "신용카드" and order["payment"]["refund_won"] == 0
    assert order["shipping"]["address_id"] == "AD-C-1-1"
    assert order["requests"] == []
    assert (order["cancelled_at"], order["cancel_reason"], order["cancel_reason_label"]) == (None, None, None)

    cancelled = ok(shop, ctx, "get_order", order_id="O-6")
    assert cancelled["cancelled_at"] == "2026-09-09 10:00"
    assert (cancelled["cancel_reason"], cancelled["cancel_reason_label"]) == ("changed_mind", "단순 변심")
    assert cancelled["payment"]["status"] == "refund_pending"


# 반품 기한이 지난 O-4의 get_order·track_shipment 출력에 계산된 기한(2026-09-09)이나 'deadline'·'returnable'
# 같은 판단이 없는지 본다. 읽기 도구가 판단을 주면 P0와 P1의 차이를 도구 출력이 메워 버리므로 사실만
# 준다(docs/experiments.md 공통 규칙 "도구와 규정").
def test_read_tools_give_facts_not_judgements(shop):
    ctx = make_ctx()
    text = call(shop, ctx, "get_order", order_id="O-4").content
    text += call(shop, ctx, "track_shipment", order_id="O-4").content
    assert "2026-09-09" not in text and "deadline" not in text and "returnable" not in text


# track_shipment가 수령·출고 시각을 KST 'YYYY-MM-DD HH:MM'으로, 도착 예정일을 날짜로 주고, 아직 출고 전이면
# None을 주는지 본다. DB에는 UTC로 저장하고 출력에서만 KST로 적는 설계다.
def test_track_shipment_formats_times_in_kst(shop):
    out = ok(shop, make_ctx(), "track_shipment", order_id="O-4")
    assert out["delivered_at"] == "2026-09-02 14:20" and out["promised_by"] == "2026-08-30"
    assert out["shipped_at"] == "2026-08-29 09:00" and out["status_label"] == "배송완료"
    pending = ok(shop, make_ctx(), "track_shipment", order_id="O-1")
    assert (pending["shipped_at"], pending["delivered_at"]) == (None, None)


# get_product가 본인 확인 없이도 되고(make_ctx(None)), 옵션을 id 순으로 재고와 함께 주는지 본다. 재고 0인
# 옵션도 숨기지 않는다.
def test_get_product_lists_variants_by_id(shop):
    out = ok(shop, make_ctx(None), "get_product", product_id="P-1")
    assert [(v["variant_id"], v["stock"]) for v in out["variants"]] == [
        ("V-1-01", 5),
        ("V-1-02", 3),
        ("V-1-03", 0),
    ]


# 본인 확인: 이름과 연락처가 모두 맞아야 하고, 대화당 고객 한 명, 확인 전에는 고객 정보에 손댈 수 없다.
# ------------------------------------------------------------------------------------------------ identity


# 연락처를 여러 꼴(하이픈, 숫자만, 띄어쓰기, 이메일, 대소문자·앞뒤 공백)로 줘도, 이름 사이에 공백이 있어도
# 같은 고객으로 확인되고 대화 상태에 확인된 고객이 기록되는지 본다. 고객이 말하는 표기가 제각각이라 정규화한
# 뒤 비교한다(docs/design.md customer_not_found 설명).
@pytest.mark.parametrize(
    "contact", ["010-0000-0001", "01000000001", "010 0000 0001", "hajun@example.com", " Hajun@Example.COM "]
)
def test_find_customer_normalises_the_contact_and_verifies(shop, contact):
    ctx = make_ctx(None)
    out = ok(shop, ctx, "find_customer", name="김 하준", contact=contact)
    assert out == {"customer_id": "C-1", "name": "김하준", "grade": "normal", "grade_label": "일반"}
    assert ctx.state.verified_customer_id == "C-1"


# 다른 고객의 전화번호, 다른 고객의 이름, 이름 일부, 빈 연락처, '모름', 빈 이름은 모두 customer_not_found이고
# 확인 상태가 남지 않는지 본다. 이름과 연락처 중 하나만 맞아도 통과시키면 남의 주문을 처리할 수 있게 된다.
@pytest.mark.parametrize(
    ("name", "contact"),
    [
        ("김하준", "010-0000-0002"),  # another customer's phone
        ("이서연", "hajun@example.com"),  # another customer's name
        ("김하", "010-0000-0001"),
        ("김하준", ""),
        ("김하준", "전화번호 모름"),
        ("", "010-0000-0001"),
    ],
)
def test_find_customer_needs_both_name_and_contact(shop, name, contact):
    ctx = make_ctx(None)
    assert code(shop, ctx, "find_customer", name=name, contact=contact) == "customer_not_found"
    assert ctx.state.verified_customer_id is None


# 한 대화에서 한 고객을 확인한 뒤 다른 고객으로 확인하려 하면 already_verified로 막고, 같은 고객을 다른
# 연락처로 다시 확인하는 것은 허용하며, 틀린 확인 시도가 기존 확인 상태를 지우지 않는지 본다.
def test_one_customer_per_conversation(shop):
    ctx = make_ctx(None)
    ok(shop, ctx, "find_customer", name="김하준", contact="010-0000-0001")
    assert code(shop, ctx, "find_customer", name="이서연", contact="010-0000-0002") == "already_verified"
    assert ctx.state.verified_customer_id == "C-1"
    again = ok(shop, ctx, "find_customer", name="김하준", contact="hajun@example.com")
    assert again["customer_id"] == "C-1"
    assert code(shop, ctx, "find_customer", name="없는사람", contact="010-9999-9999") == "customer_not_found"
    assert ctx.state.verified_customer_id == "C-1"


# 본인 확인이 필요한 도구 모두가 확인 전에는 P0·P1 모두에서 identity_not_verified로 거절되고(규정 거절이
# 아님), DB가 그대로인지 본다. 본인 확인은 실험 축이 아니라 모든 조건에서 도구가 막는다(docs/experiments.md
# 공통 규칙).
@pytest.mark.parametrize("name", [n for n in HAPPY if n not in NO_IDENTITY_NEEDED])
@pytest.mark.parametrize("p1", [True, False])
def test_everything_about_a_customer_is_refused_before_verification(shop, name, p1):
    before = db.dump_db(shop)
    result = call(shop, make_ctx(None, p1=p1), name, **HAPPY[name][0])
    assert (result.ok, result.error_code, result.policy_blocked) == (False, "identity_not_verified", False)
    assert db.dump_db(shop) == before


# 본인 확인 전에는 없는 주문 번호에도 order_not_found가 아니라 identity_not_verified가 나오는지 본다. 검사
# 순서가 본인 확인 → 대상 존재 → 소유 → 무결성 → 규정이라는 설계(docs/design.md "금액 규칙" 끝)를 지킨다.
# 그래서 확인 전에는 주문이 있는지조차 답하지 않는다.
def test_identity_is_checked_before_existence(shop):
    ctx = make_ctx(None)
    assert code(shop, ctx, "get_order", order_id="O-404") == "identity_not_verified"
    assert code(shop, ctx, "create_ticket", category="other", body="문의", order_id="O-404") == (
        "identity_not_verified"
    )


# 본인 확인이 필요 없는 도구(고객 찾기, 상품 조회, 상담원 연결, think)는 확인 전에도 성공하는지 본다. 확인을
# 못 하는 고객도 상담원에게 넘길 수 있어야 한다.
@pytest.mark.parametrize("name", sorted(NO_IDENTITY_NEEDED))
def test_tools_that_need_no_verification(shop, name):
    ok(shop, make_ctx(None), name, **HAPPY[name][0])


# 존재와 소유: 항상 막는 무결성 검사(ctx.require). 규정 모드와 무관하다.
# ---------------------------------------------------------------------------------- existence and ownership

# (기대 오류 코드, 도구, 인자) 목록. 없는 고객·상품·주문·주소·줄·옵션, 다른 고객의 것(C-2, O-9, AD-C-2-1),
# 같은 주소·같은 옵션으로 변경, 재고 없음(V-1-03), 이미 취소된 주문(O-6)에 대한 처리를 도구마다 다룬다.
# docs/design.md "항상 막는 검사" 표의 코드들이다.
REQUIRE_CASES: list[tuple[str, str, dict]] = [
    ("customer_not_found", "get_customer", {"customer_id": "C-404"}),
    ("customer_not_found", "list_orders", {"customer_id": "C-404"}),
    ("not_same_customer", "get_customer", {"customer_id": "C-2"}),
    ("not_same_customer", "list_orders", {"customer_id": "C-2"}),
    ("product_not_found", "get_product", {"product_id": "P-404"}),
    ("order_not_found", "get_order", {"order_id": "O-404"}),
    ("order_not_found", "track_shipment", {"order_id": "O-404"}),
    ("order_not_found", "cancel_order", {"order_id": "O-404", "reason": "changed_mind"}),
    ("order_not_found", "request_return", {"order_id": "O-404", "line_nos": [1], "reason": "defective"}),
    (
        "order_not_found",
        "request_exchange",
        {"order_id": "O-404", "line_no": 1, "new_variant_id": "V-1-02", "reason": "defective"},
    ),
    ("order_not_found", "change_shipping_address", {"order_id": "O-404", "address_id": "AD-C-1-2"}),
    ("order_not_found", "issue_compensation_coupon", {"order_id": "O-404", "reason": "delivery_delay"}),
    ("order_not_found", "create_ticket", {"category": "other", "body": "문의", "order_id": "O-404"}),
    ("not_order_owner", "get_order", {"order_id": "O-9"}),
    ("not_order_owner", "track_shipment", {"order_id": "O-9"}),
    ("not_order_owner", "cancel_order", {"order_id": "O-9", "reason": "changed_mind"}),
    ("not_order_owner", "request_return", {"order_id": "O-9", "line_nos": [1], "reason": "defective"}),
    (
        "not_order_owner",
        "request_exchange",
        {"order_id": "O-9", "line_no": 1, "new_variant_id": "V-1-02", "reason": "defective"},
    ),
    ("not_order_owner", "change_shipping_address", {"order_id": "O-9", "address_id": "AD-C-1-2"}),
    ("not_order_owner", "issue_compensation_coupon", {"order_id": "O-9", "reason": "delivery_delay"}),
    ("not_order_owner", "create_ticket", {"category": "other", "body": "문의", "order_id": "O-9"}),
    ("address_not_found", "change_shipping_address", {"order_id": "O-1", "address_id": "AD-404"}),
    ("address_not_owned", "change_shipping_address", {"order_id": "O-1", "address_id": "AD-C-2-1"}),
    ("same_address", "change_shipping_address", {"order_id": "O-1", "address_id": "AD-C-1-1"}),
    ("line_not_found", "request_return", {"order_id": "O-3", "line_nos": [1, 7], "reason": "defective"}),
    (
        "line_not_found",
        "request_exchange",
        {"order_id": "O-3", "line_no": 7, "new_variant_id": "V-1-02", "reason": "defective"},
    ),
    (
        "variant_not_found",
        "request_exchange",
        {"order_id": "O-3", "line_no": 1, "new_variant_id": "V-404", "reason": "defective"},
    ),
    (
        "same_variant",
        "request_exchange",
        {"order_id": "O-3", "line_no": 1, "new_variant_id": "V-1-01", "reason": "defective"},
    ),
    (
        "out_of_stock",
        "request_exchange",
        {"order_id": "O-3", "line_no": 1, "new_variant_id": "V-1-03", "reason": "defective"},
    ),
    ("already_cancelled", "cancel_order", {"order_id": "O-6", "reason": "changed_mind"}),
    ("already_cancelled", "request_return", {"order_id": "O-6", "line_nos": [1], "reason": "defective"}),
    (
        "already_cancelled",
        "request_exchange",
        {"order_id": "O-6", "line_no": 1, "new_variant_id": "V-1-02", "reason": "defective"},
    ),
    ("already_cancelled", "change_shipping_address", {"order_id": "O-6", "address_id": "AD-C-1-2"}),
]


# 위 경우마다 P0·P1 모두에서 정해진 코드로 거절되고, 규정 거절로 표시되지 않으며(policy_blocked False), DB와
# 위반 기록이 그대로인지 본다. 무결성 검사를 P0에서 통과시키면 남의 주문 취소 같은 일이 '규정 위반'으로만
# 기록되어 버린다.
@pytest.mark.parametrize(("expected", "name", "args"), REQUIRE_CASES)
@pytest.mark.parametrize("p1", [True, False])
def test_always_enforced_checks_refuse_in_both_modes(shop, expected, name, args, p1):
    before = db.dump_db(shop)
    ctx = make_ctx(p1=p1)
    result = call(shop, ctx, name, **args)
    assert (result.ok, result.error_code, result.policy_blocked) == (False, expected, False)
    assert db.dump_db(shop) == before and ctx.violations == []


# 교환할 옵션의 재고가 0이 아니어도 그 줄의 수량(2개)보다 적으면(1개) out_of_stock인지 본다. 교환은 새 옵션의
# 재고를 줄의 수량만큼 줄이므로 수량과 비교해야 한다. 무결성 검사라 P0에서도 막힌다.
def test_out_of_stock_compares_stock_with_the_ordered_quantity(shop):
    with Session(shop) as session:
        session.get(db.ProductVariant, "V-2-02").stock = 1
        session.commit()
    args = {"order_id": "O-3", "line_no": 2, "new_variant_id": "V-2-02", "reason": "defective"}
    assert code(shop, make_ctx(p1=False), "request_exchange", **args) == "out_of_stock"


# 한 주문 줄은 반품이든 교환이든 한 번만 접수되는지 본다: 반품된 줄을 다시 반품·교환하거나, 교환된 줄을
# 반품하면 already_requested. 이미 접수된 줄을 포함한 여러 줄 반품도 통째로 거절된다.
@pytest.mark.parametrize("p1", [True, False])
def test_a_line_is_requested_only_once(shop, p1):
    ctx = make_ctx(p1=p1)
    ok(shop, ctx, "request_return", order_id="O-3", line_nos=[2], reason="defective")
    assert code(shop, ctx, "request_return", order_id="O-3", line_nos=[1, 2], reason="defective") == (
        "already_requested"
    )
    exchange = {"order_id": "O-3", "line_no": 2, "new_variant_id": "V-2-02", "reason": "defective"}
    assert code(shop, ctx, "request_exchange", **exchange) == "already_requested"
    ok(shop, ctx, "request_exchange", order_id="O-3", line_no=1, new_variant_id="V-1-02", reason="defective")
    assert code(shop, ctx, "request_return", order_id="O-3", line_nos=[1], reason="defective") == (
        "already_requested"
    )


# 반품 인자의 형식 오류(빈 줄 목록, 같은 줄 반복, 없는 사유 코드, 사유 누락, 지어낸 환불액 인자)는 도구에 닿기
# 전에 invalid_arguments로 막히는지 본다. 특히 환불액은 인자가 아니라 규정이 계산하므로 모델이 금액을 지어
# 넣을 수 없어야 한다.
@pytest.mark.parametrize(
    "args",
    [
        {"order_id": "O-3", "line_nos": [], "reason": "defective"},
        {"order_id": "O-3", "line_nos": [1, 1], "reason": "defective"},
        {"order_id": "O-3", "line_nos": [1], "reason": "broken"},
        {"order_id": "O-3", "line_nos": [1]},
        {"order_id": "O-3", "line_nos": [1], "reason": "defective", "refund_won": 99999},
    ],
)
def test_bad_return_arguments_fail_validation(shop, args):
    assert code(shop, make_ctx(), "request_return", **args) == "invalid_arguments"


# 빈 티켓 본문, 빈 이관 요약, 지어낸 쿠폰 금액 인자, 인자 없는 think가 invalid_arguments인지 본다. 쿠폰 금액도
# 규정이 정한다(docs/design.md "금액 규칙").
def test_other_bad_arguments_fail_validation(shop):
    ctx = make_ctx()
    assert code(shop, ctx, "create_ticket", category="other", body="") == "invalid_arguments"
    assert code(shop, ctx, "transfer_to_human", reason="customer_request", summary="") == "invalid_arguments"
    assert code(
        shop, ctx, "issue_compensation_coupon", order_id="O-4", reason="delivery_delay", amount_won=50000
    ) == ("invalid_arguments")
    assert code(shop, ctx, "think") == "invalid_arguments"


# 규정 검사(P축): P1이면 막고, P0이면 위반으로 기록한 채 통과시킨다.
# -------------------------------------------------------------------------------------------- policy checks


# O-4에 이미 보상 쿠폰이 하나 있는 상태를 만든다(주문당 1장 규정 시험용).
def _coupon_for_the_order(session: Session) -> None:
    add_compensation_coupon(session, "CP-O-4-1", "C-1", "O-4", kst(9, 10))


# C-1에게 최근 30일 안에 보상 쿠폰 두 장이 있는 상태를 만든다(고객당 30일 2장 한도 시험용).
def _two_recent_coupons(session: Session) -> None:
    add_compensation_coupon(session, "CP-O-1-1", "C-1", "O-1", kst(8, 20))
    add_compensation_coupon(session, "CP-O-2-1", "C-1", "O-2", kst(9, 10))


# EXCHANGE: 교환 인자의 공통 부분. POLICY_CASES: (기대 규정 코드, 도구, 인자, 준비 함수) 목록. 상태가 맞지
# 않는 취소·반품·교환·배송지 변경, 기한이 지난 반품·교환(O-4), 다른 상품으로의 교환, 자격 없는 쿠폰(늦지 않음,
# 불량 접수 없음, 취소된 주문), 주문당 1장, 고객당 한도를 다룬다.
EXCHANGE = {"line_no": 1, "new_variant_id": "V-1-02", "reason": "changed_mind"}
POLICY_CASES: list[tuple[str, str, dict, object]] = [
    ("cancel_not_allowed_status", "cancel_order", {"order_id": "O-2", "reason": "changed_mind"}, None),
    ("cancel_not_allowed_status", "cancel_order", {"order_id": "O-4", "reason": "changed_mind"}, None),
    (
        "return_not_delivered",
        "request_return",
        {"order_id": "O-2", "line_nos": [1], "reason": "changed_mind"},
        None,
    ),
    (
        "return_window_expired",
        "request_return",
        {"order_id": "O-4", "line_nos": [1], "reason": "defective"},
        None,
    ),
    ("exchange_not_delivered", "request_exchange", {"order_id": "O-5", **EXCHANGE}, None),
    ("exchange_window_expired", "request_exchange", {"order_id": "O-4", **EXCHANGE}, None),
    (
        "exchange_different_product",
        "request_exchange",
        {"order_id": "O-3", "line_no": 1, "new_variant_id": "V-2-02", "reason": "changed_mind"},
        None,
    ),
    (
        "address_change_not_allowed_status",
        "change_shipping_address",
        {"order_id": "O-2", "address_id": "AD-C-1-2"},
        None,
    ),
    (
        "coupon_not_eligible",
        "issue_compensation_coupon",
        {"order_id": "O-5", "reason": "delivery_delay"},
        None,
    ),
    (
        "coupon_not_eligible",
        "issue_compensation_coupon",
        {"order_id": "O-3", "reason": "delivery_delay"},
        None,
    ),
    (
        "coupon_order_cancelled",
        "issue_compensation_coupon",
        {"order_id": "O-6", "reason": "delivery_delay"},
        None,
    ),
    (
        "coupon_not_eligible",
        "issue_compensation_coupon",
        {"order_id": "O-3", "reason": "defective_item"},
        None,
    ),
    (
        "coupon_already_issued_for_order",
        "issue_compensation_coupon",
        {"order_id": "O-4", "reason": "delivery_delay"},
        _coupon_for_the_order,
    ),
    (
        "coupon_limit_exceeded",
        "issue_compensation_coupon",
        {"order_id": "O-4", "reason": "delivery_delay"},
        _two_recent_coupons,
    ),
]


# 같은 호출을 P1과 P0로 한 번씩 부른다. P1: 정해진 코드, policy_blocked True, DB·위반 기록 그대로. P0:
# 성공하고 결과와 문맥에 그 위반 코드가 하나씩 기록되며 DB가 바뀐다. 2단계 후보 P1(규정을 도구에서도 막음)과
# 기준 P0의 차이가 바로 이 동작이다.
@pytest.mark.parametrize(("expected", "name", "args", "setup"), POLICY_CASES)
def test_p1_blocks_a_policy_violation_and_p0_lets_it_through(shop, expected, name, args, setup):
    if setup is not None:
        with Session(shop) as session:
            setup(session)
            session.commit()
    before = db.dump_db(shop)

    strict = make_ctx(p1=True)
    result = call(shop, strict, name, **args)
    assert (result.ok, result.error_code, result.policy_blocked) == (False, expected, True)
    assert db.dump_db(shop) == before and strict.violations == []

    loose = make_ctx(p1=False)
    result = call(shop, loose, name, **args)
    assert result.ok, result.content
    assert result.violations == (expected,) and loose.violations == [expected]
    assert db.dump_db(shop) != before


# POLICY_CASES가 docs/design.md "규정 검사" 표의 코드 11개를 모두 다루는지 본다. 규정을 새로 넣고 테스트를
# 잊으면 여기서 걸린다.
def test_every_policy_code_of_the_design_is_covered():
    assert {case[0] for case in POLICY_CASES} == {
        "cancel_not_allowed_status",
        "return_not_delivered",
        "return_window_expired",
        "exchange_not_delivered",
        "exchange_window_expired",
        "exchange_different_product",
        "address_change_not_allowed_status",
        "coupon_not_eligible",
        "coupon_order_cancelled",
        "coupon_already_issued_for_order",
        "coupon_limit_exceeded",
    }


# 9월 2일 14:20(KST) 수령 주문의 반품 기한이 9월 9일 23:59까지이고 9월 10일 00:00에는 닫히는지 본다. '수령일
# 다음 날부터 7일째 되는 날까지, KST 날짜 기준' 규정(docs/design.md)을 시각이 아니라 날짜로 계산하는지, UTC로
# 계산해 경계가 9시간 밀리지 않는지 확인한다. 접수 번호는 가장 작은 줄 번호로 RT-O-4-1이다.
def test_the_return_window_closes_at_kst_midnight(shop):
    args = {"order_id": "O-4", "line_nos": [1], "reason": "defective"}  # delivered 09-02 14:20 KST
    assert code(shop, make_ctx(now=kst(9, 10, 0, 0)), "request_return", **args) == "return_window_expired"
    out = ok(shop, make_ctx(now=kst(9, 9, 23, 59)), "request_return", **args)
    assert out["request_id"] == "RT-O-4-1"


# 호출 하나가 규정 두 개(배송 전 교환, 다른 상품으로 교환)를 어기면 P0에서 둘 다 순서대로 기록되는지 본다. 첫
# 위반에서 멈추면 '통과된 규정 위반' 수가 덜 세어진다.
def test_p0_records_every_rule_a_call_breaks(shop):
    ctx = make_ctx(p1=False)
    args = {"order_id": "O-5", "line_no": 1, "new_variant_id": "V-2-01", "reason": "changed_mind"}
    result = call(shop, ctx, "request_exchange", **args)
    assert result.ok and result.violations == ("exchange_not_delivered", "exchange_different_product")


# 효과: 쓰기 도구가 DB에 남기는 것과 돌려주는 값.
# -------------------------------------------------------------------------------------------------- effects


# 취소가 결제 금액 전액을 환불액으로 돌려주고, 주문과 모든 줄을 cancelled로, 결제를 refund_pending으로 바꾸며,
# 취소 시각을 ctx.now(UTC 저장)로, 사유를 그대로 남기는지 본다(docs/design.md "금액 규칙").
def test_cancel_order_refunds_the_whole_payment(shop):
    out = ok(shop, make_ctx(), "cancel_order", order_id="O-5", reason="ordered_by_mistake")
    assert out == {
        "order_id": "O-5",
        "status": "cancelled",
        "status_label": "취소됨",
        "refund_won": 38900,
        "refund_method": "card",
        "refund_method_label": "신용카드",
    }
    order = rows(shop, "orders")["O-5"]
    assert (order["status"], order["cancel_reason"]) == ("cancelled", "ordered_by_mistake")
    assert order["cancelled_at"] == "2026-09-14T01:00:00+00:00"
    assert rows(shop, "order_items")["O-5/1"]["status"] == "cancelled"
    payment = rows(shop, "payments")["O-5"]
    assert (payment["status"], payment["refund_won"], payment["amount_won"]) == (
        "refund_pending",
        38900,
        38900,
    )


# 여러 줄 반품: 줄 번호 순서와 무관하게 정렬된 인자로 기록되고, 환불액 = Σ(단가 × 수량) − 반품 배송비(단순
# 변심 3,000원) = 65,300원인지 본다. 접수 행·접수 줄 행·주문 줄 상태가 바뀌지만 결제와 주문 상태는
# 그대로다(회수 뒤 환불).
def test_request_return_computes_refund_and_fee(shop):
    before = rows(shop, "payments")
    result = call(shop, make_ctx(), "request_return", order_id="O-3", line_nos=[2, 1], reason="changed_mind")
    assert result.args["line_nos"] == [1, 2]
    assert json.loads(result.content) == {
        "request_id": "RT-O-3-1",
        "order_id": "O-3",
        "line_nos": [1, 2],
        "refund_won": 65300,
        "return_fee_won": 3000,
    }
    request = rows(shop, "service_requests")["RT-O-3-1"]
    assert (request["kind"], request["reason"], request["refund_won"]) == ("return", "changed_mind", 65300)
    assert request["created_at"] == "2026-09-14T01:00:00+00:00"
    items = rows(shop, "service_request_items")
    assert [(k, r["quantity"], r["exchange_variant_id"]) for k, r in items.items()] == [
        ("RT-O-3-1/1", 1, None),
        ("RT-O-3-1/2", 2, None),
    ]
    statuses = {k: r["status"] for k, r in rows(shop, "order_items").items() if k.startswith("O-3/")}
    assert statuses == {"O-3/1": "return_requested", "O-3/2": "return_requested"}
    assert rows(shop, "payments") == before  # the refund happens after the goods come back
    assert rows(shop, "orders")["O-3"]["status"] == "delivered"


# 줄 하나만 반품하면 접수 번호가 그 줄 번호(RT-O-3-2)로 만들어지고, 오배송(wrong_item)은 반품 배송비가
# 0원이며, 다른 줄은 건드리지 않는지 본다. 개수가 아니라 내용으로 id를 만들어 호출 순서가 결과를 바꾸지 않게
# 한다(docs/design.md "결정성").
def test_a_partial_return_is_named_after_its_lowest_line(shop):
    out = ok(shop, make_ctx(), "request_return", order_id="O-3", line_nos=[2], reason="wrong_item")
    assert out == {
        "request_id": "RT-O-3-2",
        "order_id": "O-3",
        "line_nos": [2],
        "refund_won": 29400,
        "return_fee_won": 0,
    }
    assert rows(shop, "order_items")["O-3/1"]["status"] == "ordered"


# 교환이 EX-주문-줄 번호로 접수되고, 환불액·배송비 0원, 새 옵션의 재고가 줄의 수량(2)만큼 줄며(10 → 8), 원래
# 옵션의 재고는 그대로인지 본다.
def test_request_exchange_reserves_the_new_variant(shop):
    args = {"order_id": "O-3", "line_no": 2, "new_variant_id": "V-2-02", "reason": "changed_mind"}
    out = ok(shop, make_ctx(), "request_exchange", **args)
    assert out == {
        "request_id": "EX-O-3-2",
        "order_id": "O-3",
        "line_no": 2,
        "new_variant_id": "V-2-02",
        "new_option_label": "700ml",
    }
    request = rows(shop, "service_requests")["EX-O-3-2"]
    assert (request["kind"], request["refund_won"], request["return_fee_won"]) == ("exchange", 0, 0)
    item = rows(shop, "service_request_items")["EX-O-3-2/2"]
    assert (item["quantity"], item["exchange_variant_id"]) == (2, "V-2-02")
    assert rows(shop, "product_variants")["V-2-02"]["stock"] == 8
    assert rows(shop, "product_variants")["V-2-01"]["stock"] == 10
    assert rows(shop, "order_items")["O-3/2"]["status"] == "exchange_requested"


# 배송지 변경이 주문에 저장된 배송지 사본(받는 사람, 우편번호, 주소)을 새 주소로 바꾸고, get_order에도 바로
# 보이는지 본다. 주문은 주소 id만이 아니라 주문 시점의 사본을 갖는 구조다.
def test_change_shipping_address_updates_the_snapshot(shop):
    out = ok(shop, make_ctx(), "change_shipping_address", order_id="O-5", address_id="AD-C-1-2")
    assert out == {
        "order_id": "O-5",
        "address_id": "AD-C-1-2",
        "recipient": "김하준",
        "postal_code": "04524",
        "address": "서울특별시 중구 가상로 22",
    }
    order = rows(shop, "orders")["O-5"]
    assert (order["ship_address_id"], order["ship_address"]) == ("AD-C-1-2", "서울특별시 중구 가상로 22")
    assert ok(shop, make_ctx(), "get_order", order_id="O-5")["shipping"]["address_id"] == "AD-C-1-2"


# 보상 쿠폰 금액을 규정이 정하는지 본다: 3일 늦은 O-4는 5,000원, 1일 늦은 O-2는 2,000원, 유효 기간은 발급 시각
# + 30일. 저장된 행의 종류·사유·주문·사용 여부와 UTC 시각까지 확인한다.
def test_coupon_amounts_come_from_the_rules(shop):
    ctx = make_ctx()
    late3 = ok(shop, ctx, "issue_compensation_coupon", order_id="O-4", reason="delivery_delay")
    assert late3 == {"coupon_id": "CP-O-4-1", "amount_won": 5000, "expires_at": "2026-10-14 10:00"}
    late1 = ok(shop, ctx, "issue_compensation_coupon", order_id="O-2", reason="delivery_delay")
    assert late1["amount_won"] == 2000
    coupon = rows(shop, "coupons")["CP-O-4-1"]
    assert (coupon["customer_id"], coupon["kind"], coupon["reason"]) == (
        "C-1",
        "compensation",
        "delivery_delay",
    )
    assert (coupon["order_id"], coupon["amount_won"], coupon["used_at"]) == ("O-4", 5000, None)
    assert (coupon["issued_at"], coupon["expires_at"]) == (
        "2026-09-14T01:00:00+00:00",
        "2026-10-14T01:00:00+00:00",
    )


# 불량 사유 쿠폰은 그 주문에 사유가 defective인 반품·교환 접수가 생긴 뒤에만 자격이 생기고, 금액은 3,000원인지
# 본다.
def test_a_defective_request_makes_the_order_eligible(shop):
    ctx = make_ctx()
    assert code(shop, ctx, "issue_compensation_coupon", order_id="O-3", reason="defective_item") == (
        "coupon_not_eligible"
    )
    ok(shop, ctx, "request_exchange", order_id="O-3", line_no=1, new_variant_id="V-1-02", reason="defective")
    out = ok(shop, ctx, "issue_compensation_coupon", order_id="O-3", reason="defective_item")
    assert out["coupon_id"] == "CP-O-3-1" and out["amount_won"] == 3000


# 단순 변심 반품 접수로는 불량 쿠폰 자격이 생기지 않는지 본다.
def test_a_changed_mind_request_does_not_make_the_order_eligible(shop):
    ctx = make_ctx()
    ok(shop, ctx, "request_return", order_id="O-3", line_nos=[1], reason="changed_mind")
    assert code(shop, ctx, "issue_compensation_coupon", order_id="O-3", reason="defective_item") == (
        "coupon_not_eligible"
    )


# P0에서 자격 없이 통과된 쿠폰은 사유별 최소 금액(지연 2,000원, 불량 3,000원)으로 발급되고, 같은 주문의 두
# 번째 쿠폰은 다음 번호(CP-O-5-2)를 받으며, 위반이 차례로 기록되는지 본다. P0에서도 금액이 정해져야 DB 비교가
# 결정적이다(docs/design.md "금액 규칙").
def test_p0_coupons_that_are_not_due_get_the_lowest_amount_and_the_next_number(shop):
    ctx = make_ctx(p1=False)
    first = ok(shop, ctx, "issue_compensation_coupon", order_id="O-5", reason="delivery_delay")
    assert first == {"coupon_id": "CP-O-5-1", "amount_won": 2000, "expires_at": "2026-10-14 10:00"}
    second = ok(shop, ctx, "issue_compensation_coupon", order_id="O-5", reason="defective_item")
    assert (second["coupon_id"], second["amount_won"]) == ("CP-O-5-2", 3000)
    assert ctx.violations == ["coupon_not_eligible", "coupon_not_eligible", "coupon_already_issued_for_order"]


# 고객당 30일 2장 한도가 정확히 30일 전 쿠폰은 빼고 29일 전 것은 세며, 다른 고객의 쿠폰과 판촉 쿠폰은 세지
# 않는지 본다. 29일 전 1장 + 새로 발급한 1장으로 2장이 찬 뒤, 지연이 생긴 다른 주문의 세 번째 발급이
# coupon_limit_exceeded로 막혀야 한다.
def test_the_coupon_limit_looks_back_30_days_and_ignores_promo_coupons(shop):
    with Session(shop) as session:
        add_compensation_coupon(session, "CP-O-1-1", "C-1", "O-1", NOW - timedelta(days=30))  # just outside
        add_compensation_coupon(session, "CP-O-2-1", "C-1", "O-2", NOW - timedelta(days=29))
        add_compensation_coupon(session, "CP-O-9-1", "C-2", "O-9", NOW - timedelta(days=1))  # someone else
        session.add(
            db.Coupon(
                id="CP-C-1-P1",
                customer_id="C-1",
                kind=db.CouponKind.PROMO,
                amount_won=1000,
                reason=None,
                order_id=None,
                issued_at=NOW - timedelta(days=2),
                expires_at=NOW + timedelta(days=28),
                used_at=None,
            )
        )
        session.commit()
    ctx = make_ctx()
    ok(shop, ctx, "issue_compensation_coupon", order_id="O-4", reason="delivery_delay")
    with Session(shop) as session:
        session.get(db.Shipment, "O-5").promised_by = kst(9, 10).date()  # now O-5 is late and would be due
        session.commit()
    third = {"order_id": "O-5", "reason": "delivery_delay"}
    assert code(shop, ctx, "issue_compensation_coupon", **third) == "coupon_limit_exceeded"


# 티켓이 고객별 순번(TK-C-1-1, TK-C-1-2, TK-C-2-1)으로 매겨지고, 주문 번호를 주지 않으면 NULL로, 본문과 생성
# 시각이 그대로 저장되는지 본다.
def test_tickets_are_numbered_per_customer(shop):
    ctx = make_ctx()
    first = ok(shop, ctx, "create_ticket", category="refund", body="환불 입금 확인 요청")
    assert first == {"ticket_id": "TK-C-1-1", "category": "refund", "category_label": "환불"}
    second = ok(shop, ctx, "create_ticket", category="delivery", body="배송 문의", order_id="O-2")
    assert second["ticket_id"] == "TK-C-1-2"
    tickets = rows(shop, "tickets")
    assert (tickets["TK-C-1-1"]["order_id"], tickets["TK-C-1-2"]["order_id"]) == (None, "O-2")
    assert tickets["TK-C-1-1"]["created_at"] == "2026-09-14T01:00:00+00:00"
    assert tickets["TK-C-1-1"]["body"] == "환불 입금 확인 요청"
    other = ok(shop, make_ctx("C-2"), "create_ticket", category="other", body="문의")
    assert other["ticket_id"] == "TK-C-2-1"


# 개수로 만드는 id가 이미 있는 번호(TK-C-1-2)를 건너뛰는지 본다. 개수 + 1만 쓰면 기본 키가 겹쳐
# IntegrityError(도구 버그)가 난다.
def test_count_based_ids_skip_ids_that_are_taken(shop):
    with Session(shop) as session:
        session.add(
            db.Ticket(
                id="TK-C-1-2",
                customer_id="C-1",
                order_id=None,
                category=db.TicketCategory.OTHER,
                body="기존 티켓",
                created_at=kst(9, 1),
            )
        )
        session.commit()
    assert ok(shop, make_ctx(), "create_ticket", category="other", body="문의")["ticket_id"] == "TK-C-1-3"


# 상담원 연결이 이관 행(HO-1, HO-2 순번)을 남기고 고정 안내문(HANDOFF_MESSAGE)을 돌려주는지, 본인 확인 전이면
# 고객 번호가 비고 후면 채워지는지 본다. 이관의 사유·고객 번호·요약은 판정 비교에서 빠지지만 행에는 남는다.
def test_transfer_to_human_writes_a_handoff_row(shop):
    out = ok(shop, make_ctx(None), "transfer_to_human", reason="cannot_verify", summary="본인 확인 실패")
    assert out == {"handoff_id": "HO-1", "message": HANDOFF_MESSAGE}
    out = ok(shop, make_ctx("C-1"), "transfer_to_human", reason="customer_request", summary="상담원 요청")
    assert out["handoff_id"] == "HO-2"
    handoffs = rows(shop, "handoffs")
    assert (handoffs["HO-1"]["customer_id"], handoffs["HO-2"]["customer_id"]) == (None, "C-1")
    assert handoffs["HO-1"]["reason"] == "cannot_verify"
    assert handoffs["HO-2"]["created_at"] == "2026-09-14T01:00:00+00:00"


# 반복 호출과 결정성: 같은 쓰기를 두 번 해도 버그가 아니라 정해진 답이 나오고, 같은 호출 묶음은 같은 상태를
# 만든다.
# ------------------------------------------------------------------------------ repeated calls, determinism

# 같은 인자로 두 번째 부르면 무결성 검사로 거절되어야 하는 쓰기 도구와 그 코드.
TWICE_REFUSED = {
    "cancel_order": "already_cancelled",
    "request_return": "already_requested",
    "request_exchange": "already_requested",
    "change_shipping_address": "same_address",
}


# 쓰기 도구마다 같은 호출을 두 번 해서, 두 번째가 예외(IntegrityError → ToolBugError) 없이 답으로 오는지 본다.
# 취소·반품·교환·배송지는 정해진 코드로 거절되고 DB가 그대로, 쿠폰은 P1이면 주문당 1장으로 거절·P0이면 위반
# 기록 후 CP-O-4-2 발급, 티켓과 이관은 순번이 있어 새 행이 생긴다. 두 번째 호출이 도구 버그로 처리되면
# 에이전트의 행동이 아니라 우리 코드 때문에 에피소드가 infra_error로 빠진다.
@pytest.mark.parametrize("name", sorted(WRITE_TOOLS))
@pytest.mark.parametrize("p1", [True, False])
def test_a_repeated_write_is_answered_not_crashed(shop, name, p1):
    ctx = make_ctx(p1=p1)
    args = HAPPY[name][0]
    ok(shop, ctx, name, **args)
    after_first = db.dump_db(shop)
    second = call(shop, ctx, name, **args)  # a ToolBugError (IntegrityError) would raise here
    if name in TWICE_REFUSED:
        assert (second.ok, second.error_code) == (False, TWICE_REFUSED[name])
        assert db.dump_db(shop) == after_first
    elif name == "issue_compensation_coupon" and p1:
        assert (second.ok, second.error_code) == (False, "coupon_already_issued_for_order")
        assert db.dump_db(shop) == after_first
    elif name == "issue_compensation_coupon":
        assert second.ok and second.violations == ("coupon_already_issued_for_order",)
        assert json.loads(second.content)["coupon_id"] == "CP-O-4-2"
    else:  # tickets and hand-offs are numbered, so a second one is a new row
        assert second.ok and name in {"create_ticket", "transfer_to_human"}
        assert len(rows(shop, "tickets")) + len(rows(shop, "handoffs")) == 2


# 결정성 시험용 호출 묶음: 본인 확인부터 반품, 교환, 쿠폰, 취소(하나는 P1에서 거절됨), 배송지 변경, 티켓,
# 이관까지 모든 종류의 쓰기를 한 번씩 한다. 티켓 본문과 이관 요약은 비교에서 빠지는 자유 입력이다.
SEQUENCE: list[tuple[str, dict]] = [
    ("find_customer", {"name": "김하준", "contact": "010-0000-0001"}),
    ("get_order", {"order_id": "O-3"}),
    ("request_return", {"order_id": "O-3", "line_nos": [2], "reason": "defective"}),
    (
        "request_exchange",
        {"order_id": "O-3", "line_no": 1, "new_variant_id": "V-1-02", "reason": "changed_mind"},
    ),
    ("issue_compensation_coupon", {"order_id": "O-3", "reason": "defective_item"}),
    ("cancel_order", {"order_id": "O-1", "reason": "delivery_too_slow"}),
    ("cancel_order", {"order_id": "O-2", "reason": "delivery_too_slow"}),  # refused under P1
    ("change_shipping_address", {"order_id": "O-5", "address_id": "AD-C-1-2"}),
    ("create_ticket", {"category": "product", "body": "본문은 비교에서 빠진다", "order_id": "O-3"}),
    ("transfer_to_human", {"reason": "customer_request", "summary": "요약도 비교에서 빠진다"}),
]


# source DB의 새 복사본에서 확인 전 대화로 호출 묶음을 차례로 실행하고, 각 출력 글과 최종 상태 해시를
# 돌려준다.
def play(source, *, now=NOW, sequence=SEQUENCE) -> tuple[list[str], str]:
    engine = db.memory_engine(source)
    ctx = make_ctx(None, now=now)
    outputs = [execute(REGISTRY, engine, ctx, name, args).content for name, args in sequence]
    digest = db.state_hash(db.dump_db(engine))
    engine.dispose()
    return outputs, digest


# 같은 묶음을 두 복사본에서 돌리면 출력 글과 상태 해시가 모두 같고 원본과는 다르며, 새로 생긴 id가 내용
# 기반(RT-O-3-2, EX-O-3-1)과 순번 기반(CP-O-3-1, TK-C-1-1, HO-1)으로 예상대로 나오는지 본다. 판정이 '정답만
# 실행한 DB'와의 비교이므로 이 결정성이 판정의 전제다.
def test_the_same_calls_on_two_copies_give_the_same_state(shop):
    first, second = play(shop), play(shop)
    assert first == second
    assert first[1] != db.state_hash(db.dump_db(shop))
    created = [json.loads(text) for text in first[0] if not text.startswith("Error")]
    ids = [
        out.get("request_id") or out.get("coupon_id") or out.get("ticket_id") or out.get("handoff_id")
        for out in created
    ]
    assert [i for i in ids if i] == ["RT-O-3-2", "EX-O-3-1", "CP-O-3-1", "TK-C-1-1", "HO-1"]


# 반품과 교환의 순서를 바꿔도 최종 상태가 같은지 본다. 에이전트가 정답과 다른 유효한 순서로 처리해도 성공으로
# 판정되게 하려고 반품·교환 id를 개수가 아니라 내용으로 만든다(docs/design.md "결정성").
def test_content_ids_do_not_depend_on_the_order_of_calls(shop):
    swapped = [SEQUENCE[0], SEQUENCE[3], SEQUENCE[2], *SEQUENCE[4:]]
    assert play(shop, sequence=swapped)[1] == play(shop, sequence=[SEQUENCE[0], *SEQUENCE[2:]])[1]


# 티켓 본문을 바꿔도 상태 해시가 같고(db.IGNORED_COLUMNS로 비교에서 뺌), now를 1분 바꾸면 해시가
# 달라지는지(시각은 ctx.now에서만 옴) 본다. 자유 입력은 모델마다 달라 비교하면 정답도 실패하고, 시각을 현재
# 시각에서 가져오면 실행마다 결과가 달라진다.
def test_free_text_is_left_out_of_the_comparison_and_time_comes_from_ctx_now(shop):
    reworded = [(n, {**a, "body": "다른 본문"} if n == "create_ticket" else a) for n, a in SEQUENCE]
    assert play(shop, sequence=reworded)[1] == play(shop)[1]
    assert play(shop, now=NOW + timedelta(minutes=1))[1] != play(shop)[1]


# tools.py와 rules.py를 ast로 읽어 uuid·random·secrets·time 모듈 import와 now()·utcnow()·today() 호출이 없는지
# 본다. 도구·판정 코드는 현재 시각·uuid·난수를 쓰지 않는다는 프로젝트 규칙(docs/design.md "결정성")을 코드로
# 지킨다.
@pytest.mark.parametrize("filename", ["tools.py", "rules.py"])
def test_no_wall_clock_uuid_or_random(filename):
    tree = ast.parse((PACKAGE_DIR / filename).read_text(encoding="utf-8"))
    banned_modules = {"uuid", "random", "secrets", "time"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not {alias.name.split(".")[0] for alias in node.names} & banned_modules
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned_modules
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"now", "utcnow", "today"}, f"{filename}:{node.lineno}"


# 규정 문서: 에이전트가 읽는 prompts/policy.md와 코드의 규정 숫자가 같아야 한다.
# ------------------------------------------------------------------------------------------ policy document


# policy.md에 규정의 숫자(7일, 3,000원, 2,000원, 5,000원, 30일, 2장)가 있고, rules.py의 상수로 만든 문구가
# 문서에 그대로 있는지 본다. 규정 문서에 없는 규칙을 코드가 막으면 에이전트가 알 수 없는 실패가
# 되므로, 상수를 바꾸면 문서도 같이 바꾸게 강제한다.
# 문서가 3,500자 미만인지도 본다. 도구 스키마, 대화와 함께 16k 컨텍스트를 나눠 쓰기 때문이다(마지막 줄 주석).
def test_the_policy_document_states_the_numbers_of_the_rules():
    text = (PACKAGE_DIR / "prompts" / "policy.md").read_text(encoding="utf-8")
    for literal in ["7일", "3,000원", "2,000원", "5,000원", "30일", "2장"]:
        assert literal in text, literal
    for value in [
        f"{rules.RETURN_WINDOW_DAYS}일째",
        f"단순 변심은 {rules.RETURN_FEE_WON:,}원",
        f"1~2일 {rules.COUPON_DELAY_SHORT_WON:,}원",
        f"{rules.COUPON_DELAY_LONG_FROM_DAYS}일 이상 {rules.COUPON_DELAY_LONG_WON:,}원",
        f"| {rules.COUPON_DEFECTIVE_WON:,}원 |",
        f"최근 {rules.COUPON_WINDOW_DAYS}일(발급일 기준, 오늘 포함) 동안 {rules.COUPON_WINDOW_LIMIT}장까지",
        f"발급일부터 {rules.COUPON_VALID_DAYS}일 동안",
        "주문당 1장",
    ]:
        assert value in text, value
    assert len(text) < 3500  # it shares a 16k context with the tool schemas and the conversation
