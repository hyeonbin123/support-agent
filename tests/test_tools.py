"""The 14 domain tools, always through toolkit.execute(). The shop rows are described in factories.py."""

import ast
import json
from datetime import timedelta
from pathlib import Path

import pytest
from conftest import NOW
from factories import add_compensation_coupon, extend_tiny_rows, kst
from sqlalchemy.orm import Session

import support_agent
from support_agent import db, rules
from support_agent.config import HANDOFF_MESSAGE
from support_agent.labels import choices
from support_agent.toolkit import ConversationState, ToolContext, ToolResult, execute
from support_agent.tools import build_registry

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


@pytest.fixture
def shop(tiny_engine):
    engine = db.memory_engine(tiny_engine)
    with Session(engine) as session:
        extend_tiny_rows(session)
        session.commit()
    yield engine
    engine.dispose()


def make_ctx(customer_id: str | None = "C-1", *, p1: bool = True, now=NOW) -> ToolContext:
    return ToolContext(now=now, enforce_policy=p1, state=ConversationState(verified_customer_id=customer_id))


def call(engine, ctx: ToolContext, name: str, /, **args) -> ToolResult:
    return execute(REGISTRY, engine, ctx, name, args)


def ok(engine, ctx: ToolContext, name: str, /, **args) -> dict:
    result = call(engine, ctx, name, **args)
    assert result.ok, result.content
    return json.loads(result.content)


def code(engine, ctx: ToolContext, name: str, /, **args) -> str | None:
    result = call(engine, ctx, name, **args)
    assert not result.ok, result.content
    assert result.content.startswith(f"Error: [{result.error_code}] ")
    return result.error_code


def rows(engine, table: str) -> dict[str, dict]:
    """Rows of one table keyed by their primary key values joined with '/'."""
    pk = [c.name for c in db.Base.metadata.tables[table].primary_key.columns]
    return {"/".join(str(r[k]) for k in pk): r for r in db.dump_db(engine, ignore=frozenset())[table]}


# ------------------------------------------------------------------------------------------------ registry


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


def test_the_schemas_are_small_enough_for_a_16k_context():
    text = json.dumps([spec.schema() for spec in REGISTRY.values()], ensure_ascii=False)
    print(f"\ntool schemas: {len(text)} characters")
    assert len(text) < 9000


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
    assert props["request_return"]["properties"]["line_nos"]["items"] == {"type": "integer"}


# ------------------------------------------------------------------------------ happy paths and output keys

HAPPY: dict[str, tuple[dict, list[str]]] = {
    "find_customer": (
        {"name": "김하준", "contact": "010-0000-0001"},
        ["customer_id", "name", "grade", "grade_label"],
    ),
    "get_customer": (
        {"customer_id": "C-1"},
        ["customer_id", "name", "phone", "email", "grade", "grade_label", "joined_at", "addresses"],
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


def test_the_happy_table_covers_every_tool():
    assert set(HAPPY) == set(REGISTRY)


@pytest.mark.parametrize("name", list(HAPPY))
def test_happy_path_output_keys_match_the_design_table(shop, name):
    args, keys = HAPPY[name]
    before = db.dump_db(shop)
    out = ok(shop, make_ctx(), name, **args)
    assert list(out) == keys
    assert (db.dump_db(shop) != before) == REGISTRY[name].write


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


# ------------------------------------------------------------------------------------------- read contents


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


def test_read_tools_give_facts_not_judgements(shop):
    ctx = make_ctx()
    text = call(shop, ctx, "get_order", order_id="O-4").content
    text += call(shop, ctx, "track_shipment", order_id="O-4").content
    assert "2026-09-09" not in text and "deadline" not in text and "returnable" not in text


def test_track_shipment_formats_times_in_kst(shop):
    out = ok(shop, make_ctx(), "track_shipment", order_id="O-4")
    assert out["delivered_at"] == "2026-09-02 14:20" and out["promised_by"] == "2026-08-30"
    assert out["shipped_at"] == "2026-08-29 09:00" and out["status_label"] == "배송완료"
    pending = ok(shop, make_ctx(), "track_shipment", order_id="O-1")
    assert (pending["shipped_at"], pending["delivered_at"]) == (None, None)


def test_get_product_lists_variants_by_id(shop):
    out = ok(shop, make_ctx(None), "get_product", product_id="P-1")
    assert [(v["variant_id"], v["stock"]) for v in out["variants"]] == [
        ("V-1-01", 5),
        ("V-1-02", 3),
        ("V-1-03", 0),
    ]


# ------------------------------------------------------------------------------------------------ identity


@pytest.mark.parametrize(
    "contact", ["010-0000-0001", "01000000001", "010 0000 0001", "hajun@example.com", " Hajun@Example.COM "]
)
def test_find_customer_normalises_the_contact_and_verifies(shop, contact):
    ctx = make_ctx(None)
    out = ok(shop, ctx, "find_customer", name="김 하준", contact=contact)
    assert out == {"customer_id": "C-1", "name": "김하준", "grade": "normal", "grade_label": "일반"}
    assert ctx.state.verified_customer_id == "C-1"


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


def test_one_customer_per_conversation(shop):
    ctx = make_ctx(None)
    ok(shop, ctx, "find_customer", name="김하준", contact="010-0000-0001")
    assert code(shop, ctx, "find_customer", name="이서연", contact="010-0000-0002") == "already_verified"
    assert ctx.state.verified_customer_id == "C-1"
    again = ok(shop, ctx, "find_customer", name="김하준", contact="hajun@example.com")
    assert again["customer_id"] == "C-1"
    assert code(shop, ctx, "find_customer", name="없는사람", contact="010-9999-9999") == "customer_not_found"
    assert ctx.state.verified_customer_id == "C-1"


@pytest.mark.parametrize("name", [n for n in HAPPY if n not in NO_IDENTITY_NEEDED])
@pytest.mark.parametrize("p1", [True, False])
def test_everything_about_a_customer_is_refused_before_verification(shop, name, p1):
    before = db.dump_db(shop)
    result = call(shop, make_ctx(None, p1=p1), name, **HAPPY[name][0])
    assert (result.ok, result.error_code, result.policy_blocked) == (False, "identity_not_verified", False)
    assert db.dump_db(shop) == before


def test_identity_is_checked_before_existence(shop):
    ctx = make_ctx(None)
    assert code(shop, ctx, "get_order", order_id="O-404") == "identity_not_verified"
    assert code(shop, ctx, "create_ticket", category="other", body="문의", order_id="O-404") == (
        "identity_not_verified"
    )


@pytest.mark.parametrize("name", sorted(NO_IDENTITY_NEEDED))
def test_tools_that_need_no_verification(shop, name):
    ok(shop, make_ctx(None), name, **HAPPY[name][0])


# ---------------------------------------------------------------------------------- existence and ownership

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


@pytest.mark.parametrize(("expected", "name", "args"), REQUIRE_CASES)
@pytest.mark.parametrize("p1", [True, False])
def test_always_enforced_checks_refuse_in_both_modes(shop, expected, name, args, p1):
    before = db.dump_db(shop)
    ctx = make_ctx(p1=p1)
    result = call(shop, ctx, name, **args)
    assert (result.ok, result.error_code, result.policy_blocked) == (False, expected, False)
    assert db.dump_db(shop) == before and ctx.violations == []


def test_out_of_stock_compares_stock_with_the_ordered_quantity(shop):
    with Session(shop) as session:
        session.get(db.ProductVariant, "V-2-02").stock = 1
        session.commit()
    args = {"order_id": "O-3", "line_no": 2, "new_variant_id": "V-2-02", "reason": "defective"}
    assert code(shop, make_ctx(p1=False), "request_exchange", **args) == "out_of_stock"


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


def test_other_bad_arguments_fail_validation(shop):
    ctx = make_ctx()
    assert code(shop, ctx, "create_ticket", category="other", body="") == "invalid_arguments"
    assert code(shop, ctx, "transfer_to_human", reason="customer_request", summary="") == "invalid_arguments"
    assert code(
        shop, ctx, "issue_compensation_coupon", order_id="O-4", reason="delivery_delay", amount_won=50000
    ) == ("invalid_arguments")
    assert code(shop, ctx, "think") == "invalid_arguments"


# -------------------------------------------------------------------------------------------- policy checks


def _coupon_for_the_order(session: Session) -> None:
    add_compensation_coupon(session, "CP-O-4-1", "C-1", "O-4", kst(9, 10))


def _two_recent_coupons(session: Session) -> None:
    add_compensation_coupon(session, "CP-O-1-1", "C-1", "O-1", kst(8, 20))
    add_compensation_coupon(session, "CP-O-2-1", "C-1", "O-2", kst(9, 10))


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
        "coupon_not_eligible",
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
        "coupon_already_issued_for_order",
        "coupon_limit_exceeded",
    }


def test_the_return_window_closes_at_kst_midnight(shop):
    args = {"order_id": "O-4", "line_nos": [1], "reason": "defective"}  # delivered 09-02 14:20 KST
    assert code(shop, make_ctx(now=kst(9, 10, 0, 0)), "request_return", **args) == "return_window_expired"
    out = ok(shop, make_ctx(now=kst(9, 9, 23, 59)), "request_return", **args)
    assert out["request_id"] == "RT-O-4-1"


def test_p0_records_every_rule_a_call_breaks(shop):
    ctx = make_ctx(p1=False)
    args = {"order_id": "O-5", "line_no": 1, "new_variant_id": "V-2-01", "reason": "changed_mind"}
    result = call(shop, ctx, "request_exchange", **args)
    assert result.ok and result.violations == ("exchange_not_delivered", "exchange_different_product")


# -------------------------------------------------------------------------------------------------- effects


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


def test_a_defective_request_makes_the_order_eligible(shop):
    ctx = make_ctx()
    assert code(shop, ctx, "issue_compensation_coupon", order_id="O-3", reason="defective_item") == (
        "coupon_not_eligible"
    )
    ok(shop, ctx, "request_exchange", order_id="O-3", line_no=1, new_variant_id="V-1-02", reason="defective")
    out = ok(shop, ctx, "issue_compensation_coupon", order_id="O-3", reason="defective_item")
    assert out["coupon_id"] == "CP-O-3-1" and out["amount_won"] == 3000


def test_a_changed_mind_request_does_not_make_the_order_eligible(shop):
    ctx = make_ctx()
    ok(shop, ctx, "request_return", order_id="O-3", line_nos=[1], reason="changed_mind")
    assert code(shop, ctx, "issue_compensation_coupon", order_id="O-3", reason="defective_item") == (
        "coupon_not_eligible"
    )


def test_p0_coupons_that_are_not_due_get_the_lowest_amount_and_the_next_number(shop):
    ctx = make_ctx(p1=False)
    first = ok(shop, ctx, "issue_compensation_coupon", order_id="O-5", reason="delivery_delay")
    assert first == {"coupon_id": "CP-O-5-1", "amount_won": 2000, "expires_at": "2026-10-14 10:00"}
    second = ok(shop, ctx, "issue_compensation_coupon", order_id="O-5", reason="defective_item")
    assert (second["coupon_id"], second["amount_won"]) == ("CP-O-5-2", 3000)
    assert ctx.violations == ["coupon_not_eligible", "coupon_not_eligible", "coupon_already_issued_for_order"]


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


def test_transfer_to_human_writes_a_handoff_row(shop):
    out = ok(shop, make_ctx(None), "transfer_to_human", reason="cannot_verify", summary="본인 확인 실패")
    assert out == {"handoff_id": "HO-1", "message": HANDOFF_MESSAGE}
    out = ok(shop, make_ctx("C-1"), "transfer_to_human", reason="customer_request", summary="상담원 요청")
    assert out["handoff_id"] == "HO-2"
    handoffs = rows(shop, "handoffs")
    assert (handoffs["HO-1"]["customer_id"], handoffs["HO-2"]["customer_id"]) == (None, "C-1")
    assert handoffs["HO-1"]["reason"] == "cannot_verify"
    assert handoffs["HO-2"]["created_at"] == "2026-09-14T01:00:00+00:00"


# ------------------------------------------------------------------------------ repeated calls, determinism

TWICE_REFUSED = {
    "cancel_order": "already_cancelled",
    "request_return": "already_requested",
    "request_exchange": "already_requested",
    "change_shipping_address": "same_address",
}


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


def play(source, *, now=NOW, sequence=SEQUENCE) -> tuple[list[str], str]:
    engine = db.memory_engine(source)
    ctx = make_ctx(None, now=now)
    outputs = [execute(REGISTRY, engine, ctx, name, args).content for name, args in sequence]
    digest = db.state_hash(db.dump_db(engine))
    engine.dispose()
    return outputs, digest


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


def test_content_ids_do_not_depend_on_the_order_of_calls(shop):
    swapped = [SEQUENCE[0], SEQUENCE[3], SEQUENCE[2], *SEQUENCE[4:]]
    assert play(shop, sequence=swapped)[1] == play(shop, sequence=[SEQUENCE[0], *SEQUENCE[2:]])[1]


def test_free_text_is_left_out_of_the_comparison_and_time_comes_from_ctx_now(shop):
    reworded = [(n, {**a, "body": "다른 본문"} if n == "create_ticket" else a) for n, a in SEQUENCE]
    assert play(shop, sequence=reworded)[1] == play(shop)[1]
    assert play(shop, now=NOW + timedelta(minutes=1))[1] != play(shop)[1]


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


# ------------------------------------------------------------------------------------------ policy document


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
        f"최근 {rules.COUPON_WINDOW_DAYS}일 동안 {rules.COUPON_WINDOW_LIMIT}장까지",
        f"발급일부터 {rules.COUPON_VALID_DAYS}일 동안",
        "주문당 1장",
    ]:
        assert value in text, value
    assert len(text) < 3500  # it shares a 16k context with the tool schemas and the conversation
