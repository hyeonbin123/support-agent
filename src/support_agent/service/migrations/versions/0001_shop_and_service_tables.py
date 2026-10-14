"""쇼핑몰 테이블과 서비스 테이블

Revision ID: 0001
Revises:
"""

# [파일 역할] 서비스 DB의 첫(이자 지금 유일한) 마이그레이션. 쇼핑몰 테이블 13개(db.py)와 서비스 테이블
# 3개(service/store.py)를 빈 DB에 만든다.
# [설계 이유] 본문은 Alembic의 autogenerate가 모델 선언에서 만든 것이다. 손으로 고치지 않고 선언과 같게 두며,
# tests/test_postgres.py가 "마이그레이션 결과 = 선언한 테이블"인지 확인한다(docs/service.md 파일 표).
# [주요 구성] upgrade는 외래 키가 가리키는 부모 테이블부터 만들고, downgrade는 그 역순으로 지운다.
# [연결] service/bootstrap.py의 migrate가 앱이 뜰 때 head까지 올린다.

import sqlalchemy as sa
from alembic import op

# UTCDateTime(시간대 있는 UTC 시각만 받는 열 타입)을 열 정의에 그대로 쓰기 위해 프로젝트의 db 모듈을 가져온다.
import support_agent.db

# Alembic의 버전 사슬 정보. down_revision이 None이라 이 파일이 사슬의 첫 고리다.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


# 빈 DB에 테이블과 인덱스를 만든다.
# Enum 열은 native_enum=False, create_constraint=True라서 PostgreSQL의 고유 ENUM 타입 대신 문자열 열 + CHECK
# 제약이 된다. SQLite와 PostgreSQL에서 같은 방식으로 값이 검사되고 저장된다(db.py의 설명과 같다).
# 제약 이름(op.f("pk_...") 등)은 db.py의 NAMING_CONVENTION 규칙으로 정해진 이름이다.
def upgrade() -> None:
    # ### Alembic이 자동 생성한 명령 - 검토하고 필요하면 고친다 ###
    # --- 쇼핑몰 테이블: 다른 테이블을 가리키지 않는 고객·상품부터 만든다 ---
    # customers: 가상 고객. 본인 확인(find_customer)이 이름 + 전화번호/이메일로 찾는 대상이다.
    op.create_table(
        "customers",
        sa.Column("id", sa.String(length=20), nullable=False),
        sa.Column("name", sa.String(length=40), nullable=False),
        sa.Column("phone", sa.String(length=20), nullable=False),
        sa.Column("email", sa.String(length=80), nullable=False),
        sa.Column(
            "grade",
            sa.Enum(
                "normal", "vip", name="customer_grade", native_enum=False, create_constraint=True, length=30
            ),
            nullable=False,
        ),
        sa.Column("joined_at", support_agent.db.UTCDateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_customers")),
    )
    # products: 상품. 옵션(색·크기)별 가격과 재고는 product_variants에 따로 둔다.
    op.create_table(
        "products",
        sa.Column("id", sa.String(length=20), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("category", sa.String(length=40), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_products")),
    )
    # customer_addresses: 고객의 배송지 목록. 배송지 변경 도구는 이 표의 주소만 고를 수 있다.
    op.create_table(
        "customer_addresses",
        sa.Column("id", sa.String(length=30), nullable=False),
        sa.Column("customer_id", sa.String(length=20), nullable=False),
        sa.Column("label", sa.String(length=20), nullable=False),
        sa.Column("recipient", sa.String(length=40), nullable=False),
        sa.Column("postal_code", sa.String(length=10), nullable=False),
        sa.Column("address", sa.String(length=200), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.id"], name=op.f("fk_customer_addresses_customer_id_customers")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_customer_addresses")),
    )
    # handoffs: 상담원 이관 기록. 본인 확인 전 이관도 있어 customer_id가 비어 있을 수 있다. 쓸데없는 이관이
    # 판정에서 DB 차이로 드러나도록 행으로 남긴다(db.py).
    op.create_table(
        "handoffs",
        sa.Column("id", sa.String(length=20), nullable=False),
        sa.Column("customer_id", sa.String(length=20), nullable=True),
        sa.Column(
            "reason",
            sa.Enum(
                "customer_request",
                "out_of_scope",
                "cannot_verify",
                name="handoff_reason",
                native_enum=False,
                create_constraint=True,
                length=30,
            ),
            nullable=False,
        ),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("created_at", support_agent.db.UTCDateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.id"], name=op.f("fk_handoffs_customer_id_customers")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_handoffs")),
    )
    # product_variants: 상품 옵션별 가격·재고. 교환은 같은 상품의 다른 옵션으로만 되고 재고를 줄인다.
    op.create_table(
        "product_variants",
        sa.Column("id", sa.String(length=20), nullable=False),
        sa.Column("product_id", sa.String(length=20), nullable=False),
        sa.Column("option_label", sa.String(length=60), nullable=False),
        sa.Column("price_won", sa.Integer(), nullable=False),
        sa.Column("stock", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["product_id"], ["products.id"], name=op.f("fk_product_variants_product_id_products")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_product_variants")),
    )
    # orders: 주문. 배송지 열(ship_*)은 주문 당시 주소의 복사본이라 고객이 주소록을 바꿔도 주문의 배송지는
    # 그대로다. 취소 시각과 사유는 취소된 주문에만 있다.
    op.create_table(
        "orders",
        sa.Column("id", sa.String(length=20), nullable=False),
        sa.Column("customer_id", sa.String(length=20), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "paid",
                "preparing",
                "shipped",
                "delivered",
                "cancelled",
                name="order_status",
                native_enum=False,
                create_constraint=True,
                length=30,
            ),
            nullable=False,
        ),
        sa.Column("ordered_at", support_agent.db.UTCDateTime(timezone=True), nullable=False),
        sa.Column("items_won", sa.Integer(), nullable=False),
        sa.Column("shipping_fee_won", sa.Integer(), nullable=False),
        sa.Column("discount_won", sa.Integer(), nullable=False),
        sa.Column("total_won", sa.Integer(), nullable=False),
        sa.Column("ship_address_id", sa.String(length=30), nullable=False),
        sa.Column("ship_recipient", sa.String(length=40), nullable=False),
        sa.Column("ship_postal_code", sa.String(length=10), nullable=False),
        sa.Column("ship_address", sa.String(length=200), nullable=False),
        sa.Column("cancelled_at", support_agent.db.UTCDateTime(timezone=True), nullable=True),
        sa.Column(
            "cancel_reason",
            sa.Enum(
                "changed_mind",
                "ordered_by_mistake",
                "delivery_too_slow",
                name="cancel_reason",
                native_enum=False,
                create_constraint=True,
                length=30,
            ),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.id"], name=op.f("fk_orders_customer_id_customers")
        ),
        sa.ForeignKeyConstraint(
            ["ship_address_id"],
            ["customer_addresses.id"],
            name=op.f("fk_orders_ship_address_id_customer_addresses"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_orders")),
    )
    # coupons: 쿠폰. kind가 compensation인 보상 쿠폰만 도구가 발급하며 사유(reason)와 주문(order_id)이 붙는다.
    op.create_table(
        "coupons",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("customer_id", sa.String(length=20), nullable=False),
        sa.Column(
            "kind",
            sa.Enum(
                "promo",
                "compensation",
                name="coupon_kind",
                native_enum=False,
                create_constraint=True,
                length=30,
            ),
            nullable=False,
        ),
        sa.Column("amount_won", sa.Integer(), nullable=False),
        sa.Column(
            "reason",
            sa.Enum(
                "delivery_delay",
                "defective_item",
                name="compensation_reason",
                native_enum=False,
                create_constraint=True,
                length=30,
            ),
            nullable=True,
        ),
        sa.Column("order_id", sa.String(length=20), nullable=True),
        sa.Column("issued_at", support_agent.db.UTCDateTime(timezone=True), nullable=False),
        sa.Column("expires_at", support_agent.db.UTCDateTime(timezone=True), nullable=False),
        sa.Column("used_at", support_agent.db.UTCDateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.id"], name=op.f("fk_coupons_customer_id_customers")
        ),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"], name=op.f("fk_coupons_order_id_orders")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_coupons")),
    )
    # order_items: 주문의 줄(상품 한 종류). 기본 키가 (주문, 줄 번호)이고, 반품·교환은 줄 단위로 상태가
    # 바뀐다. 상품명·옵션명은 주문 당시 값의 복사본이다.
    op.create_table(
        "order_items",
        sa.Column("order_id", sa.String(length=20), nullable=False),
        sa.Column("line_no", sa.Integer(), nullable=False),
        sa.Column("variant_id", sa.String(length=20), nullable=False),
        sa.Column("product_name", sa.String(length=80), nullable=False),
        sa.Column("option_label", sa.String(length=60), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("unit_price_won", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "ordered",
                "cancelled",
                "return_requested",
                "exchange_requested",
                name="item_status",
                native_enum=False,
                create_constraint=True,
                length=30,
            ),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"], name=op.f("fk_order_items_order_id_orders")),
        sa.ForeignKeyConstraint(
            ["variant_id"], ["product_variants.id"], name=op.f("fk_order_items_variant_id_product_variants")
        ),
        sa.PrimaryKeyConstraint("order_id", "line_no", name=op.f("pk_order_items")),
    )
    # payments: 주문당 결제 한 건(기본 키가 order_id). 취소하면 상태가 refund_pending이 되고 refund_won에
    # 환불액이 들어간다.
    op.create_table(
        "payments",
        sa.Column("order_id", sa.String(length=20), nullable=False),
        sa.Column(
            "method",
            sa.Enum(
                "card",
                "bank_transfer",
                "simple_pay",
                name="payment_method",
                native_enum=False,
                create_constraint=True,
                length=30,
            ),
            nullable=False,
        ),
        sa.Column("amount_won", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "paid",
                "refund_pending",
                "refunded",
                name="payment_status",
                native_enum=False,
                create_constraint=True,
                length=30,
            ),
            nullable=False,
        ),
        sa.Column("refund_won", sa.Integer(), nullable=False),
        sa.Column("paid_at", support_agent.db.UTCDateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"], name=op.f("fk_payments_order_id_orders")),
        sa.PrimaryKeyConstraint("order_id", name=op.f("pk_payments")),
    )
    # service_requests: 반품·교환 접수. id는 내용에서 만든다(RT-주문-줄, EX-주문-줄). 서비스의 승인 관문은
    # 같은 주문에 접수된 반품의 refund_won 합계를 이 표에서 읽는다(core.py).
    op.create_table(
        "service_requests",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("order_id", sa.String(length=20), nullable=False),
        sa.Column(
            "kind",
            sa.Enum(
                "return",
                "exchange",
                name="request_kind",
                native_enum=False,
                create_constraint=True,
                length=30,
            ),
            nullable=False,
        ),
        sa.Column(
            "reason",
            sa.Enum(
                "changed_mind",
                "defective",
                "wrong_item",
                name="request_reason",
                native_enum=False,
                create_constraint=True,
                length=30,
            ),
            nullable=False,
        ),
        sa.Column("refund_won", sa.Integer(), nullable=False),
        sa.Column("return_fee_won", sa.Integer(), nullable=False),
        sa.Column("created_at", support_agent.db.UTCDateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["order_id"], ["orders.id"], name=op.f("fk_service_requests_order_id_orders")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_service_requests")),
    )
    # shipments: 주문당 배송 한 건. promised_by(도착 예정일, 날짜만)는 배송 지연 보상 쿠폰의 자격 계산에
    # 쓰인다.
    op.create_table(
        "shipments",
        sa.Column("order_id", sa.String(length=20), nullable=False),
        sa.Column("carrier", sa.String(length=40), nullable=False),
        sa.Column("tracking_no", sa.String(length=30), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "ready",
                "in_transit",
                "delivered",
                name="shipment_status",
                native_enum=False,
                create_constraint=True,
                length=30,
            ),
            nullable=False,
        ),
        sa.Column("shipped_at", support_agent.db.UTCDateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", support_agent.db.UTCDateTime(timezone=True), nullable=True),
        sa.Column("promised_by", sa.Date(), nullable=False),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"], name=op.f("fk_shipments_order_id_orders")),
        sa.PrimaryKeyConstraint("order_id", name=op.f("pk_shipments")),
    )
    # tickets: 상담 티켓. 본문(body)은 모델이 쓰는 자유 글이라 평가의 DB 비교에서 뺀다(db.IGNORED_COLUMNS).
    op.create_table(
        "tickets",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("customer_id", sa.String(length=20), nullable=False),
        sa.Column("order_id", sa.String(length=20), nullable=True),
        sa.Column(
            "category",
            sa.Enum(
                "delivery",
                "refund",
                "product",
                "account",
                "other",
                name="ticket_category",
                native_enum=False,
                create_constraint=True,
                length=30,
            ),
            nullable=False,
        ),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", support_agent.db.UTCDateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.id"], name=op.f("fk_tickets_customer_id_customers")
        ),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"], name=op.f("fk_tickets_order_id_orders")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tickets")),
    )
    # service_request_items: 반품·교환 접수에 포함된 줄. 교환이면 바꿀 옵션(exchange_variant_id)이 붙는다.
    op.create_table(
        "service_request_items",
        sa.Column("request_id", sa.String(length=40), nullable=False),
        sa.Column("line_no", sa.Integer(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("exchange_variant_id", sa.String(length=20), nullable=True),
        sa.ForeignKeyConstraint(
            ["exchange_variant_id"],
            ["product_variants.id"],
            name=op.f("fk_service_request_items_exchange_variant_id_product_variants"),
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["service_requests.id"],
            name=op.f("fk_service_request_items_request_id_service_requests"),
        ),
        sa.PrimaryKeyConstraint("request_id", "line_no", name=op.f("pk_service_request_items")),
    )
    # --- 서비스 테이블(store.py): 판정 덤프에 들어가지 않는 별도 메타데이터 ---
    # chat_sessions: 상담 세션과 대화 상태(JSON). id가 곧 세션의 열쇠다.
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("created_at", support_agent.db.UTCDateTime(timezone=True), nullable=False),
        sa.Column("updated_at", support_agent.db.UTCDateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("turns", sa.Integer(), nullable=False),
        sa.Column("verified_customer_id", sa.String(length=20), nullable=True),
        sa.Column("state", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_sessions")),
    )
    # approvals: 사람 승인 대기열. 세션별·상태별로 찾으므로 아래에서 두 열에 인덱스를 만든다.
    op.create_table(
        "approvals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", support_agent.db.UTCDateTime(timezone=True), nullable=False),
        sa.Column("tool", sa.String(length=64), nullable=False),
        sa.Column("args", sa.JSON(), nullable=False),
        sa.Column("customer_id", sa.String(length=20), nullable=True),
        sa.Column("refund_won", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("decided_at", support_agent.db.UTCDateTime(timezone=True), nullable=True),
        sa.Column("decided_by", sa.String(length=64), nullable=True),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("result", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["chat_sessions.id"], name=op.f("fk_approvals_session_id_chat_sessions")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_approvals")),
    )
    op.create_index(op.f("ix_approvals_session_id"), "approvals", ["session_id"], unique=False)
    op.create_index(op.f("ix_approvals_status"), "approvals", ["status"], unique=False)
    # audit_events: 추가만 하는 감사 로그. 세션별·종류별 조회용 인덱스를 아래에서 만든다.
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("at", support_agent.db.UTCDateTime(timezone=True), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["chat_sessions.id"], name=op.f("fk_audit_events_session_id_chat_sessions")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_events")),
    )
    op.create_index(op.f("ix_audit_events_kind"), "audit_events", ["kind"], unique=False)
    op.create_index(op.f("ix_audit_events_session_id"), "audit_events", ["session_id"], unique=False)
    # ### Alembic 자동 생성 명령 끝 ###


# upgrade를 되돌린다. 자식 테이블과 인덱스를 먼저 지우고 부모 테이블을 나중에 지워 외래 키 제약에 걸리지 않게
# 한다. 프로젝트 코드와 테스트에서 이 함수를 부르는 곳은 없다.
def downgrade() -> None:
    # ### Alembic이 자동 생성한 명령 - 검토하고 필요하면 고친다 ###
    op.drop_index(op.f("ix_audit_events_session_id"), table_name="audit_events")
    op.drop_index(op.f("ix_audit_events_kind"), table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_index(op.f("ix_approvals_status"), table_name="approvals")
    op.drop_index(op.f("ix_approvals_session_id"), table_name="approvals")
    op.drop_table("approvals")
    op.drop_table("chat_sessions")
    op.drop_table("service_request_items")
    op.drop_table("tickets")
    op.drop_table("shipments")
    op.drop_table("service_requests")
    op.drop_table("payments")
    op.drop_table("order_items")
    op.drop_table("coupons")
    op.drop_table("orders")
    op.drop_table("product_variants")
    op.drop_table("handoffs")
    op.drop_table("customer_addresses")
    op.drop_table("products")
    op.drop_table("customers")
    # ### Alembic 자동 생성 명령 끝 ###
