"""The seed database: generated background rows plus the hand-written rows that tasks refer to.

Everything is derived from one random.Random(SEED) instance and explicit timestamps, so two builds give
the same state hash. Build once with build_seed_engine(), copy per episode with db.memory_engine(seed).
"""

from __future__ import annotations

import functools
import random
from datetime import date, datetime, timedelta

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from support_agent import db
from support_agent.clock import KST, kst_date

SEED = 20260919
BULK_START = datetime(2026, 6, 1, 0, 0, tzinfo=KST)
SEED_END = datetime(2026, 9, 10, 23, 59, tzinfo=KST)  # no generated row has a datetime after this

FIXTURE_CUSTOMER_IDS = ("C-9001", "C-9002", "C-9003", "C-9004", "C-9005")
FIXTURE_ORDER_IDS = ("O-90001", "O-90002", "O-90003", "O-90004", "O-90005", "O-90006")

N_CUSTOMERS = 40
N_ORDERS = 120
N_REQUESTS = 8
N_COUPONS = 30
N_TICKETS = 15
FREE_SHIPPING_FROM_WON = 30000
SHIPPING_FEE_WON = 3000
RETURN_FEE_WON = 3000

SURNAMES = ("김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황")
GIVEN_NAMES = (
    "민준", "서준", "도현", "시우", "주원", "지호", "건우", "우진", "선우", "현우",
    "서윤", "지안", "하은", "수아", "지유", "채원", "다은", "예린", "소율", "유나",
    "은우", "연우", "정우", "승현", "태민", "가은", "나연", "미소", "보람", "세아",
)  # fmt: skip
FIXTURE_NAMES = frozenset({"김하준", "이서연", "박도윤", "최지우", "정예준"})

CITIES = ("가온시", "누리시", "다솜시", "라온시", "마루시")
DISTRICTS = ("한빛구", "새별구", "푸른구", "은하구", "솔내구", "달빛구")
ROADS = ("가상로", "새싹길", "너울로", "도담길", "미르로", "소담길", "아름로", "하늘길")
ADDRESS_LABELS = ("집", "회사", "부모님 댁")
CARRIERS = ("한빛택배", "새벽택배", "나래택배", "두루택배")

# (name, category, ((option_label, price_won), ...))
PRODUCTS = (
    ("블루투스 키보드", "전자기기", (("블랙", 32900), ("화이트", 32900))),
    ("무선 마우스", "전자기기", (("블랙", 19800), ("그레이", 19800), ("핑크", 19800))),
    ("보조배터리", "전자기기", (("10000mAh", 27500), ("20000mAh", 39500))),
    ("USB 충전기", "전자기기", (("1포트", 12900), ("2포트", 17900))),
    ("스마트폰 거치대", "전자기기", (("블랙", 9900), ("실버", 9900))),
    ("LED 스탠드", "생활용품", (("화이트", 28900), ("블랙", 28900))),
    ("스테인리스 보온병", "주방용품", (("350ml", 16500), ("500ml", 18500), ("750ml", 21500))),
    ("머그컵 세트", "주방용품", (("2인", 13800), ("4인", 24800))),
    ("프라이팬", "주방용품", (("24cm", 25900), ("28cm", 29900))),
    ("도마 세트", "주방용품", (("소", 11900), ("대", 15900))),
    ("밀폐용기 세트", "주방용품", (("6종", 17400), ("10종", 26400))),
    ("면 티셔츠", "의류", (("화이트 / M", 15900), ("화이트 / L", 15900), ("블랙 / M", 15900))),
    ("후드 집업", "의류", (("그레이 / M", 42800), ("그레이 / L", 42800), ("네이비 / L", 42800))),
    ("기모 트레이닝 팬츠", "의류", (("블랙 / M", 29700), ("블랙 / L", 29700))),
    ("경량 패딩 조끼", "의류", (("베이지 / M", 49500), ("카키 / L", 49500))),
    ("니트 머플러", "의류", (("아이보리", 18600), ("차콜", 18600), ("와인", 18600))),
    ("캔버스 스니커즈", "신발", (("240mm", 36700), ("250mm", 36700), ("260mm", 36700))),
    ("슬리퍼", "신발", (("M", 8900), ("L", 8900))),
    ("등산 양말", "의류", (("그레이", 4900), ("블랙", 4900))),
    ("요가 매트", "스포츠", (("6mm", 22400), ("10mm", 27400))),
    ("폼롤러", "스포츠", (("45cm", 14300), ("90cm", 23300))),
    ("아령 세트", "스포츠", (("2kg", 12600), ("4kg", 19600), ("6kg", 26600))),
    ("백팩", "가방", (("블랙", 45200), ("네이비", 45200))),
    ("에코백", "가방", (("아이보리", 9800), ("블랙", 9800))),
    ("수면 안대", "생활용품", (("그레이", 6400), ("네이비", 6400))),
)  # fmt: skip

TICKET_BODIES = {
    db.TicketCategory.DELIVERY: ("배송이 언제 오는지 문의", "부재 시 문 앞에 놓아 달라는 요청"),
    db.TicketCategory.REFUND: ("환불 입금 시점 문의", "환불 금액 확인 요청"),
    db.TicketCategory.PRODUCT: ("상품 사용법 문의", "재입고 일정 문의"),
    db.TicketCategory.ACCOUNT: ("이메일 주소 변경 문의", "회원 등급 기준 문의"),
    db.TicketCategory.OTHER: ("영수증 발급 문의", "포장 방법 관련 의견"),
}

_MINUTE = timedelta(minutes=1)
_HOUR = timedelta(hours=1)
_DAY = timedelta(days=1)


def _between(rng: random.Random, start: datetime, end: datetime) -> datetime:
    """A whole-minute time in [start, end]."""
    return start + rng.randrange((end - start) // _MINUTE + 1) * _MINUTE


def kst(month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=KST)


def shipping_fee(items_won: int) -> int:
    return 0 if items_won >= FREE_SHIPPING_FROM_WON else SHIPPING_FEE_WON


def new_order(
    order_id: str,
    customer_id: str,
    status: db.OrderStatus,
    ordered_at: datetime,
    address: db.CustomerAddress,
    lines: list[tuple[db.Product, db.ProductVariant, int]],
    discount_won: int = 0,
    cancelled_at: datetime | None = None,
    cancel_reason: db.CancelReason | None = None,
) -> tuple[db.Order, list[db.OrderItem]]:
    """An order and its items, with totals and snapshots derived from the sources."""
    items_won = sum(variant.price_won * qty for _, variant, qty in lines)
    fee = shipping_fee(items_won)
    order = db.Order(
        id=order_id,
        customer_id=customer_id,
        status=status,
        ordered_at=ordered_at,
        items_won=items_won,
        shipping_fee_won=fee,
        discount_won=discount_won,
        total_won=items_won + fee - discount_won,
        ship_address_id=address.id,
        ship_recipient=address.recipient,
        ship_postal_code=address.postal_code,
        ship_address=address.address,
        cancelled_at=cancelled_at,
        cancel_reason=cancel_reason,
    )
    item_status = db.ItemStatus.CANCELLED if status == db.OrderStatus.CANCELLED else db.ItemStatus.ORDERED
    items = [
        db.OrderItem(
            order_id=order_id,
            line_no=line_no,
            variant_id=variant.id,
            product_name=product.name,
            option_label=variant.option_label,
            quantity=qty,
            unit_price_won=variant.price_won,
            status=item_status,
        )
        for line_no, (product, variant, qty) in enumerate(lines, start=1)
    ]
    return order, items


class Rows:
    """Rows grouped by table, flushed in foreign-key order."""

    def __init__(self) -> None:
        self.customers: list[db.Customer] = []
        self.addresses: list[db.CustomerAddress] = []
        self.products: list[db.Product] = []
        self.variants: list[db.ProductVariant] = []
        self.orders: list[db.Order] = []
        self.items: list[db.OrderItem] = []
        self.payments: list[db.Payment] = []
        self.shipments: list[db.Shipment] = []
        self.requests: list[db.ServiceRequest] = []
        self.request_items: list[db.ServiceRequestItem] = []
        self.coupons: list[db.Coupon] = []
        self.tickets: list[db.Ticket] = []

    def groups(self) -> list[list]:
        return [
            self.customers,
            self.addresses,
            self.products,
            self.variants,
            self.orders,
            self.items + self.payments + self.shipments,
            self.requests,
            self.request_items,
            self.coupons + self.tickets,
        ]


def _bulk_customers(rng: random.Random, rows: Rows) -> None:
    phones = rng.sample(range(1000, 9000), N_CUSTOMERS)  # fixtures use 9001..9005
    for i in range(1, N_CUSTOMERS + 1):
        name = rng.choice(SURNAMES) + rng.choice(GIVEN_NAMES)
        while name in FIXTURE_NAMES:
            name = rng.choice(SURNAMES) + rng.choice(GIVEN_NAMES)
        customer = db.Customer(
            id=f"C-{i:04d}",
            name=name,
            phone=f"0100000{phones[i - 1]:04d}",
            email=f"user{i:04d}@example.com",
            grade=db.CustomerGrade.VIP if rng.random() < 0.15 else db.CustomerGrade.NORMAL,
            joined_at=_between(rng, BULK_START, kst(8, 5, 23, 59)),
        )
        rows.customers.append(customer)
        n_addresses = 1 + (rng.random() < 0.55) + (rng.random() < 0.2)
        for n in range(1, n_addresses + 1):
            recipient = name if n < 3 else rng.choice(SURNAMES) + rng.choice(GIVEN_NAMES)
            road = f"{rng.choice(CITIES)} {rng.choice(DISTRICTS)} {rng.choice(ROADS)} {rng.randrange(1, 200)}"
            unit = f"{rng.randrange(101, 110)}동 {rng.randrange(1, 20)}0{rng.randrange(1, 5)}호"
            rows.addresses.append(
                db.CustomerAddress(
                    id=f"AD-{customer.id}-{n}",
                    customer_id=customer.id,
                    label=ADDRESS_LABELS[n - 1],
                    recipient=recipient,
                    postal_code=f"{rng.randrange(1000, 100000):05d}",
                    address=f"{road}, {unit}",
                    is_default=n == 1,
                )
            )


def _bulk_products(rng: random.Random, rows: Rows) -> None:
    for i, (name, category, options) in enumerate(PRODUCTS, start=1):
        rows.products.append(db.Product(id=f"P-{i:04d}", name=name, category=category))
        for n, (label, price) in enumerate(options, start=1):
            stock = 0 if rng.random() < 0.1 else rng.randrange(1, 41)
            rows.variants.append(
                db.ProductVariant(
                    id=f"V-{i:04d}-{n:02d}",
                    product_id=f"P-{i:04d}",
                    option_label=label,
                    price_won=price,
                    stock=stock,
                )
            )


_STATUS_WEIGHTS = (
    (db.OrderStatus.DELIVERED, 60),
    (db.OrderStatus.CANCELLED, 12),
    (db.OrderStatus.SHIPPED, 10),
    (db.OrderStatus.PREPARING, 9),
    (db.OrderStatus.PAID, 9),
)


def _ordered_at(rng: random.Random, status: db.OrderStatus, joined_at: datetime) -> datetime:
    """Early enough that every later event of this status still fits before SEED_END."""
    if status == db.OrderStatus.DELIVERED:
        return _between(rng, joined_at + _HOUR, SEED_END - 6 * _DAY)
    if status == db.OrderStatus.CANCELLED:
        return _between(rng, joined_at + _HOUR, SEED_END - 2 * _DAY)
    if status == db.OrderStatus.SHIPPED:
        return _between(rng, SEED_END - 4 * _DAY, SEED_END - 2 * _DAY)
    if status == db.OrderStatus.PREPARING:
        return _between(rng, SEED_END - 2 * _DAY, SEED_END - 12 * _HOUR)
    return _between(rng, SEED_END - _DAY, SEED_END - _HOUR)


def _bulk_orders(rng: random.Random, rows: Rows) -> None:
    products = {p.id: p for p in rows.products}
    addresses: dict[str, list[db.CustomerAddress]] = {}
    for address in rows.addresses:
        addresses.setdefault(address.customer_id, []).append(address)
    statuses = [status for status, _ in _STATUS_WEIGHTS]
    weights = [weight for _, weight in _STATUS_WEIGHTS]

    plans = []
    for _ in range(N_ORDERS):
        customer = rng.choice(rows.customers)
        status = rng.choices(statuses, weights)[0]
        plans.append((_ordered_at(rng, status, customer.joined_at), customer.id, status))
    plans.sort(key=lambda plan: (plan[0], plan[1]))  # order ids grow with time

    joined = {c.id: c.joined_at for c in rows.customers}
    promo_count: dict[str, int] = {}
    for number, (ordered_at, customer_id, status) in enumerate(plans, start=10001):
        order_id = f"O-{number}"
        variants = rng.sample(rows.variants, rng.choices((1, 2, 3), (60, 30, 10))[0])
        lines = [(products[v.product_id], v, rng.choices((1, 2), (80, 20))[0]) for v in variants]
        items_won = sum(v.price_won * qty for _, v, qty in lines)

        # A discount always comes from a promo coupon that expired (or was used) before SEED_END.
        discount = 0
        if ordered_at <= SEED_END - 30 * _DAY and rng.random() < 0.2:
            amounts = [a for a in (1000, 2000, 3000, 5000) if a * 2 <= items_won]
            if amounts:
                discount = rng.choice(amounts)
                issued_at = _between(rng, max(joined[customer_id], ordered_at - 10 * _DAY), ordered_at)
                promo_count[customer_id] = promo_count.get(customer_id, 0) + 1
                rows.coupons.append(
                    db.Coupon(
                        id=f"CP-{customer_id}-P{promo_count[customer_id]}",
                        customer_id=customer_id,
                        kind=db.CouponKind.PROMO,
                        amount_won=discount,
                        reason=None,
                        order_id=None,
                        issued_at=issued_at,
                        expires_at=issued_at + 30 * _DAY,
                        used_at=ordered_at,
                    )
                )

        cancelled = status == db.OrderStatus.CANCELLED
        cancelled_at = ordered_at + rng.randrange(10, 20 * 60) * _MINUTE if cancelled else None
        order, items = new_order(
            order_id,
            customer_id,
            status,
            ordered_at,
            rng.choice(addresses[customer_id]),
            lines,
            discount_won=discount,
            cancelled_at=cancelled_at,
            cancel_reason=rng.choice(list(db.CancelReason)) if cancelled else None,
        )
        rows.orders.append(order)
        rows.items.extend(items)

        if not cancelled:
            pay_status = db.PaymentStatus.PAID
        elif cancelled_at <= SEED_END - 5 * _DAY:
            pay_status = db.PaymentStatus.REFUNDED
        else:
            pay_status = db.PaymentStatus.REFUND_PENDING
        rows.payments.append(
            db.Payment(
                order_id=order_id,
                method=rng.choice(list(db.PaymentMethod)),
                amount_won=order.total_won,
                status=pay_status,
                refund_won=order.total_won if cancelled else 0,
                paid_at=ordered_at + rng.randrange(0, 11) * _MINUTE,
            )
        )

        shipped_at = delivered_at = None
        ship_status = db.ShipmentStatus.READY
        if status in (db.OrderStatus.SHIPPED, db.OrderStatus.DELIVERED):
            shipped_at = ordered_at + rng.randrange(18 * 60, 47 * 60) * _MINUTE
            ship_status = db.ShipmentStatus.IN_TRANSIT
        if status == db.OrderStatus.DELIVERED:
            delivered_at = shipped_at + rng.randrange(20 * 60, 72 * 60) * _MINUTE
            ship_status = db.ShipmentStatus.DELIVERED
        rows.shipments.append(
            db.Shipment(
                order_id=order_id,
                carrier=rng.choice(CARRIERS),
                tracking_no=f"{rng.randrange(1000, 5550)}-{rng.randrange(1000, 10000)}-{number % 10000:04d}",
                status=ship_status,
                shipped_at=shipped_at,
                delivered_at=delivered_at,
                promised_by=kst_date(ordered_at) + 3 * _DAY,
            )
        )


def _bulk_requests(rng: random.Random, rows: Rows) -> None:
    """A few returns and exchanges on delivered orders; at most one request per order."""
    shipments = {s.order_id: s for s in rows.shipments}
    items: dict[str, list[db.OrderItem]] = {}
    for item in rows.items:
        items.setdefault(item.order_id, []).append(item)
    variants = {v.id: v for v in rows.variants}
    candidates = [o for o in rows.orders if o.status == db.OrderStatus.DELIVERED and o.discount_won == 0]

    for index, order in enumerate(rng.sample(candidates, N_REQUESTS)):
        delivered_at = shipments[order.id].delivered_at
        latest = min(delivered_at + 5 * _DAY, SEED_END)
        created_at = _between(rng, delivered_at + _HOUR, latest)
        reason = rng.choice(list(db.RequestReason))
        fee = RETURN_FEE_WON if reason == db.RequestReason.CHANGED_MIND else 0
        lines = items[order.id]

        swap = None
        if index >= 5:  # the last three are exchanges when another option is in stock
            line = rng.choice(lines)
            current = variants[line.variant_id]
            others = [
                v
                for v in rows.variants
                if v.product_id == current.product_id and v.id != current.id and v.stock > 0
            ]
            if others:
                swap = (line, rng.choice(others))

        if swap is not None:
            line, new_variant = swap
            line.status = db.ItemStatus.EXCHANGE_REQUESTED
            request_id = f"EX-{order.id}-{line.line_no}"
            kind, refund, chosen = db.RequestKind.EXCHANGE, 0, [(line, new_variant.id)]
        else:
            picked = sorted(rng.sample(lines, rng.randrange(1, len(lines) + 1)), key=lambda it: it.line_no)
            for line in picked:
                line.status = db.ItemStatus.RETURN_REQUESTED
            request_id = f"RT-{order.id}-{picked[0].line_no}"
            refund = sum(it.unit_price_won * it.quantity for it in picked) - fee
            kind, chosen = db.RequestKind.RETURN, [(it, None) for it in picked]

        rows.requests.append(
            db.ServiceRequest(
                id=request_id,
                order_id=order.id,
                kind=kind,
                reason=reason,
                refund_won=refund,
                return_fee_won=fee,
                created_at=created_at,
            )
        )
        for line, exchange_variant_id in chosen:
            rows.request_items.append(
                db.ServiceRequestItem(
                    request_id=request_id,
                    line_no=line.line_no,
                    quantity=line.quantity,
                    exchange_variant_id=exchange_variant_id,
                )
            )


def _bulk_coupons(rng: random.Random, rows: Rows) -> None:
    """Compensation coupons where the rule allows one, then unused promo coupons up to N_COUPONS.

    Coupons are issued early enough that expires_at (issue + 30 days) is not after SEED_END.
    """
    shipments = {s.order_id: s for s in rows.shipments}
    orders = {o.id: o for o in rows.orders}
    limit = SEED_END - 31 * _DAY
    grants: list[tuple[db.Order, db.CompensationReason, int, datetime]] = []
    for request in rows.requests:
        if request.reason == db.RequestReason.DEFECTIVE and request.created_at <= limit:
            reason = db.CompensationReason.DEFECTIVE_ITEM
            grants.append((orders[request.order_id], reason, 3000, request.created_at))
    late = []
    for order in rows.orders:
        shipment = shipments[order.id]
        if shipment.delivered_at is not None and shipment.delivered_at <= limit:
            days_late = (kst_date(shipment.delivered_at) - shipment.promised_by).days
            if days_late >= 1:
                late.append((order, 2000 if days_late <= 2 else 5000, shipment.delivered_at))
    for order, amount, delivered_at in rng.sample(late, min(4, len(late))):
        grants.append((order, db.CompensationReason.DELIVERY_DELAY, amount, delivered_at))

    seen_customers: set[str] = set()
    for order, reason, amount, base in grants:
        if order.customer_id in seen_customers:  # keeps "one per order, two per 30 days" trivially true
            continue
        seen_customers.add(order.customer_id)
        issued_at = base + rng.randrange(30, 24 * 60) * _MINUTE
        rows.coupons.append(
            db.Coupon(
                id=f"CP-{order.id}-1",
                customer_id=order.customer_id,
                kind=db.CouponKind.COMPENSATION,
                amount_won=amount,
                reason=reason,
                order_id=order.id,
                issued_at=issued_at,
                expires_at=issued_at + 30 * _DAY,
                used_at=None,
            )
        )

    promo_count: dict[str, int] = {}
    for coupon in rows.coupons:
        if coupon.kind == db.CouponKind.PROMO:
            promo_count[coupon.customer_id] = promo_count.get(coupon.customer_id, 0) + 1
    while len(rows.coupons) < N_COUPONS:
        customer = rng.choice(rows.customers)
        valid_days = rng.choice((14, 30))
        issued_at = _between(rng, customer.joined_at, SEED_END - valid_days * _DAY)
        promo_count[customer.id] = promo_count.get(customer.id, 0) + 1
        rows.coupons.append(
            db.Coupon(
                id=f"CP-{customer.id}-P{promo_count[customer.id]}",
                customer_id=customer.id,
                kind=db.CouponKind.PROMO,
                amount_won=rng.choice((1000, 2000, 3000, 5000)),
                reason=None,
                order_id=None,
                issued_at=issued_at,
                expires_at=issued_at + valid_days * _DAY,
                used_at=None,
            )
        )


def _bulk_tickets(rng: random.Random, rows: Rows) -> None:
    orders: dict[str, list[db.Order]] = {}
    for order in rows.orders:
        orders.setdefault(order.customer_id, []).append(order)
    count: dict[str, int] = {}
    for _ in range(N_TICKETS):
        customer = rng.choice(rows.customers)
        own = orders.get(customer.id, [])
        order = rng.choice(own) if own and rng.random() < 0.7 else None
        start = order.ordered_at if order is not None else customer.joined_at
        category = rng.choice(list(db.TicketCategory))
        count[customer.id] = count.get(customer.id, 0) + 1
        rows.tickets.append(
            db.Ticket(
                id=f"TK-{customer.id}-{count[customer.id]}",
                customer_id=customer.id,
                order_id=order.id if order is not None else None,
                category=category,
                body=rng.choice(TICKET_BODIES[category]),
                created_at=_between(rng, start, min(start + 3 * _DAY, SEED_END)),
            )
        )


# --- fixtures: the rows tasks refer to (docs/design.md). Hand-written, no random numbers. ---

# (customer_id, name, phone, email, joined_at, ((label, postal_code, address), ...))
_FIXTURE_CUSTOMERS = (
    ("C-9001", "김하준", "01000009001", "hajun.kim@example.com", kst(5, 11, 10, 0),
     (("집", "10391", "가온시 한빛구 가상로 12, 101동 1203호"),)),
    ("C-9002", "이서연", "01000009002", "seoyeon.lee@example.com", kst(5, 12, 10, 0),
     (("집", "20417", "누리시 새별구 새싹길 45, 203동 502호"),)),
    ("C-9003", "박도윤", "01000009003", "doyun.park@example.com", kst(5, 13, 10, 0),
     (("집", "30528", "다솜시 푸른구 너울로 8, 105동 904호"),)),
    ("C-9004", "최지우", "01000009004", "jiwoo.choi@example.com", kst(5, 14, 10, 0),
     (("집", "40639", "라온시 은하구 도담길 77, 102동 301호"),)),
    ("C-9005", "정예준", "01000009005", "yejun.jung@example.com", kst(5, 15, 10, 0),
     (("집", "50741", "마루시 솔내구 미르로 23, 107동 1501호"),
      ("회사", "50852", "마루시 달빛구 소담길 5, 가상빌딩 8층"))),
)  # fmt: skip

# (product_id, name, category, ((option_label, price_won, stock), ...)); variant ids are V-900n-01, -02, ...
_FIXTURE_PRODUCTS = (
    ("P-9001", "무선 이어폰", "전자기기", (("블랙", 38900, 12), ("화이트", 38900, 8))),
    ("P-9002", "전기 주전자", "주방용품", (("화이트", 47300, 10), ("그레이", 47300, 6))),
    ("P-9003", "텀블러", "주방용품", (("실버 / 500ml", 14700, 30), ("네이비 / 500ml", 14700, 20))),
    ("P-9004", "블루투스 스피커", "전자기기", (("블랙", 56800, 7), ("민트", 56800, 5))),
    ("P-9005", "러닝화", "신발",
     (("블랙 / 260mm", 89100, 4), ("블랙 / 270mm", 89100, 6), ("화이트 / 270mm", 89100, 3))),
    ("P-9006", "양말 세트", "의류", (("그레이 / 5켤레", 7300, 50), ("블랙 / 5켤레", 7300, 40))),
)  # fmt: skip

_CARD = db.PaymentMethod.CARD
_SIMPLE_PAY = db.PaymentMethod.SIMPLE_PAY

# order_id: customer, status, ordered_at, address, variant, quantity, payment method, tracking number,
# shipped_at, delivered_at, promised_by
_FIXTURE_ORDERS = (
    ("O-90001", "C-9001", db.OrderStatus.SHIPPED, kst(9, 11, 20, 15), "AD-C-9001-1", "V-9001-01", 1, _CARD,
     "5550-1207-9001", kst(9, 12, 16, 30), None, date(2026, 9, 15)),
    ("O-90002", "C-9002", db.OrderStatus.CANCELLED, kst(9, 10, 21, 40), "AD-C-9002-1", "V-9002-01", 1,
     _SIMPLE_PAY, "5550-1207-9002", None, None, date(2026, 9, 14)),
    ("O-90003", "C-9003", db.OrderStatus.PAID, kst(9, 13, 11, 10), "AD-C-9003-1", "V-9003-01", 2, _CARD,
     "5550-1207-9003", None, None, date(2026, 9, 16)),
    ("O-90004", "C-9004", db.OrderStatus.DELIVERED, kst(8, 30, 19, 25), "AD-C-9004-1", "V-9004-01", 1, _CARD,
     "5550-1207-9004", kst(9, 1, 15, 0), kst(9, 2, 14, 20), date(2026, 9, 3)),
    ("O-90005", "C-9005", db.OrderStatus.PREPARING, kst(9, 12, 8, 50), "AD-C-9005-1", "V-9005-02", 1,
     _SIMPLE_PAY, "5550-1207-9005", None, None, date(2026, 9, 16)),
    ("O-90006", "C-9005", db.OrderStatus.PAID, kst(9, 13, 22, 5), "AD-C-9005-1", "V-9006-01", 3, _CARD,
     "5550-1207-9006", None, None, date(2026, 9, 17)),
)  # fmt: skip
_FIXTURE_CANCELLED_AT = kst(9, 11, 9, 5)  # O-90002


def _fixtures(rows: Rows) -> None:
    addresses: dict[str, db.CustomerAddress] = {}
    for customer_id, name, phone, email, joined_at, places in _FIXTURE_CUSTOMERS:
        rows.customers.append(
            db.Customer(
                id=customer_id,
                name=name,
                phone=phone,
                email=email,
                grade=db.CustomerGrade.NORMAL,
                joined_at=joined_at,
            )
        )
        for n, (label, postal_code, address) in enumerate(places, start=1):
            row = db.CustomerAddress(
                id=f"AD-{customer_id}-{n}",
                customer_id=customer_id,
                label=label,
                recipient=name,
                postal_code=postal_code,
                address=address,
                is_default=n == 1,
            )
            addresses[row.id] = row
            rows.addresses.append(row)

    products: dict[str, db.Product] = {}
    variants: dict[str, db.ProductVariant] = {}
    for product_id, name, category, options in _FIXTURE_PRODUCTS:
        products[product_id] = db.Product(id=product_id, name=name, category=category)
        rows.products.append(products[product_id])
        for n, (label, price, stock) in enumerate(options, start=1):
            variant = db.ProductVariant(
                id=f"V-{product_id[2:]}-{n:02d}",
                product_id=product_id,
                option_label=label,
                price_won=price,
                stock=stock,
            )
            variants[variant.id] = variant
            rows.variants.append(variant)

    for (
        order_id,
        customer_id,
        status,
        ordered_at,
        address_id,
        variant_id,
        quantity,
        method,
        tracking_no,
        shipped_at,
        delivered_at,
        promised_by,
    ) in _FIXTURE_ORDERS:
        cancelled = status == db.OrderStatus.CANCELLED
        variant = variants[variant_id]
        order, items = new_order(
            order_id,
            customer_id,
            status,
            ordered_at,
            addresses[address_id],
            [(products[variant.product_id], variant, quantity)],
            cancelled_at=_FIXTURE_CANCELLED_AT if cancelled else None,
            cancel_reason=db.CancelReason.CHANGED_MIND if cancelled else None,
        )
        rows.orders.append(order)
        rows.items.extend(items)
        rows.payments.append(
            db.Payment(
                order_id=order_id,
                method=method,
                amount_won=order.total_won,
                status=db.PaymentStatus.REFUND_PENDING if cancelled else db.PaymentStatus.PAID,
                refund_won=order.total_won if cancelled else 0,
                paid_at=ordered_at + 2 * _MINUTE,
            )
        )
        if delivered_at is not None:
            ship_status = db.ShipmentStatus.DELIVERED
        elif shipped_at is not None:
            ship_status = db.ShipmentStatus.IN_TRANSIT
        else:
            ship_status = db.ShipmentStatus.READY
        rows.shipments.append(
            db.Shipment(
                order_id=order_id,
                carrier="한빛택배",
                tracking_no=tracking_no,
                status=ship_status,
                shipped_at=shipped_at,
                delivered_at=delivered_at,
                promised_by=promised_by,
            )
        )


@functools.lru_cache(maxsize=1)
def build_seed_engine() -> Engine:
    """The seed database, built once per process. Never write to it: copy it with db.memory_engine(...)."""
    rng = random.Random(SEED)
    rows = Rows()
    _bulk_customers(rng, rows)
    _bulk_products(rng, rows)
    _bulk_orders(rng, rows)
    _bulk_requests(rng, rows)
    _bulk_coupons(rng, rows)
    _bulk_tickets(rng, rows)
    _fixtures(rows)
    # Rows that the dev and test tasks refer to. Imported here because those modules use this one's helpers.
    from support_agent import fixtures_dev, fixtures_test

    fixtures_dev.add_fixtures(rows)
    fixtures_test.add_fixtures(rows)

    engine = db.memory_engine()
    db.Base.metadata.create_all(engine)
    with Session(engine) as session:
        for group in rows.groups():
            session.add_all(group)
            session.flush()
        session.commit()
    return engine
