"""Korean display labels of the enum codes. The DB and tool arguments use the ASCII codes; tool outputs add
the label next to the code (`"status": "delivered", "status_label": "배송완료"`)."""

from __future__ import annotations

import enum

from support_agent import db

LABELS: dict[type[enum.Enum], dict[str, str]] = {
    db.CustomerGrade: {"normal": "일반", "vip": "VIP"},
    db.OrderStatus: {
        "paid": "결제완료",
        "preparing": "배송준비중",
        "shipped": "배송중",
        "delivered": "배송완료",
        "cancelled": "취소됨",
    },
    db.ItemStatus: {
        "ordered": "주문됨",
        "cancelled": "취소됨",
        "return_requested": "반품접수",
        "exchange_requested": "교환접수",
    },
    db.PaymentMethod: {"card": "신용카드", "bank_transfer": "계좌이체", "simple_pay": "간편결제"},
    db.PaymentStatus: {"paid": "결제완료", "refund_pending": "환불진행중", "refunded": "환불완료"},
    db.ShipmentStatus: {"ready": "배송준비", "in_transit": "배송중", "delivered": "배송완료"},
    db.CancelReason: {
        "changed_mind": "단순 변심",
        "ordered_by_mistake": "주문 실수",
        "delivery_too_slow": "배송 지연",
    },
    db.RequestKind: {"return": "반품", "exchange": "교환"},
    db.RequestReason: {"changed_mind": "단순 변심", "defective": "상품 불량", "wrong_item": "오배송"},
    db.CouponKind: {"promo": "프로모션", "compensation": "보상"},
    db.CompensationReason: {"delivery_delay": "배송 지연", "defective_item": "상품 불량"},
    db.TicketCategory: {
        "delivery": "배송",
        "refund": "환불",
        "product": "상품",
        "account": "계정",
        "other": "기타",
    },
    db.HandoffReason: {
        "customer_request": "고객 요청",
        "out_of_scope": "업무 범위 밖",
        "cannot_verify": "본인 확인 불가",
    },
}


def label(member: enum.Enum) -> str:
    return LABELS[type(member)][member.value]


def choices(enum_cls: type[enum.Enum]) -> str:
    """`changed_mind=단순 변심, defective=상품 불량` for argument descriptions."""
    return ", ".join(f"{code}={text}" for code, text in LABELS[enum_cls].items())
