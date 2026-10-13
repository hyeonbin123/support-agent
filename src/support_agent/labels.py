"""Korean display labels of the enum codes. The DB and tool arguments use the ASCII codes; tool outputs add
the label next to the code (`"status": "delivered", "status_label": "배송완료"`)."""

# [파일 역할] DB enum 코드(ASCII)와 한국어 표시 이름의 대응표다.
# [설계] DB와 도구 인자는 ASCII 코드만 쓰고, 도구 출력에는 코드 옆에 _label을 함께 넣으며, enum 인자 설명에는
# '코드=한국어' 목록을 적는다(docs/design.md 도구 절). 코드는 DB 저장과 정답 비교에 쓰는 값이고, 표시 이름은
# 모델이 고객에게 한국어로 옮겨 말할 값이다.
# [연결] tools.py가 label()로 출력의 _label 값을, choices()로 enum 인자 설명을 만든다. tests/test_tools.py도
# choices를 쓴다.

from __future__ import annotations

import enum

# enum 클래스(주문 상태, 반품 사유 등)는 db.py에 정의돼 있고, 여기서는 그 클래스를 키로 써서 표시 이름만
# 붙인다.
from support_agent import db

# enum 클래스 → {코드: 한국어}. 표시 이름은 모델이 고객에게 그대로 옮겨 말할 수 있는 짧은 한국어로 둔다. 새
# enum 값을 db.py에 더하면 여기에도 더해야 한다(없으면 label()이 KeyError를 낸다).
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
        "delivery_too_slow": "배송이 너무 늦어서",
    },
    db.RequestKind: {"return": "반품", "exchange": "교환"},
    db.RequestReason: {"changed_mind": "단순 변심", "defective": "상품 불량", "wrong_item": "오배송"},
    db.CouponKind: {"promo": "프로모션", "compensation": "보상"},
    db.CompensationReason: {"delivery_delay": "배송 지연 보상", "defective_item": "상품 불량 보상"},
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


# enum 멤버 하나의 한국어 표시 이름. 도구 출력의 status_label, reason_label 등에 쓴다.
def label(member: enum.Enum) -> str:
    return LABELS[type(member)][member.value]


# 인자 설명용 '코드=한국어' 목록 문자열. 모델이 한국어 사유를 듣고 맞는 ASCII 코드를 고르게 돕는다.
def choices(enum_cls: type[enum.Enum]) -> str:
    """`changed_mind=단순 변심, defective=상품 불량` for argument descriptions."""
    return ", ".join(f"{code}={text}" for code, text in LABELS[enum_cls].items())
