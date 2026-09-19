"""Hand-written rows that tasks/dev.yaml refers to. Every timestamp is explicit; nothing is random.

One customer per task: C-91NN belongs to dev-0NN. Orders are O-91001..O-91031, tracking numbers are
5551-0914-91NN for order O-910NN. Every timestamp is before 2026-09-14 10:00 KST (the `now` of the tasks),
except the expiry of the one compensation coupon.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import NamedTuple

from support_agent import db
from support_agent.seed import Rows, kst, new_order

# (customer_id, name, phone, email, joined_at, ((label, postal_code, address), ...)); the first address is
# the default one and address ids are AD-{customer_id}-1, -2, ...
_CUSTOMERS = (
    ("C-9101", "고은채", "01000009101", "eunchae.ko@example.com", kst(3, 4, 9, 30),
     (("집", "11846", "가온시 새별구 도담길 58, 104동 702호"),)),
    ("C-9102", "배성훈", "01000009102", "sunghoon.bae@example.com", kst(3, 9, 20, 10),
     (("집", "24375", "누리시 푸른구 하늘길 14, 201동 1104호"),)),
    ("C-9103", "문소희", "01000009103", "sohee.moon@example.com", kst(3, 15, 13, 45),
     (("집", "35192", "다솜시 달빛구 아름로 96, 3층"),
      ("회사", "35240", "다솜시 한빛구 가상로 151, 누리타워 12층"))),
    ("C-9104", "양태호", "01000009104", "taeho.yang@example.com", kst(3, 22, 8, 15),
     (("집", "46718", "라온시 솔내구 너울로 33, 102동 1506호"),)),
    ("C-9105", "허윤서", "01000009105", "yunseo.heo@example.com", kst(4, 2, 18, 40),
     (("집", "38027", "마루시 은하구 소담길 72, 106동 403호"),
      ("회사", "38164", "마루시 한빛구 미르로 210, 새길빌딩 5층"),
      ("부모님 댁", "12593", "가온시 푸른구 새싹길 9"))),
    ("C-9106", "노경민", "01000009106", "kyungmin.noh@example.com", kst(4, 8, 11, 5),
     (("집", "20861", "누리시 은하구 도담길 127, 108동 905호"),)),
    ("C-9107", "안재희", "01000009107", "jaehee.ahn@example.com", kst(4, 13, 22, 20),
     (("집", "31457", "다솜시 솔내구 하늘길 61, 2층"),)),
    ("C-9108", "송민규", "01000009108", "mingyu.song@example.com", kst(4, 19, 10, 50),
     (("집", "42936", "라온시 한빛구 아름로 18, 103동 1201호"),)),
    ("C-9109", "명하윤", "01000009109", "hayoon.myung@example.com", kst(4, 25, 16, 35),
     (("집", "53608", "마루시 새별구 가상로 84, 105동 306호"),)),
    ("C-9110", "백승우", "01000009110", "seungwoo.baek@example.com", kst(5, 1, 7, 55),
     (("집", "14279", "가온시 달빛구 너울로 47, 101동 508호"),
      ("회사", "14402", "가온시 은하구 미르로 5, 한솔센터 3층"))),
    ("C-9111", "차예원", "01000009111", "yewon.cha@example.com", kst(5, 6, 21, 25),
     (("집", "25731", "누리시 솔내구 소담길 39, 107동 1402호"),
      ("회사", "25816", "누리시 한빛구 새싹길 176, 가람빌딩 9층"),
      ("부모님 댁", "27459", "누리시 달빛구 하늘길 203"))),
    ("C-9112", "구본석", "01000009112", "bonseok.koo@example.com", kst(5, 12, 12, 0),
     (("집", "36084", "다솜시 새별구 도담길 25, 102동 801호"),)),
    ("C-9113", "엄다현", "01000009113", "dahyun.eom@example.com", kst(5, 18, 9, 45),
     (("집", "47352", "라온시 푸른구 가상로 119, 109동 204호"),)),
    ("C-9114", "표준영", "01000009114", "junyoung.pyo@example.com", kst(5, 23, 19, 30),
     (("집", "58126", "마루시 달빛구 아름로 66, 1층"),)),
    ("C-9115", "진수빈", "01000009115", "subin.jin@example.com", kst(5, 29, 14, 15),
     (("집", "19643", "가온시 솔내구 미르로 88, 104동 1703호"),)),
    ("C-9116", "변정훈", "01000009116", "junghoon.byun@example.com", kst(6, 3, 8, 40),
     (("집", "21508", "누리시 새별구 너울로 12, 203동 607호"),)),
    ("C-9117", "석가영", "01000009117", "gayoung.seok@example.com", kst(6, 9, 17, 5),
     (("집", "32794", "다솜시 은하구 소담길 140, 101동 1005호"),)),
    ("C-9118", "길태윤", "01000009118", "taeyun.gil@example.com", kst(6, 14, 23, 10),
     (("집", "43861", "라온시 달빛구 새싹길 53, 105동 902호"),
      ("회사", "43917", "라온시 새별구 하늘길 7, 다온프라자 6층"))),
    ("C-9119", "채민서", "01000009119", "minseo.chae@example.com", kst(6, 20, 10, 35),
     (("집", "54237", "마루시 푸른구 도담길 101, 106동 1308호"),)),
    ("C-9120", "하도경", "01000009120", "dokyung.ha@example.com", kst(6, 26, 15, 50),
     (("집", "15983", "가온시 한빛구 아름로 34, 3층"),)),
    ("C-9121", "우서진", "01000009121", "seojin.woo@example.com", kst(7, 1, 11, 20),
     (("집", "26419", "누리시 푸른구 가상로 77, 102동 1601호"),)),
    ("C-9122", "탁현수", "01000009122", "hyunsoo.tak@example.com", kst(7, 7, 20, 45),
     (("집", "37652", "다솜시 한빛구 미르로 29, 108동 503호"),
      ("회사", "37708", "다솜시 새별구 너울로 164, 이음타워 15층"))),
    ("C-9123", "금나래", "01000009123", "narae.keum@example.com", kst(7, 13, 9, 10),
     (("집", "48205", "라온시 은하구 소담길 91, 107동 1109호"),)),
    ("C-9124", "봉세린", "01000009124", "serin.bong@example.com", kst(7, 19, 13, 55),
     (("집", "59371", "마루시 솔내구 새싹길 22, 103동 406호"),)),
)  # fmt: skip

# (product_id, name, category, ((option_label, price_won, stock), ...)); variant ids are V-91nn-01, -02, ...
_PRODUCTS = (
    ("P-9101", "핸드 블렌더", "주방용품", (("화이트", 36400, 11), ("블랙", 36400, 7))),
    ("P-9102", "실리콘 조리도구 세트", "주방용품", (("5종", 21900, 25), ("8종", 29300, 16))),
    ("P-9103", "무선 충전 패드", "전자기기", (("블랙", 23600, 18), ("화이트", 23600, 13))),
    ("P-9104", "에어프라이어", "주방용품", (("3.5L", 79400, 6), ("5.5L", 97600, 4))),
    ("P-9105", "전동 칫솔", "생활용품", (("기본형", 38600, 14), ("살균 케이스 포함", 52300, 9))),
    ("P-9106", "극세사 담요", "생활용품",
     (("그레이 / 싱글", 26400, 20), ("그레이 / 퀸", 34800, 12), ("베이지 / 싱글", 26400, 15))),
    ("P-9107", "메모리폼 베개", "생활용품", (("표준형", 18700, 22), ("경추형", 24900, 17))),
    ("P-9108", "캠핑 의자", "스포츠", (("카키", 43900, 8), ("블랙", 43900, 10))),
    ("P-9109", "니트 가디건", "의류",
     (("아이보리 / M", 48700, 6), ("아이보리 / L", 48700, 5), ("차콜 / M", 48700, 7))),
    ("P-9110", "면 양말 5켤레", "의류", (("화이트", 9800, 40), ("블랙", 9800, 35))),
    ("P-9111", "기모 후드티", "의류",
     (("차콜 / L", 33300, 9), ("차콜 / XL", 33300, 8), ("네이비 / L", 33300, 6))),
    ("P-9112", "전기 그릴", "주방용품", (("1~2인용", 64500, 5), ("3~4인용", 88200, 3))),
    ("P-9113", "트레킹화", "신발",
     (("그레이 / 250mm", 72800, 4), ("그레이 / 255mm", 72800, 2), ("그레이 / 260mm", 72800, 5))),
    ("P-9114", "견과류 선물 세트", "식품", (("30입", 42600, 30), ("50입", 61200, 18))),
    ("P-9115", "미니 가습기", "생활용품", (("화이트", 27300, 16), ("민트", 27300, 9))),
    ("P-9116", "스팀 다리미", "생활용품", (("핸디형", 46900, 12), ("스탠드형", 83500, 4))),
    ("P-9117", "원목 선반", "가구", (("2단", 54100, 7), ("3단", 68700, 5))),
    ("P-9118", "방수 등산 재킷", "의류",
     (("블랙 / 95", 85600, 6), ("블랙 / 100", 85600, 8), ("카키 / 100", 85600, 3))),
    ("P-9119", "슬림핏 청바지", "의류", (("28인치", 39900, 7), ("30인치", 39900, 10), ("32인치", 39900, 6))),
    ("P-9120", "기계식 키보드", "전자기기", (("적축", 74300, 9), ("청축", 74300, 5))),
    ("P-9121", "아기 물티슈 10팩", "생활용품", (("캡형", 19400, 60), ("리필형", 16800, 45))),
    ("P-9122", "크로스백", "가방", (("브라운", 57600, 5), ("블랙", 57600, 8))),
    ("P-9123", "토트백", "가방", (("브라운", 57600, 6), ("아이보리", 57600, 4))),
    ("P-9124", "드립 커피 세트", "식품", (("20입", 24700, 28), ("40입", 44100, 19))),
    ("P-9125", "블루투스 체중계", "전자기기", (("화이트", 31800, 10), ("블랙", 31800, 12))),
    ("P-9126", "LED 무드등", "생활용품", (("원형", 22300, 15), ("사각", 22300, 11))),
    ("P-9127", "유리 밀폐용기", "주방용품", (("4개입", 17800, 24), ("8개입", 31500, 13))),
    ("P-9128", "전기요", "생활용품", (("싱글", 58900, 8), ("더블", 76300, 6))),
    ("P-9129", "경량 우산", "생활용품", (("네이비", 15600, 14), ("그레이", 15600, 9), ("버건디", 15600, 7))),
)  # fmt: skip

_CARD = db.PaymentMethod.CARD
_BANK = db.PaymentMethod.BANK_TRANSFER
_SIMPLE = db.PaymentMethod.SIMPLE_PAY
_PAID = db.OrderStatus.PAID
_PREPARING = db.OrderStatus.PREPARING
_SHIPPED = db.OrderStatus.SHIPPED
_DELIVERED = db.OrderStatus.DELIVERED
_CANCELLED = db.OrderStatus.CANCELLED
_CHANGED_MIND = db.CancelReason.CHANGED_MIND


class _Order(NamedTuple):
    order_id: str
    customer_id: str
    status: db.OrderStatus
    ordered_at: datetime
    lines: tuple[tuple[str, int], ...]  # (variant_id, quantity); line numbers follow this order
    method: db.PaymentMethod
    carrier: str
    promised_by: date
    shipped_at: datetime | None = None
    delivered_at: datetime | None = None
    cancelled_at: datetime | None = None
    cancel_reason: db.CancelReason | None = None
    refunded: bool = False  # cancelled orders only: the refund has already been paid out
    address_no: int = 1


def _d(month: int, day: int) -> date:
    return date(2026, month, day)


_ORDERS = (
    # dev-001: cancelled two days ago, refund of 58,300 won still pending.
    _Order("O-91001", "C-9101", _CANCELLED, kst(9, 11, 19, 45), (("V-9101-01", 1), ("V-9102-01", 1)), _BANK,
           "나래택배", _d(9, 15), cancelled_at=kst(9, 12, 15, 30), cancel_reason=_CHANGED_MIND),
    # dev-002: in transit and not late.
    _Order("O-91002", "C-9102", _SHIPPED, kst(9, 10, 13, 5), (("V-9103-02", 1),), _CARD,
           "새벽택배", _d(9, 15), shipped_at=kst(9, 11, 17, 20)),
    # dev-003: three orders; the customer asks about the air fryer bought in August.
    _Order("O-91003", "C-9103", _DELIVERED, kst(8, 12, 9, 40), (("V-9104-02", 1),), _CARD,
           "두루택배", _d(8, 15), shipped_at=kst(8, 13, 14, 0), delivered_at=kst(8, 14, 16, 25)),
    _Order("O-91004", "C-9103", _DELIVERED, kst(9, 2, 22, 15), (("V-9102-02", 1),), _SIMPLE,
           "한빛택배", _d(9, 5), shipped_at=kst(9, 3, 15, 10), delivered_at=kst(9, 4, 12, 50)),
    _Order("O-91005", "C-9103", _PAID, kst(9, 13, 18, 30), (("V-9103-01", 2),), _CARD,
           "한빛택배", _d(9, 16)),
    # dev-004: owns the basic toothbrush; once ordered and cancelled the option he now asks about.
    _Order("O-91006", "C-9104", _DELIVERED, kst(8, 16, 11, 20), (("V-9105-01", 1),), _CARD,
           "나래택배", _d(8, 19), shipped_at=kst(8, 17, 13, 0), delivered_at=kst(8, 18, 15, 45)),
    _Order("O-91007", "C-9104", _CANCELLED, kst(8, 25, 10, 10), (("V-9105-02", 1),), _CARD,
           "나래택배", _d(8, 28), cancelled_at=kst(8, 25, 10, 40),
           cancel_reason=db.CancelReason.ORDERED_BY_MISTAKE, refunded=True),
    # dev-005: an address book question; the order is only background.
    _Order("O-91008", "C-9105", _DELIVERED, kst(8, 28, 20, 5), (("V-9103-01", 1),), _SIMPLE,
           "새벽택배", _d(8, 31), shipped_at=kst(8, 29, 16, 30), delivered_at=kst(8, 30, 11, 30)),
    # dev-006: the pillows (line 2) already have a return request, see _REQUESTS.
    _Order("O-91009", "C-9106", _DELIVERED, kst(9, 3, 8, 55), (("V-9106-02", 1), ("V-9107-01", 2)), _CARD,
           "두루택배", _d(9, 6), shipped_at=kst(9, 4, 10, 30), delivered_at=kst(9, 5, 14, 10)),
    # dev-007: still being prepared, not late yet.
    _Order("O-91010", "C-9107", _PREPARING, kst(9, 9, 21, 30), (("V-9108-01", 1),), _SIMPLE,
           "한빛택배", _d(9, 15)),
    # dev-008: three lines, lines 1 and 3 go back in one request.
    _Order("O-91011", "C-9108", _DELIVERED, kst(9, 8, 12, 40),
           (("V-9109-01", 1), ("V-9110-01", 1), ("V-9111-01", 2)), _CARD,
           "나래택배", _d(9, 11), shipped_at=kst(9, 9, 16, 0), delivered_at=kst(9, 10, 13, 20)),
    # dev-009: delivered on 9/7, so 9/14 is the last day of the return window.
    _Order("O-91012", "C-9109", _DELIVERED, kst(9, 4, 23, 10), (("V-9112-01", 1),), _SIMPLE,
           "새벽택배", _d(9, 8), shipped_at=kst(9, 5, 15, 30), delivered_at=kst(9, 7, 18, 40)),
    # dev-010: size exchange inside the window.
    _Order("O-91013", "C-9110", _DELIVERED, kst(9, 9, 7, 50), (("V-9113-01", 1),), _CARD,
           "두루택배", _d(9, 12), shipped_at=kst(9, 10, 11, 0), delivered_at=kst(9, 11, 15, 5)),
    # dev-011: paid yesterday, the address can still change.
    _Order("O-91014", "C-9111", _PAID, kst(9, 13, 20, 10), (("V-9114-02", 1),), _CARD,
           "한빛택배", _d(9, 17)),
    # dev-012: promised 9/11, delivered 9/12: one day late.
    _Order("O-91015", "C-9112", _DELIVERED, kst(9, 8, 10, 25), (("V-9115-02", 1),), _BANK,
           "나래택배", _d(9, 11), shipped_at=kst(9, 10, 18, 45), delivered_at=kst(9, 12, 13, 30)),
    # dev-013: cancelled on 9/1 and marked refunded, but the customer has not seen the money.
    _Order("O-91016", "C-9113", _CANCELLED, kst(8, 31, 14, 20), (("V-9116-01", 1),), _CARD,
           "새벽택배", _d(9, 3), cancelled_at=kst(9, 1, 9, 15), cancel_reason=_CHANGED_MIND,
           refunded=True),
    # dev-014: nothing wrong with the order; the customer wants a human.
    _Order("O-91017", "C-9114", _DELIVERED, kst(9, 6, 16, 35), (("V-9117-02", 1),), _CARD,
           "두루택배", _d(9, 10), shipped_at=kst(9, 8, 9, 0), delivered_at=kst(9, 9, 19, 10)),
    # dev-015: delivered on 9/6, the window ended on 9/13 (one day before `now`).
    _Order("O-91018", "C-9115", _DELIVERED, kst(9, 3, 19, 0), (("V-9118-02", 1),), _CARD,
           "한빛택배", _d(9, 7), shipped_at=kst(9, 4, 17, 30), delivered_at=kst(9, 6, 11, 20)),
    # dev-016: delivered on 8/31, the exchange window ended on 9/7.
    _Order("O-91019", "C-9116", _DELIVERED, kst(8, 28, 13, 15), (("V-9119-02", 1),), _SIMPLE,
           "나래택배", _d(9, 1), shipped_at=kst(8, 29, 16, 40), delivered_at=kst(8, 31, 10, 5)),
    # dev-017: shipped yesterday, so it cannot be cancelled.
    _Order("O-91020", "C-9117", _SHIPPED, kst(9, 12, 11, 45), (("V-9120-01", 1),), _CARD,
           "새벽택배", _d(9, 16), shipped_at=kst(9, 13, 9, 10)),
    # dev-018: shipped, so the address cannot change.
    _Order("O-91021", "C-9118", _SHIPPED, kst(9, 11, 9, 30), (("V-9121-01", 2),), _SIMPLE,
           "두루택배", _d(9, 15), shipped_at=kst(9, 12, 14, 50)),
    # dev-019: inside the window, but the customer wants a different product.
    _Order("O-91022", "C-9119", _DELIVERED, kst(9, 7, 21, 50), (("V-9122-01", 1),), _CARD,
           "한빛택배", _d(9, 11), shipped_at=kst(9, 8, 15, 20), delivered_at=kst(9, 10, 12, 15)),
    # dev-020: promised 9/6, delivered 9/8; the delay coupon was already issued, see _COUPONS.
    _Order("O-91023", "C-9120", _DELIVERED, kst(9, 3, 10, 0), (("V-9124-02", 1),), _CARD,
           "나래택배", _d(9, 6), shipped_at=kst(9, 6, 19, 0), delivered_at=kst(9, 8, 14, 35)),
    # dev-021: a defective exchange is already open on O-91024 (see _REQUESTS); O-91025 is a mistaken order.
    _Order("O-91024", "C-9121", _DELIVERED, kst(9, 6, 9, 20), (("V-9125-01", 1),), _SIMPLE,
           "새벽택배", _d(9, 10), shipped_at=kst(9, 7, 14, 15), delivered_at=kst(9, 9, 16, 0)),
    _Order("O-91025", "C-9121", _PAID, kst(9, 13, 23, 5), (("V-9126-01", 1),), _CARD,
           "새벽택배", _d(9, 17)),
    # dev-022: line 2 (two sets) arrived broken; O-91027 should go to the office instead.
    _Order("O-91026", "C-9122", _DELIVERED, kst(9, 7, 12, 0), (("V-9102-01", 1), ("V-9127-01", 2)), _CARD,
           "두루택배", _d(9, 10), shipped_at=kst(9, 8, 13, 40), delivered_at=kst(9, 9, 17, 45)),
    _Order("O-91027", "C-9122", _PREPARING, kst(9, 12, 19, 15), (("V-9108-02", 1),), _CARD,
           "두루택배", _d(9, 16)),
    # dev-023: promised 9/10 and still in transit on 9/14: four days late. O-91029 is a product question.
    _Order("O-91028", "C-9123", _SHIPPED, kst(9, 7, 15, 0), (("V-9107-02", 2),), _BANK,
           "한빛택배", _d(9, 10), shipped_at=kst(9, 8, 18, 20)),
    _Order("O-91029", "C-9123", _DELIVERED, kst(9, 1, 10, 45), (("V-9128-02", 1),), _CARD,
           "나래택배", _d(9, 4), shipped_at=kst(9, 2, 11, 30), delivered_at=kst(9, 3, 15, 30)),
    # dev-024: two bent umbrellas to exchange; then a price adjustment request that ends with a hand-off.
    _Order("O-91030", "C-9124", _DELIVERED, kst(9, 8, 8, 10), (("V-9129-01", 2),), _SIMPLE,
           "새벽택배", _d(9, 11), shipped_at=kst(9, 9, 12, 30), delivered_at=kst(9, 10, 14, 40)),
    _Order("O-91031", "C-9124", _DELIVERED, kst(9, 10, 22, 30), (("V-9101-02", 1),), _SIMPLE,
           "새벽택배", _d(9, 13), shipped_at=kst(9, 11, 16, 0), delivered_at=kst(9, 12, 12, 20)),
)  # fmt: skip

# (order_id, kind, reason, line_nos, exchange_variant_id, created_at)
_REQUESTS = (
    ("O-91009", db.RequestKind.RETURN, db.RequestReason.CHANGED_MIND, (2,), None, kst(9, 8, 20, 15)),
    ("O-91024", db.RequestKind.EXCHANGE, db.RequestReason.DEFECTIVE, (1,), "V-9125-02", kst(9, 11, 10, 30)),
)

# (order_id, reason, amount_won, issued_at): compensation coupons that already exist
_COUPONS = (("O-91023", db.CompensationReason.DELIVERY_DELAY, 2000, kst(9, 9, 14, 5)),)

_RETURN_FEE_WON = 3000  # changed_mind only


def add_fixtures(rows: Rows) -> None:
    """Append customers, addresses, products, orders and the rest to `rows`."""
    addresses: dict[str, db.CustomerAddress] = {}
    for customer_id, name, phone, email, joined_at, places in _CUSTOMERS:
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
    for product_id, name, category, options in _PRODUCTS:
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

    orders: dict[str, db.Order] = {}
    items: dict[tuple[str, int], db.OrderItem] = {}
    for spec in _ORDERS:
        cancelled = spec.status == db.OrderStatus.CANCELLED
        lines = [(products[variants[v].product_id], variants[v], qty) for v, qty in spec.lines]
        order, order_items = new_order(
            spec.order_id,
            spec.customer_id,
            spec.status,
            spec.ordered_at,
            addresses[f"AD-{spec.customer_id}-{spec.address_no}"],
            lines,
            cancelled_at=spec.cancelled_at,
            cancel_reason=spec.cancel_reason,
        )
        orders[order.id] = order
        rows.orders.append(order)
        rows.items.extend(order_items)
        items.update({(order.id, item.line_no): item for item in order_items})

        if not cancelled:
            pay_status = db.PaymentStatus.PAID
        elif spec.refunded:
            pay_status = db.PaymentStatus.REFUNDED
        else:
            pay_status = db.PaymentStatus.REFUND_PENDING
        rows.payments.append(
            db.Payment(
                order_id=order.id,
                method=spec.method,
                amount_won=order.total_won,
                status=pay_status,
                refund_won=order.total_won if cancelled else 0,
                paid_at=spec.ordered_at + timedelta(minutes=2),
            )
        )

        if spec.delivered_at is not None:
            ship_status = db.ShipmentStatus.DELIVERED
        elif spec.shipped_at is not None:
            ship_status = db.ShipmentStatus.IN_TRANSIT
        else:
            ship_status = db.ShipmentStatus.READY
        rows.shipments.append(
            db.Shipment(
                order_id=order.id,
                carrier=spec.carrier,
                tracking_no=f"5551-0914-91{order.id[-2:]}",
                status=ship_status,
                shipped_at=spec.shipped_at,
                delivered_at=spec.delivered_at,
                promised_by=spec.promised_by,
            )
        )

    for order_id, kind, reason, line_nos, exchange_variant_id, created_at in _REQUESTS:
        chosen = [items[(order_id, line_no)] for line_no in line_nos]
        fee = _RETURN_FEE_WON if reason == db.RequestReason.CHANGED_MIND else 0
        if kind == db.RequestKind.RETURN:
            request_id = f"RT-{order_id}-{min(line_nos)}"
            refund = sum(item.unit_price_won * item.quantity for item in chosen) - fee
            item_status = db.ItemStatus.RETURN_REQUESTED
        else:
            request_id = f"EX-{order_id}-{line_nos[0]}"
            refund = 0
            item_status = db.ItemStatus.EXCHANGE_REQUESTED
        rows.requests.append(
            db.ServiceRequest(
                id=request_id,
                order_id=order_id,
                kind=kind,
                reason=reason,
                refund_won=refund,
                return_fee_won=fee,
                created_at=created_at,
            )
        )
        for item in chosen:
            item.status = item_status
            rows.request_items.append(
                db.ServiceRequestItem(
                    request_id=request_id,
                    line_no=item.line_no,
                    quantity=item.quantity,
                    exchange_variant_id=exchange_variant_id,
                )
            )

    for order_id, reason, amount_won, issued_at in _COUPONS:
        rows.coupons.append(
            db.Coupon(
                id=f"CP-{order_id}-1",
                customer_id=orders[order_id].customer_id,
                kind=db.CouponKind.COMPENSATION,
                amount_won=amount_won,
                reason=reason,
                order_id=order_id,
                issued_at=issued_at,
                expires_at=issued_at + timedelta(days=30),
                used_at=None,
            )
        )
