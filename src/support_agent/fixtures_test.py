"""Hand-written rows that tasks/test.yaml refers to. Every timestamp is explicit; nothing is random.

Ids: customers C-9201.. (one per task: test-0NN talks to C-92NN), addresses AD-C-92NN-n, products P-9201..,
variants V-92pp-nn, orders O-92001... Every timestamp is before 2026-09-14 10:00 KST, the usual `now` of the
test tasks. The reasoning behind each row (dates, amounts, the rule it probes) is in the `notes` of the task.
"""

# [파일 역할] 시험용 과제 40개(tasks/test.yaml)가 가리키는 고정 행이다. 고객 40명(C-9201~C-9240, test-0NN은
# C-92NN과 대화), 상품 20개(P-9201~), 주문 54건(O-92001~O-92054), 이미 있는 반품·교환 3건, 보상 쿠폰 5장, 티켓
# 1건이다.
# [설계 이유] 개발용(fixtures_dev.py)과 고객·주문을 공유하지 않고, 같은 종류의 규정을 다른 금액·날짜·상품으로
# 다룬다. 규정 검사 코드 중 return_not_delivered, coupon_not_eligible, coupon_limit_exceeded,
# coupon_order_cancelled는 시험용에만 있다(docs/experiments.md 1단계 과제 묶음). 각 행의 날짜·금액이 어느
# 규정을 시험하는지는 과제의 notes에 적혀 있다(docstring).
# [주의] 시험용은 단계가 끝날 때 고른 구성만 한 번 재며, 시험용 에피소드를 보면서 프롬프트·가드를 고치지
# 않는다. 측정을 시작한 뒤 이 행을 고치면 seed 해시가 바뀐다.
# [연결] seed.build_seed_engine이 add_fixtures를 부른다. 주소는 seed.py의 가상 지명(CITIES 등)으로 만들고,
# 주문은 seed.new_order로 만든다.

from __future__ import annotations

# frozen dataclass로 주문 명세를 두어 기본값(결제 수단 카드, 택배사 한빛택배, 첫 주소)을 생략하고 짧게 쓴다.
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from support_agent import db
from support_agent.seed import CITIES, DISTRICTS, ROADS, Rows, kst, new_order

# 아래 표를 짧게 쓰기 위한 enum 별칭들.
_CARD = db.PaymentMethod.CARD
_BANK = db.PaymentMethod.BANK_TRANSFER
_SIMPLE = db.PaymentMethod.SIMPLE_PAY

_PAID = db.OrderStatus.PAID
_PREPARING = db.OrderStatus.PREPARING
_SHIPPED = db.OrderStatus.SHIPPED
_DELIVERED = db.OrderStatus.DELIVERED
_CANCELLED = db.OrderStatus.CANCELLED

_RETURN = db.RequestKind.RETURN
_EXCHANGE = db.RequestKind.EXCHANGE
_DELAY = db.CompensationReason.DELIVERY_DELAY
_DEFECT = db.CompensationReason.DEFECTIVE_ITEM

# 시험용 고객. 번호 N이 고객 C-92NN, 전화 0100000-92NN이 된다. 배경 고객 생성에 쓰는 성(seed.SURNAMES)은 쓰지
# 않는다.
# (number, name, e-mail local part): customer C-92NN, phone 0100000-92NN. No surname of the generated rows.
_CUSTOMERS = (
    (1, "남궁현", "hyun.namgung"),
    (2, "문태오", "taeo.moon"),
    (3, "백승아", "seunga.baek"),
    (4, "류하린", "harin.ryu"),
    (5, "설지후", "jihu.seol"),
    (6, "차은결", "eungyeol.cha"),
    (7, "표민서", "minseo.pyo"),
    (8, "함도경", "dokyung.ham"),
    (9, "공서진", "seojin.kong"),
    (10, "길라희", "rahee.gil"),
    (11, "노은채", "eunchae.noh"),
    (12, "도재윤", "jaeyun.do"),
    (13, "마예서", "yeseo.ma"),
    (14, "방시현", "sihyun.bang"),
    (15, "변하람", "haram.byun"),
    (16, "석주하", "juha.seok"),
    (17, "선다인", "dain.sun"),
    (18, "소이준", "ijun.so"),
    (19, "안로운", "roun.ahn"),
    (20, "양세린", "serin.yang"),
    (21, "엄태윤", "taeyun.eom"),
    (22, "여민재", "minjae.yeo"),
    (23, "염규리", "gyuri.yeom"),
    (24, "우서하", "seoha.woo"),
    (25, "원지한", "jihan.won"),
    (26, "유채아", "chaea.yoo"),
    (27, "은시후", "sihu.eun"),
    (28, "인하율", "hayul.in"),
    (29, "전루아", "rua.jeon"),
    (30, "제민혁", "minhyuk.je"),
    (31, "주예담", "yedam.joo"),
    (32, "지온유", "onyu.ji"),
    (33, "진서우", "seowoo.jin"),
    (34, "천가람", "garam.cheon"),
    (35, "추이든", "eden.chu"),
    (36, "탁윤슬", "yunseul.tak"),
    (37, "편도하", "doha.pyeon"),
    (38, "하새봄", "saebom.ha"),
    (39, "허단우", "danu.heo"),
    (40, "홍리안", "rian.hong"),
)

# 추가 주소록. 첫 주소(집)는 _home이 고객 번호로 만들고, 여기 있는 고객만 회사·부모님 댁 주소가 더 있다.
# 수령인이 빈 문자열이면 고객 본인이다. 주소 조회(test-005)와 배송지 변경 과제가 이 주소를 쓴다.
# Address 1 of every customer is '집' (derived from the customer number). Further addresses, in order:
# customer number -> ((label, recipient or "" for the customer, postal_code, address), ...)
_EXTRA_ADDRESSES: dict[int, tuple[tuple[str, str, str, str], ...]] = {
    5: (("회사", "", "40925", "라온시 달빛구 아름로 88, 가상타워 5층"),),
    12: (("회사", "", "51364", "마루시 은하구 미르로 140, 가상센터 3층"),),
    16: (
        ("회사", "", "18452", "가온시 푸른구 도담길 9, 가상프라자 11층"),
        ("부모님 댁", "석경호", "21763", "누리시 솔내구 하늘길 31, 104동 702호"),
    ),
    26: (("회사", "", "62208", "다솜시 새별구 소담길 64, 가상빌딩 2층"),),
    33: (("회사", "", "30871", "다솜시 한빛구 너울로 117, 가상타워 9층"),),
    36: (("부모님 댁", "탁명수", "74519", "라온시 솔내구 새싹길 52, 102동 1401호"),),
    40: (("회사", "", "83146", "마루시 푸른구 가상로 201, 가상센터 6층"),),
}

# 시험용 상품. 개발용·배경 상품과 따로 두어 재고가 섞이지 않는다. 차콜 / M은 재고 0으로 두어 재입고 문의
# 과제(test-036)에 쓴다.
# (product_id, name, category, ((option_label, price_won, stock), ...)); variant ids are V-92pp-01, -02, ...
_PRODUCTS = (
    ("P-9201", "가습기", "생활가전", (("화이트", 43700, 9), ("민트", 43700, 5))),
    ("P-9202", "핸드 블렌더", "주방가전", (("실버", 52600, 7), ("블랙", 52600, 4))),
    # 차콜 / M is sold out on purpose: test-036 asks when it comes back.
    ("P-9203", "니트 가디건", "의류",
     (("베이지 / M", 46200, 6), ("베이지 / L", 46200, 8), ("차콜 / M", 46200, 0))),
    ("P-9204", "등산 스틱", "스포츠", (("블랙", 33400, 10), ("블루", 33400, 6))),
    ("P-9205", "캠핑 의자", "스포츠", (("카키", 27800, 12), ("샌드", 27800, 9))),
    ("P-9206", "도자기 접시 세트", "주방용품", (("4p", 21300, 14), ("6p", 29600, 7))),
    ("P-9207", "욕실 매트", "생활용품", (("그레이", 12400, 25), ("아이보리", 12400, 18))),
    ("P-9208", "워킹화", "신발",
     (("그레이 / 240mm", 67900, 5), ("그레이 / 250mm", 67900, 7), ("네이비 / 250mm", 69400, 11))),
    ("P-9209", "노트북 파우치", "가방", (("13인치", 18700, 16), ("15인치", 21900, 13))),
    ("P-9210", "아로마 디퓨저", "생활용품", (("라벤더", 16300, 20), ("시트러스", 16300, 15))),
    ("P-9211", "전동 칫솔", "생활가전", (("화이트", 58400, 8), ("핑크", 58400, 6))),
    ("P-9212", "차량용 거치대", "자동차용품", (("송풍구형", 14600, 22), ("대시보드형", 17200, 17))),
    ("P-9213", "극세사 이불", "침구", (("싱글", 39300, 9), ("퀸", 48700, 6))),
    ("P-9214", "스테인리스 냄비", "주방용품", (("18cm", 31500, 10), ("22cm", 37800, 8))),
    ("P-9215", "우산", "잡화", (("네이비", 11700, 30), ("버건디", 11700, 24))),
    ("P-9216", "데스크 매트", "사무용품", (("그레이", 13900, 19), ("브라운", 13900, 14))),
    ("P-9217", "무선 충전 패드", "전자기기", (("블랙", 24300, 13), ("화이트", 24300, 9))),
    ("P-9218", "레인 부츠", "신발",
     (("옐로 / 240mm", 35600, 4), ("옐로 / 250mm", 35600, 6), ("카키 / 250mm", 35600, 5))),
    ("P-9219", "에어프라이어", "주방가전", (("3L", 64200, 7), ("5L", 83500, 5))),
    ("P-9220", "필라테스 링", "스포츠", (("핑크", 9400, 21), ("퍼플", 9400, 16))),
)  # fmt: skip


# 고정 주문 한 건의 명세. customer는 고객 번호(9 → C-9209), lines는 (옵션 id, 수량)이고 순서가 줄 번호다.
# refunded는 이미 환불이 끝난 취소 주문, address는 주소록의 몇 번째 주소인지다.
@dataclass(frozen=True)
class _Order:
    order_id: str
    customer: int  # customer number: 9 -> C-9209
    status: db.OrderStatus
    ordered_at: datetime
    lines: tuple[tuple[str, int], ...]  # (variant_id, quantity), line_no = position
    promised_by: date
    method: db.PaymentMethod = _CARD
    carrier: str = "한빛택배"
    shipped_at: datetime | None = None
    delivered_at: datetime | None = None
    cancelled_at: datetime | None = None
    cancel_reason: db.CancelReason | None = None
    refunded: bool = False  # a cancelled order whose refund is already paid out
    address: int = 1  # n of AD-C-92NN-n


# 2026년 날짜를 짧게 쓰는 도우미(도착 예정일용).
def _d(month: int, day: int) -> date:
    return date(2026, month, day)


# 시험용 주문 표. 각 주문 위의 영어 주석은 그 주문을 쓰는 과제와 정답 금액의 계산이고, 한국어 주석은
# tasks/test.yaml의 purpose를 옮긴 것이다. 짝 과제(test-011/023, test-017/031)는 같은 규정의 경계 양쪽을
# 시험한다.
_ORDERS = (
    # test-001 조회: 취소된 두 줄 주문의 환불 금액(27,100원, 환불진행중)과 취소 시각을 알려 주는지.
    # test-001 lookup: cancelled two-line order, 12,400 + 11,700 + 3,000 = 27,100, refund pending
    _Order("O-92001", 1, _CANCELLED, kst(9, 8, 20, 10), (("V-9207-01", 1), ("V-9215-02", 1)), _d(9, 11),
           method=_SIMPLE, cancelled_at=kst(9, 9, 8, 45), cancel_reason=db.CancelReason.CHANGED_MIND),
    # test-002 조회: 배송중 주문의 도착 예정일(9월 16일)과 송장 번호를 알려 주는지.
    # test-002 lookup: in transit, promised 9/16
    _Order("O-92002", 2, _SHIPPED, kst(9, 11, 13, 25), (("V-9214-01", 1),), _d(9, 16), carrier="새벽택배",
           shipped_at=kst(9, 12, 17, 10)),
    # test-003 조회: 배송완료 시각과 택배사를 알려 주는지. 도착 예정일(9월 11일)과 수령일(9월 10일)이 달라
    # 둘을 구별해야 한다.
    # test-003 lookup: delivered 9/10 (promised 9/11, so the two dates differ)
    _Order("O-92003", 3, _DELIVERED, kst(9, 7, 9, 30), (("V-9209-01", 1),), _d(9, 11), carrier="나래택배",
           shipped_at=kst(9, 8, 15, 20), delivered_at=kst(9, 10, 16, 40)),
    # test-004 조회: 주문 번호를 모르는 고객의 주문 세 건 중 8월 주문을 찾아 주문 번호와 결제 금액을 알려
    # 주는지.
    # test-004 lookup: three orders, the August one is 43,700 + 9,400 = 53,100
    _Order("O-92004", 4, _DELIVERED, kst(8, 19, 21, 5), (("V-9201-01", 1), ("V-9220-02", 1)), _d(8, 22),
           method=_SIMPLE, carrier="두루택배", shipped_at=kst(8, 20, 14, 0), delivered_at=kst(8, 21, 13, 15)),
    _Order("O-92005", 4, _DELIVERED, kst(9, 2, 10, 40), (("V-9215-01", 1),), _d(9, 5),
           shipped_at=kst(9, 3, 13, 30), delivered_at=kst(9, 4, 11, 30)),
    _Order("O-92006", 4, _PAID, kst(9, 13, 19, 20), (("V-9216-02", 1),), _d(9, 17)),
    # test-006 조회: 이미 접수된 반품(1번 줄, 단순 변심, _REQUESTS)의 환불 예정 금액과 접수일을 알려 주는지.
    # test-006 lookup: a return (line 1, changed mind) was filed on 9/11: 33,400 - 3,000 = 30,400
    _Order("O-92007", 6, _DELIVERED, kst(9, 5, 12, 15), (("V-9204-01", 1), ("V-9212-01", 1)), _d(9, 8),
           method=_BANK, carrier="새벽택배", shipped_at=kst(9, 6, 16, 45), delivered_at=kst(9, 8, 15, 5)),
    # test-007 조회: 주문에서 상품 번호를 알아낸 뒤 get_product로 다른 옵션의 가격과 재고를 알려 주는지.
    # test-007 lookup: product id and price come from this order, stock from get_product
    _Order("O-92008", 7, _DELIVERED, kst(8, 24, 17, 50), (("V-9208-01", 1),), _d(8, 27), carrier="나래택배",
           shipped_at=kst(8, 25, 15, 0), delivered_at=kst(8, 27, 12, 20)),
    # test-008 조회: 최근 주문 두 건 중 아직 오지 않은 주문을 찾아 도착 예정일과 송장 번호를 알려 주는지.
    # test-008 lookup: one of two recent orders is still on its way
    _Order("O-92009", 8, _DELIVERED, kst(9, 9, 8, 20), (("V-9206-02", 1),), _d(9, 12), carrier="두루택배",
           shipped_at=kst(9, 10, 11, 0), delivered_at=kst(9, 11, 14, 50)),
    _Order("O-92010", 8, _SHIPPED, kst(9, 10, 22, 40), (("V-9217-02", 1),), _d(9, 15), method=_SIMPLE,
           shipped_at=kst(9, 12, 9, 35)),
    # test-009 처리: 배송준비중 주문을 고객이 말한 사유로 취소하고 환불 금액을 알려 주는지.
    # test-009 action: cancel while preparing, 83,500
    _Order("O-92011", 9, _PREPARING, kst(9, 11, 9, 10), (("V-9219-02", 1),), _d(9, 16)),
    # test-010 처리: 주문 번호를 모르는 고객의 결제완료 주문을 목록에서 찾아 취소하는지.
    # test-010 action: the customer does not know the order number; 11,700 x 2 + 3,000 = 26,400
    _Order("O-92012", 10, _DELIVERED, kst(8, 15, 11, 30), (("V-9220-01", 2),), _d(8, 18), carrier="새벽택배",
           shipped_at=kst(8, 16, 15, 10), delivered_at=kst(8, 18, 10, 25)),
    _Order("O-92013", 10, _PAID, kst(9, 13, 23, 10), (("V-9215-01", 2),), _d(9, 17), method=_SIMPLE),
    # test-011 처리: 반품 기한의 마지막 날(9월 7일 수령 → 9월 14일)에 온 단순 변심 반품을 한 줄만 접수하는지.
    # test-023과 짝을 이루는 경계 사례다.
    # test-011 action: received 9/7, so 9/14 is the last day; line 1 only: 48,700 - 3,000 = 45,700
    _Order("O-92014", 11, _DELIVERED, kst(9, 4, 18, 30), (("V-9213-02", 1), ("V-9207-02", 1)), _d(9, 8),
           carrier="나래택배", shipped_at=kst(9, 5, 16, 0), delivered_at=kst(9, 7, 19, 45)),
    # test-012 처리: 같은 주문의 두 줄(1, 3번)을 한 번의 호출로 묶어 반품하는지. 배송비는 한 번만 빠진다.
    # test-012 action: lines 1 and 3 in one call: 46,200 + 35,600 - 3,000 = 78,800
    _Order("O-92015", 12, _DELIVERED, kst(9, 6, 11, 0),
           (("V-9203-01", 1), ("V-9215-01", 1), ("V-9218-01", 1)), _d(9, 10), carrier="두루택배",
           shipped_at=kst(9, 8, 10, 30), delivered_at=kst(9, 10, 12, 30)),
    # test-013 처리: 상품 불량 반품을 배송비 없이, 수량 2개인 줄 전체로 접수하는지.
    # test-013 action: defective, quantity 2: 16,300 x 2 = 32,600, no fee
    _Order("O-92016", 13, _DELIVERED, kst(9, 7, 14, 45), (("V-9210-01", 2),), _d(9, 10), method=_BANK,
           shipped_at=kst(9, 8, 17, 0), delivered_at=kst(9, 10, 10, 10)),
    # test-014 처리: 기한 안의 사이즈 교환(240mm → 250mm)을 같은 상품의 맞는 옵션 번호로 접수하는지.
    # test-014 action: size exchange 240mm -> 250mm
    _Order("O-92017", 14, _DELIVERED, kst(9, 5, 20, 20), (("V-9208-01", 1),), _d(9, 9), carrier="새벽택배",
           shipped_at=kst(9, 7, 9, 40), delivered_at=kst(9, 9, 13, 40)),
    # test-015 처리: 두 줄 주문에서 불량인 2번 줄만 같은 상품의 다른 색으로 교환하는지.
    # test-015 action: defective exchange of line 2 (black -> white)
    _Order("O-92018", 15, _DELIVERED, kst(9, 6, 9, 50), (("V-9216-01", 1), ("V-9217-01", 1)), _d(9, 10),
           method=_SIMPLE, carrier="나래택배", shipped_at=kst(9, 7, 14, 20), delivered_at=kst(9, 9, 17, 25)),
    # test-016 처리: 배송준비중 주문의 배송지를 주소록의 세 번째 주소로 바꾸는지.
    # test-016 action: address change to the third address
    _Order("O-92019", 16, _PREPARING, kst(9, 12, 15, 35), (("V-9213-01", 1),), _d(9, 16)),
    # test-017 처리: 예전 보상 쿠폰 두 장 중 한 장만 최근 30일 안에 있을 때 하루 지연 쿠폰을 발급하는지(한도
    # 2장 안). test-031과 짝이다.
    # test-017 action: coupons of 8/15 (outside the 30 days) and 9/5 (inside); O-92022 came one day late
    _Order("O-92020", 17, _DELIVERED, kst(8, 8, 10, 0), (("V-9212-02", 1),), _d(8, 11), carrier="두루택배",
           shipped_at=kst(8, 10, 11, 0), delivered_at=kst(8, 14, 15, 30)),
    _Order("O-92021", 17, _DELIVERED, kst(8, 30, 19, 40), (("V-9209-02", 1),), _d(9, 2),
           shipped_at=kst(9, 1, 10, 0), delivered_at=kst(9, 4, 14, 10)),
    _Order("O-92022", 17, _DELIVERED, kst(9, 8, 10, 5), (("V-9204-02", 1),), _d(9, 11), carrier="새벽택배",
           shipped_at=kst(9, 10, 16, 30), delivered_at=kst(9, 12, 18, 20)),
    # test-018 처리: 이미 불량 교환이 접수된 주문에 상품 불량 보상 쿠폰을 발급하는지.
    # test-018 action: a defective exchange was filed on 9/10, the coupon was not asked for then
    _Order("O-92023", 18, _DELIVERED, kst(9, 4, 21, 15), (("V-9211-02", 1),), _d(9, 8), carrier="나래택배",
           shipped_at=kst(9, 6, 10, 10), delivered_at=kst(9, 8, 11, 0)),
    # test-019 처리: 환불완료로 표시되지만 입금을 못 받았다는 문의를 환불 분류 티켓으로 남기는지.
    # test-019 action: refund marked as paid out, the customer has not seen the money
    _Order("O-92024", 19, _CANCELLED, kst(9, 1, 13, 0), (("V-9218-02", 1),), _d(9, 4), method=_BANK,
           cancelled_at=kst(9, 2, 9, 30), cancel_reason=db.CancelReason.CHANGED_MIND, refunded=True),
    # test-020 처리: 주문과 무관한 상품 문의를 상품 분류 티켓으로 남기는지. 기존 티켓(_TICKETS)이 있어 새
    # 번호는 TK-C-9220-2다.
    # test-020 action: an old order with an old ticket, so the new ticket is TK-C-9220-2
    _Order("O-92025", 20, _DELIVERED, kst(7, 27, 16, 20), (("V-9207-01", 1),), _d(7, 30), carrier="두루택배",
           shipped_at=kst(7, 28, 13, 0), delivered_at=kst(7, 30, 11, 45)),
    # test-021 처리: 고객이 처음부터 사람 상담원을 분명히 요구할 때 이관하는지.
    # test-021 action (hand-off on request)
    _Order("O-92026", 21, _DELIVERED, kst(9, 8, 12, 40), (("V-9206-01", 1),), _d(9, 11),
           shipped_at=kst(9, 9, 15, 30), delivered_at=kst(9, 11, 13, 5)),
    # test-022 처리: 업무 범위 밖 요청(결제 수단 변경)을 거절한 뒤 고객이 요구하면 이관하는지.
    # test-022 action (hand-off, out of scope: payment method change)
    _Order("O-92027", 22, _PAID, kst(9, 13, 16, 45), (("V-9219-01", 1),), _d(9, 17)),
    # test-023 거절: 반품 기한이 하루 지난(9월 6일 수령 → 9월 13일까지) 단순 변심 반품을 거절하는지.
    # test-011의 짝이다.
    # test-023 refusal: received 9/6, last day was 9/13
    _Order("O-92028", 23, _DELIVERED, kst(9, 3, 10, 30), (("V-9202-01", 1),), _d(9, 7), carrier="새벽택배",
           shipped_at=kst(9, 4, 14, 0), delivered_at=kst(9, 6, 15, 30)),
    # test-024 거절: 아직 배송 중인 주문의 반품을 거절하는지(return_not_delivered, 시험용에만 있는 코드).
    # test-024 refusal: still in transit
    _Order("O-92029", 24, _SHIPPED, kst(9, 11, 19, 0), (("V-9205-02", 2),), _d(9, 16), method=_SIMPLE,
           carrier="나래택배", shipped_at=kst(9, 13, 10, 20)),
    # test-025 거절: 이미 출고된 주문의 취소를 거절하는지.
    # test-025 refusal: shipped, cannot be cancelled
    _Order("O-92030", 25, _SHIPPED, kst(9, 11, 8, 40), (("V-9214-02", 1),), _d(9, 15), carrier="두루택배",
           shipped_at=kst(9, 13, 14, 0)),
    # test-026 거절: 이미 출고된 주문의 배송지 변경을 거절하는지.
    # test-026 refusal: shipped 9/12, address cannot change
    _Order("O-92031", 26, _SHIPPED, kst(9, 10, 12, 0), (("V-9206-02", 1),), _d(9, 15),
           shipped_at=kst(9, 12, 18, 30)),
    # test-027 거절: 기한 안이어도 다른 상품으로의 교환을 거절하는지.
    # test-027 refusal: exchange for another product
    _Order("O-92032", 27, _DELIVERED, kst(9, 7, 8, 55), (("V-9201-02", 1),), _d(9, 10), method=_SIMPLE,
           carrier="새벽택배", shipped_at=kst(9, 8, 13, 40), delivered_at=kst(9, 10, 14, 0)),
    # test-028 거절: 교환 기한이 하루 지난 사이즈 교환을 거절하는지. 이 과제만 now가 9월 16일이다.
    # test-028 refusal (now 9/16): received 9/8, last day was 9/15
    _Order("O-92033", 28, _DELIVERED, kst(9, 5, 13, 15), (("V-9218-01", 1),), _d(9, 9), carrier="나래택배",
           shipped_at=kst(9, 7, 11, 30), delivered_at=kst(9, 8, 20, 10)),
    # test-029 거절: 도착 예정일보다 먼저 받은 주문의 지연 쿠폰 요청을 거절하는지(coupon_not_eligible,
    # 시험용에만 있는 코드).
    # test-029 refusal: arrived a day before the promised date
    _Order("O-92034", 29, _DELIVERED, kst(9, 7, 22, 15), (("V-9212-01", 2),), _d(9, 11), carrier="두루택배",
           shipped_at=kst(9, 9, 9, 0), delivered_at=kst(9, 10, 9, 50)),
    # test-030 거절: 이미 보상 쿠폰이 발급된 주문에 또 발급하지 않는지(주문당 1장).
    # test-030 refusal: four days late, coupon already issued on 9/9
    _Order("O-92035", 30, _DELIVERED, kst(9, 1, 9, 25), (("V-9213-02", 1),), _d(9, 4),
           shipped_at=kst(9, 3, 17, 45), delivered_at=kst(9, 8, 13, 30)),
    # test-031 거절: 최근 30일에 보상 쿠폰을 2장 받은 고객의 세 번째 발급을 거절하는지. 30일의 첫날(8월
    # 16일)에 걸린 쿠폰을 포함해 세야 한다(coupon_limit_exceeded). test-017의 짝이다.
    # test-031 refusal: coupons of 8/16 (first day inside the 30 days) and 9/2; O-92038 came one day late
    _Order("O-92036", 31, _DELIVERED, kst(8, 10, 14, 0), (("V-9216-01", 1),), _d(8, 13), carrier="새벽택배",
           shipped_at=kst(8, 12, 10, 0), delivered_at=kst(8, 15, 16, 20)),
    _Order("O-92037", 31, _DELIVERED, kst(8, 28, 20, 30), (("V-9217-01", 1),), _d(8, 31), carrier="나래택배",
           shipped_at=kst(8, 29, 15, 0), delivered_at=kst(8, 31, 12, 0)),
    _Order("O-92038", 31, _DELIVERED, kst(9, 8, 9, 45), (("V-9209-01", 1),), _d(9, 11), carrier="두루택배",
           shipped_at=kst(9, 10, 14, 30), delivered_at=kst(9, 12, 11, 40)),
    # test-032 거절: 취소된 주문의 지연 쿠폰 요청을 거절하는지(coupon_order_cancelled, 시험용에만 있는 코드).
    # test-032 refusal: cancelled after the promised date had passed, 39,300 refund pending
    _Order("O-92039", 32, _CANCELLED, kst(9, 6, 9, 0), (("V-9213-01", 1),), _d(9, 9),
           cancelled_at=kst(9, 12, 20, 30), cancel_reason=db.CancelReason.DELIVERY_TOO_SLOW),
    # test-033 복합: 한 고객의 두 주문에 취소와 배송지 변경을 각각 맞는 주문에 처리하는지.
    # test-033 composite: cancel O-92041 (17,200 + 3,000 = 20,200), move O-92040 to the office
    _Order("O-92040", 33, _PREPARING, kst(9, 12, 10, 20), (("V-9211-01", 1),), _d(9, 16)),
    _Order("O-92041", 33, _PAID, kst(9, 13, 12, 50), (("V-9212-02", 1),), _d(9, 17)),
    # test-034 복합: 오배송 반품(배송비 0원)과 다른 주문의 4일 지연 쿠폰(5,000원 구간)을 함께 처리하는지. 이
    # 과제의 now는 9월 15일이다.
    # test-034 composite (now 9/15): wrong item on line 2 (27,800, no fee); O-92043 came four days late
    _Order("O-92042", 34, _DELIVERED, kst(9, 6, 15, 10), (("V-9204-02", 1), ("V-9205-02", 1)), _d(9, 9),
           shipped_at=kst(9, 7, 16, 0), delivered_at=kst(9, 9, 12, 45)),
    _Order("O-92043", 34, _DELIVERED, kst(9, 3, 11, 35), (("V-9220-01", 1),), _d(9, 6), method=_SIMPLE,
           carrier="새벽택배", shipped_at=kst(9, 6, 10, 0), delivered_at=kst(9, 10, 15, 15)),
    # test-035 복합: 사이즈 교환을 접수하고, 직접 처리할 수 없는 계정 문의는 계정 분류 티켓으로 남기는지.
    # test-035 composite: exchange M -> L, plus an account ticket
    _Order("O-92044", 35, _DELIVERED, kst(9, 8, 19, 5), (("V-9203-01", 1),), _d(9, 11), carrier="나래택배",
           shipped_at=kst(9, 9, 16, 20), delivered_at=kst(9, 11, 10, 30)),
    # test-036 복합: 서로 다른 주문의 배송지 변경과 취소, 품절 옵션의 재입고 문의 티켓까지 세 가지를 모두
    # 처리하는지.
    # test-036 composite: address change, cancel (14,600 + 3,000 = 17,600), product ticket
    _Order("O-92045", 36, _PAID, kst(9, 13, 9, 15), (("V-9214-01", 1),), _d(9, 17)),
    _Order("O-92046", 36, _PREPARING, kst(9, 12, 22, 0), (("V-9212-01", 1),), _d(9, 16), method=_SIMPLE),
    # test-037 복합: 취소는 처리하고, 업무 범위 밖인 가격 조정은 거절한 뒤 고객이 요구하면 이관하는지.
    # test-037 composite: cancel (13,900 x 2 + 3,000 = 30,800), then a price adjustment nobody here can do
    _Order("O-92047", 37, _PAID, kst(9, 13, 21, 30), (("V-9216-01", 2),), _d(9, 17)),
    _Order("O-92048", 37, _DELIVERED, kst(9, 7, 10, 10), (("V-9219-01", 1),), _d(9, 10), carrier="두루택배",
           shipped_at=kst(9, 8, 15, 0), delivered_at=kst(9, 10, 11, 20)),
    # test-038 복합: 불량 반품을 먼저 접수해야 자격이 생기는 불량 보상 쿠폰을 같은 주문에 발급하는지(순서가
    # 중요).
    # test-038 composite: defective return of line 1 (52,600), then the defective-item coupon
    _Order("O-92049", 38, _DELIVERED, kst(9, 6, 17, 25), (("V-9202-02", 1), ("V-9210-02", 1)), _d(9, 9),
           shipped_at=kst(9, 7, 15, 30), delivered_at=kst(9, 9, 14, 0)),
    # test-039 복합: 기한 안의 교환은 접수하고, 기한이 지난 다른 주문의 반품은 거절하는지.
    # test-039 composite: exchange on O-92050 is fine, return of O-92051 (received 9/1) is too late
    _Order("O-92050", 39, _DELIVERED, kst(9, 6, 12, 0), (("V-9218-01", 1),), _d(9, 9), carrier="새벽택배",
           shipped_at=kst(9, 7, 17, 0), delivered_at=kst(9, 9, 10, 50)),
    _Order("O-92051", 39, _DELIVERED, kst(8, 29, 18, 45), (("V-9215-02", 1),), _d(9, 2), carrier="나래택배",
           shipped_at=kst(8, 31, 9, 30), delivered_at=kst(9, 1, 15, 40)),
    # test-040 복합: 세 주문에 걸친 여러 줄 반품, 배송지 변경, 2일 지연 쿠폰을 모두 맞게 처리하는지.
    # test-040 composite: return lines 1+2 (21,300 + 12,400 x 2 - 3,000 = 43,100), address change,
    # coupon for O-92054 (two days late)
    _Order("O-92052", 40, _DELIVERED, kst(9, 8, 13, 30),
           (("V-9206-01", 1), ("V-9207-01", 2), ("V-9215-02", 1)), _d(9, 11), carrier="두루택배",
           shipped_at=kst(9, 9, 14, 0), delivered_at=kst(9, 11, 16, 10)),
    _Order("O-92053", 40, _PREPARING, kst(9, 12, 8, 30), (("V-9209-02", 1),), _d(9, 16)),
    _Order("O-92054", 40, _DELIVERED, kst(9, 8, 21, 0), (("V-9210-01", 1),), _d(9, 11), method=_SIMPLE,
           shipped_at=kst(9, 11, 10, 0), delivered_at=kst(9, 13, 13, 20)),
)  # fmt: skip

# 대화 전에 이미 있는 요청: test-006의 단순 변심 반품, test-018의 불량 교환, test-031 고객의 불량 반품(그 뒤
# 불량 쿠폰이 발급됨).
# Returns and exchanges that exist before the conversation:
# (order_id, kind, reason, line_nos, exchange variant or None, created_at)
_REQUESTS = (
    ("O-92007", _RETURN, db.RequestReason.CHANGED_MIND, (1,), None, kst(9, 11, 10, 30)),
    ("O-92023", _EXCHANGE, db.RequestReason.DEFECTIVE, (1,), "V-9211-01", kst(9, 10, 14, 15)),
    ("O-92037", _RETURN, db.RequestReason.DEFECTIVE, (1,), None, kst(9, 2, 11, 10)),
)

# 대화 전에 이미 있는 보상 쿠폰. 30일 한도 과제의 핵심이다: test-017 고객은 8/15(30일 밖)·9/5(안) 두 장,
# test-031 고객은 8/16(30일의 첫날)·9/2 두 장, test-030 주문은 이미 한 장. 금액은 rules.py의 지연 일수 구간과
# 같다.
# Compensation coupons that exist before the conversation: (order_id, reason, amount_won, issued_at)
_COUPONS = (
    ("O-92020", _DELAY, 5000, kst(8, 15, 11, 20)),  # three days late
    ("O-92021", _DELAY, 2000, kst(9, 5, 9, 40)),  # two days late
    ("O-92035", _DELAY, 5000, kst(9, 9, 10, 15)),  # four days late
    ("O-92036", _DELAY, 2000, kst(8, 16, 9, 30)),  # two days late
    ("O-92037", _DEFECT, 3000, kst(9, 2, 15, 0)),  # after the defective return above
)

# 이미 있는 티켓: test-020 고객의 옛 배송 티켓. 그래서 새 티켓 번호는 TK-C-9220-2가 되어야 한다(티켓 id는
# 고객별 순번).
# (customer number, order_id or None, category, body, created_at)
_TICKETS = (
    (20, "O-92025", db.TicketCategory.DELIVERY, "부재 시 경비실에 맡겨 달라는 요청", kst(7, 28, 9, 10)),
)


# 고객 번호로 첫 주소(집)의 우편번호와 주소를 만든다. 난수 없이 번호의 나머지로 가상 도시·구·도로를 고르므로
# 언제나 같은 주소가 나온다.
def _home(number: int) -> tuple[str, str]:
    """(postal_code, address) of address 1, spread over the fictional cities by the customer number."""
    city, district, road = CITIES[number % 5], DISTRICTS[number % 6], ROADS[number % 8]
    unit = f"{101 + number % 9}동 {number % 15 + 1}0{number % 4 + 1}호"
    return f"{10000 + number * 2017:05d}", f"{city} {district} {road} {10 + number * 3}, {unit}"


# 시험용 고정 행을 Rows에 더한다. 순서: 고객과 주소 → 상품과 옵션 → 주문·결제·배송 → 이미 있는 요청 → 쿠폰 →
# 티켓. 호출 위치: seed.build_seed_engine.
def add_fixtures(rows: Rows) -> None:
    """Append customers, addresses, products, orders and the rest to `rows`."""
    addresses: dict[str, db.CustomerAddress] = {}
    for number, name, email in _CUSTOMERS:
        customer_id = f"C-{9200 + number}"
        rows.customers.append(
            db.Customer(
                id=customer_id,
                name=name,
                phone=f"0100000{9200 + number}",
                email=f"{email}@example.com",
                grade=db.CustomerGrade.NORMAL,
                # 가입 시각도 번호에서 계산한다(4~7월 사이).
                joined_at=kst(4 + number % 4, 1 + number % 27, 9 + number % 10, 0),
            )
        )
        places = [("집", "", *_home(number)), *_EXTRA_ADDRESSES.get(number, ())]
        for n, (label, recipient, postal_code, address) in enumerate(places, start=1):
            row = db.CustomerAddress(
                id=f"AD-{customer_id}-{n}",
                customer_id=customer_id,
                label=label,
                recipient=recipient or name,
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

    items: dict[tuple[str, int], db.OrderItem] = {}
    # 주문마다 new_order로 금액과 복사본을 계산하고, 결제(주문 3분 뒤)와 배송 행을 만든다. 취소면 환불액은
    # 결제 금액 전액이다. 송장 번호는 5552-0914-(9200 + 표 순번)이라 다른 고정 행과 겹치지 않는다.
    for index, spec in enumerate(_ORDERS, start=1):
        customer_id = f"C-{9200 + spec.customer}"
        cancelled = spec.status == _CANCELLED
        order, order_items = new_order(
            spec.order_id,
            customer_id,
            spec.status,
            spec.ordered_at,
            addresses[f"AD-{customer_id}-{spec.address}"],
            [(products[variants[v].product_id], variants[v], qty) for v, qty in spec.lines],
            cancelled_at=spec.cancelled_at,
            cancel_reason=spec.cancel_reason,
        )
        rows.orders.append(order)
        rows.items.extend(order_items)
        items.update({(spec.order_id, item.line_no): item for item in order_items})

        if not cancelled:
            pay_status = db.PaymentStatus.PAID
        elif spec.refunded:
            pay_status = db.PaymentStatus.REFUNDED
        else:
            pay_status = db.PaymentStatus.REFUND_PENDING
        rows.payments.append(
            db.Payment(
                order_id=spec.order_id,
                method=spec.method,
                amount_won=order.total_won,
                status=pay_status,
                refund_won=order.total_won if cancelled else 0,
                paid_at=spec.ordered_at + timedelta(minutes=3),
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
                order_id=spec.order_id,
                carrier=spec.carrier,
                tracking_no=f"5552-0914-{9200 + index}",
                status=ship_status,
                shipped_at=spec.shipped_at,
                delivered_at=spec.delivered_at,
                promised_by=spec.promised_by,
            )
        )

    # 이미 있는 요청 행. id와 환불액은 도구와 같은 규칙(RT/EX-주문-줄 번호, 반품 환불액 = Σ(단가 × 수량) −
    # 단순 변심 배송비 3,000원)이고, 해당 줄을 접수됨 상태로 바꾼다. 줄 번호는 첫 번째 것을 쓰는데, 여기
    # 요청은 모두 줄 하나라 '가장 작은 줄 번호' 규칙과 같다.
    for order_id, kind, reason, line_nos, exchange_variant_id, created_at in _REQUESTS:
        lines = [items[(order_id, n)] for n in line_nos]
        is_return = kind == _RETURN
        fee = 3000 if reason == db.RequestReason.CHANGED_MIND else 0
        request_id = f"{'RT' if is_return else 'EX'}-{order_id}-{line_nos[0]}"
        rows.requests.append(
            db.ServiceRequest(
                id=request_id,
                order_id=order_id,
                kind=kind,
                reason=reason,
                refund_won=sum(x.unit_price_won * x.quantity for x in lines) - fee if is_return else 0,
                return_fee_won=fee,
                created_at=created_at,
            )
        )
        for line in lines:
            line.status = db.ItemStatus.RETURN_REQUESTED if is_return else db.ItemStatus.EXCHANGE_REQUESTED
            rows.request_items.append(
                db.ServiceRequestItem(
                    request_id=request_id,
                    line_no=line.line_no,
                    quantity=line.quantity,
                    exchange_variant_id=exchange_variant_id,
                )
            )

    # 쿠폰 행에는 고객 id가 필요하므로 주문 → 고객 대응표를 만든다. 만료는 발급 + 30일이다.
    owners = {spec.order_id: f"C-{9200 + spec.customer}" for spec in _ORDERS}
    for order_id, reason, amount, issued_at in _COUPONS:
        rows.coupons.append(
            db.Coupon(
                id=f"CP-{order_id}-1",
                customer_id=owners[order_id],
                kind=db.CouponKind.COMPENSATION,
                amount_won=amount,
                reason=reason,
                order_id=order_id,
                issued_at=issued_at,
                expires_at=issued_at + timedelta(days=30),
                used_at=None,
            )
        )

    # 티켓 id는 도구와 같은 TK-고객-순번이다.
    count: dict[int, int] = {}
    for number, order_id, category, body, created_at in _TICKETS:
        count[number] = count.get(number, 0) + 1
        rows.tickets.append(
            db.Ticket(
                id=f"TK-C-{9200 + number}-{count[number]}",
                customer_id=f"C-{9200 + number}",
                order_id=order_id,
                category=category,
                body=body,
                created_at=created_at,
            )
        )
