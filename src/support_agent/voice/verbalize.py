"""글로 쓴 한국어 채팅 글 -> 고객이 소리 내어 말할 낱말(한글만).

음성 합성기는 "O-91010"이나 "38,900원"을 아무도 그렇게 말하지 않는 식으로 읽으므로, 먼저 글을 풀어 쓴다:
식별자와 전화번호는 한 자리씩, 금액과 날짜는 한자어 수, 세는 말 앞은 고유어 수, 로마자는 글자 이름으로.
모델 없는 순수 함수들이다.
"""

# [파일 역할] 고객이 채팅에 쓴 글을 사람이 소리 내어 말하는 글(한글, 공백, 문장부호만)로 풀어 쓴다. 음성
# 채널은 이 글을 합성기에 넣는다.
# [왜 필요한가] 합성기에 "O-91010"이나 "38,900원"을 그대로 주면 아무도 그렇게 말하지 않는 소리가 나온다
# (docs/voice.md "말로 풀어 쓰기"). 그래서 번호는 한 자리씩, 금액·날짜는 한자어 수, 세는 말 앞은 고유어 수,
# 로마자는 글자 이름으로 바꾼다.
# [설계 이유] 모델 없는 순수 함수라 같은 글은 언제나 같은 풀어 쓰기가 되고, 캐시 키(말로 푼 글)도 고정된다.
# 개발용 기준선 기록의 고객 발화 512개가 모두 한글·공백·문장부호만 남는지를 테스트가 확인한다 (docs/voice.md).
# [연결] voice/channel.py(평가 채널의 hear)와 service/voice_frontend.py(에이전트 답을 소리로)가 verbalize()를
# 부른다. spoken.py는 전화번호·주문 번호에 대해 이 함수의 역방향을 맡는다.

from __future__ import annotations

# re: 번호·날짜·금액 패턴 찾기와 치환. unicodedata: NFKC 정규화로 전각 숫자·로마자 같은 변형 문자를 보통
# 문자로 맞춘 뒤 패턴을 돌린다.
import re
import unicodedata

# 자리 하나씩 읽을 때의 숫자 이름. 인덱스가 숫자 값이라 DIGIT_NAMES[int(ch)]로 읽는다. 0은 전화번호를 부를
# 때처럼 '공'이다.
DIGIT_NAMES = "공일이삼사오육칠팔구"  # 한 자리씩 읽기: 전화번호와 식별자(0은 "공")
# 한자어 수: 한 자리 숫자, 네 자리 안의 자리 단위(십·백·천), 네 자리 묶음 단위(만·억·조).
_SINO = ("", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
_SMALL_UNITS = ("", "십", "백", "천")
_BIG_UNITS = ("", "만", "억", "조")
# 고유어 수(1~99): 일의 자리(한·두·세…)와 십의 자리(열·스물·서른…). 세는 말 앞에서 쓴다.
_NATIVE_ONES = ("", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉")
_NATIVE_TENS = ("", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔")
# 로마자 글자의 한국어 이름. 이메일 아이디와 주문 번호의 영문자를 한 글자씩 읽을 때 쓴다.
LETTER_NAMES = {
    "a": "에이", "b": "비", "c": "씨", "d": "디", "e": "이", "f": "에프", "g": "지", "h": "에이치",
    "i": "아이", "j": "제이", "k": "케이", "l": "엘", "m": "엠", "n": "엔", "o": "오", "p": "피",
    "q": "큐", "r": "알", "s": "에스", "t": "티", "u": "유", "v": "브이", "w": "더블유", "x": "엑스",
    "y": "와이", "z": "지",
}  # fmt: skip
# 글자 하나씩이 아니라 낱말로 읽는 로마자 (도메인 조각, 단위 등). 예: example → 이그잼플, mm → 밀리미터.
WORDS = {
    "example": "이그잼플", "com": "컴", "net": "넷", "kr": "케이알", "co": "씨오", "gmail": "지메일",
    "naver": "네이버", "mm": "밀리미터", "cm": "센티미터", "km": "킬로미터", "kg": "킬로그램",
    "g": "그램", "ml": "밀리리터", "l": "리터", "ok": "오케이", "vip": "브이아이피", "as": "에이에스",
}  # fmt: skip
# 고유어 수로 세는 말("두 개", "열 시"). 그 밖의 단위는 모두 한자어 수로 읽는다.
# _number()는 이 튜플에서 앞에서부터 처음 맞는 것을 고르므로 순서가 중요하다: '개월'이 '개'보다, '시간'이
# '시'보다 앞에 있어야 긴 말이 먼저 잡힌다.
NATIVE_COUNTERS = (
    "개월", "개", "벌", "켤레", "장", "병", "명", "시간", "시", "살", "잔", "마리", "대", "권", "통",
    "박스", "상자", "가지", "군데", "곳", "달", "번째",
)  # fmt: skip
# '개월'은 한자어 수로 읽는다 ('3개월' → '삼 개월', docs/voice.md 표). NATIVE_COUNTERS에 넣은 것은 "개"가
# "개월"의 앞부분에 맞지 않게 하려는 것뿐이다.
_SINO_ONLY = {"개월"}
# 불규칙하게 읽는 달: 6월은 '육월'이 아니라 '유월', 10월은 '십월'이 아니라 '시월'.
_MONTHS = {6: "유", 10: "시"}  # 유월, 시월

# 아래 정규식들은 verbalize()가 이 순서대로 적용한다. 이메일·전화번호·주문 번호처럼 숫자와 영문이 섞인 값을
# 먼저 통째로 바꿔야, 뒤의 숫자·로마자 규칙이 그 일부를 따로 읽지 않는다.
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
# 010-1234-5678, 01012345678, (010) 1234-5678, +82 10-1234-5678: find_customer가 받는 꼴들
_PHONE = re.compile(r"(?<!\d)(?:\+82[- ]?|\(?0)(1\d)\)?[- .]?(\d{3,4})[- .]?(\d{4})(?!\d)")
# 주문·접수 번호 같은 식별자: 영문 1~3자 뒤에 '-'로 이어진 덩어리 (metrics.py의 id 패턴과 같은 형태).
_IDENT = re.compile(r"(?<![A-Za-z])[A-Za-z]{1,3}(?:-[A-Za-z0-9]+)+")
# _ISO_DATE: 2026-09-14 같은 연-월-일(구분자는 같은 것끼리, 역참조 \2). _MINUS: 앞에 낱말이나 닫는 괄호가 없는
# '-숫자'는 음수 부호로 읽는다.
_ISO_DATE = re.compile(r"(?<![\d./-])(\d{4})([-./])(\d{1,2})\2(\d{1,2})(?![\d/-])")
_MINUS = re.compile(r"(?<![\w가-힣)])-(?=\d)")
# 숫자만으로 된 묶음이 하이픈으로 셋 이상 이어진 번호(송장 번호 5551-0914-9121, 지역 번호 전화 02-1234-5678).
# 범위는 수 두 개라 묶음이 셋 이상이면 범위가 아니다. 연월일(2026-09-14)은 _ISO_DATE가 먼저 바꾼다.
_DIGIT_GROUPS = re.compile(r"(?<![\d.-])\d+(?:-\d+){2,}(?![\d-]|\.\d)")
# _DATE: 'N월 N일'. _MONTH: 일 없이 'N월'. _CLOCK: 'HH:MM' 시각.
_DATE = re.compile(r"(?<!\d)(\d{1,2})월(\s*)(\d{1,2})일")
_MONTH = re.compile(r"(?<!\d)(\d{1,2})월")
_CLOCK = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
# '3-5일', '2~3개'처럼 단위나 %가 뒤따르는 범위. 앞 수만 바꾸고 '에서'로 이은 뒤, 뒤 수는 단위와 함께
# _NUMBER가 읽는다. 앞을 보는 묶음(3번)은 뒤따르는 단위로, 앞 수를 고유어로 읽을지 정하는 데만 쓴다(_range).
# 숫자 묶음이 셋 이상인 번호는 이보다 먼저 _DIGIT_GROUPS가 바꾸므로 여기에 오지 않는다.
_RANGE = re.compile(r"(?<!\d)(\d+)\s*[-~]\s*(\d+)(?=\s*([가-힣]+|%))")
# 수 하나: 쉼표 천 단위(38,900) 또는 숫자열, 소수부, 뒤의 공백, 단위(% 또는 바로 뒤의 한글 낱말). 단위를 함께
# 잡아야 '2개'와 '2월'을 다르게 읽을 수 있다.
_NUMBER = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(\s*)(%|[가-힣]+)?")
# _LATIN: 남은 로마자 낱말. _KEEP: 한글·공백·.,?!~ 이외의 모든 문자(마지막에 지운다).
_LATIN = re.compile(r"[A-Za-z]+")
_KEEP = re.compile(r"[^가-힣 .,?!~]")


# 네 자리씩 묶어 아래 묶음부터 만·억·조를 붙이고, 묶음 사이는 띄어 쓴다. 만 묶음이 정확히 1이면 '일만'이
# 아니라 '만'이다("10,000원" → "만 원"). 0은 '영'이다.
def sino(number: int) -> str:
    """정수를 한자어 수로: 38900 -> "삼만 팔천구백". 말할 때처럼 십, 백, 천, 만 앞의 "일"은 뺀다."""
    if number == 0:
        return "영"
    groups = []
    for index, big in enumerate(_BIG_UNITS):
        number, part = divmod(number, 10_000)
        if part:
            # 묶음 안에서 천의 자리부터 0이 아닌 자리만 읽는다. 자리 단위가 붙는 자리(place > 0)의 1은 읽지
            # 않는다 ("일십"이 아니라 "십").
            digits = "".join(
                ("" if digit == 1 and place else _SINO[digit]) + _SMALL_UNITS[place]
                for place in (3, 2, 1, 0)
                if (digit := part // 10**place % 10)
            )
            if part == 1 and index == 1:
                digits = ""  # "일만 원"이 아니라 "만 원"
            groups.append(digits + big)
        if not number:
            break
    return " ".join(reversed(groups))


# 20은 세는 말 앞에서 '스물'이 아니라 '스무'다. 범위 밖(0, 100 이상)은 한자어 수로 읽는다.
def native(number: int) -> str:
    """세는 말 앞의 고유어 수, 1..99: 2 -> "두", 20 -> "스무", 21 -> "스물한". 더 큰 수는 한자어 수다."""
    if not 0 < number < 100:
        return sino(number)
    if number == 20:
        return "스무"
    tens, ones = divmod(number, 10)
    return _NATIVE_TENS[tens] + _NATIVE_ONES[ones]


# 숫자를 한 자리씩 읽는다(전화번호, 식별자, 긴 숫자열). 숫자가 아닌 문자는 버린다.
def digits(text: str) -> str:
    return "".join(DIGIT_NAMES[int(ch)] for ch in text if ch.isdigit())


# 로마자 낱말: WORDS에 있으면 낱말로, 없으면 글자 이름으로 하나씩 읽는다.
def letters(word: str) -> str:
    known = WORDS.get(word.lower())
    return known if known else " ".join(LETTER_NAMES[ch] for ch in word.lower())


# 이메일을 소리로: 아이디는 영문 글자·숫자·기호를 하나씩(점 → '점', 밑줄 → '언더바', - → '다시'), @는
# '골뱅이', 도메인은 조각마다 낱말로 읽고 마지막 '점 컴'은 '닷컴'으로 바꾼다. 예: hajun@example.com → 에이치
# 에이 제이 유 엔 골뱅이 이그잼플 닷컴 (docs/voice.md 표).
# 앞뒤에 공백을 붙여 이어진 글과 들러붙지 않게 한다.
def _email(match: re.Match[str]) -> str:
    local, domain = match.group().split("@", 1)

    def spell(part: str) -> str:
        out = []
        for piece in re.findall(r"[A-Za-z]+|\d+|[._%+-]", part):
            if piece.isdigit():
                out.append(digits(piece))
            elif piece.isalpha():
                out.append(" ".join(LETTER_NAMES[ch] for ch in piece.lower()))
            else:
                out.append({".": "점", "_": "언더바", "+": "플러스", "-": "다시", "%": "퍼센트"}[piece])
        return " ".join(out)

    names = " 점 ".join(letters(label) for label in domain.split(".")).replace(" 점 컴", " 닷컴")
    return f" {spell(local)} 골뱅이 {names} "


# 식별자를 소리로: '-'로 나눈 조각마다 영문은 글자 이름, 숫자는 한 자리씩 읽고, 조각 사이의 '-'는 '다시'로
# 읽는다. 예: O-91010 → 오 다시 구일공일공, RT-O-91009-1 → 알 티 다시 오 다시 구일공공구 다시 일.
def _identifier(match: re.Match[str]) -> str:
    parts = []
    for part in match.group().split("-"):
        pieces = re.findall(r"[A-Za-z]+|\d+", part)
        # 영문은 글자 하나씩 읽는다: 식별자 안의 "G"는 그램이 아니라 글자다.
        parts.append(
            " ".join(
                digits(p) if p.isdigit() else " ".join(LETTER_NAMES[c] for c in p.lower()) for p in pieces
            )
        )
    return " " + " 다시 ".join(parts) + " "  # "-"는 "다시"로 읽는다


# 숫자만으로 된 묶음 번호를 소리로: 전화번호와 같이 묶음마다 한 자리씩 읽고 쉼표로 끊는다. 예:
# 5551-0914-9121 → 오오오일, 공구일사, 구일이일. 영문이 든 식별자(O-91010)는 _identifier가 '다시'로 잇는다.
# 2026-10-13 전에는 이 규칙이 없어 범위 규칙이 뒤 두 묶음을 잡았다: '5551-0914-9121입니다' →
# '오천오백오십일 구백십사에서 구천백이십일 입니다'. 커밋된 음성 기록에서 이것으로 바뀌는 고객 발화 21개는
# tests/test_voice.py에 위치를 적었고, 그 기록의 수치를 다시 재는 계획은 docs/experiments.md 9단계 끝에 있다.
def _digit_groups(match: re.Match[str]) -> str:
    return " " + ", ".join(digits(group) for group in match.group().split("-")) + " "


# 수 하나를 단위에 맞게 읽는다. 위에서부터 처음 맞는 규칙을 쓴다.
# - 16자리를 넘으면 조 위의 단위가 없으므로 한 자리씩.
# - 소수: 한자어 수 + '점' + 소수부를 한 자리씩.
# - %: 한자어 수 + '퍼센트'.
# - 세는 말('개', '시' 등, '개월' 제외)이 뒤따르면 고유어 수: '2개' → '두 개'.
# - 쉼표 없는 5자리 이상, 또는 0으로 시작하는 여러 자리 수는 금액이 아니면 식별자로 보고 한 자리씩 읽는다.
# - 나머지는 한자어 수 + 단위.
def _number(match: re.Match[str]) -> str:
    whole, fraction, gap, unit = match.groups()
    plain = whole.replace(",", "")
    unit = unit or ""
    if len(plain) > 16:  # 조 위의 단위는 여기 없으니 한 자리씩 읽는다
        return f" {digits(plain)} {unit}"
    if fraction:
        return f"{sino(int(plain))} 점 {digits(fraction)}{gap}{'퍼센트' if unit == '%' else unit}"
    if unit == "%":
        return f"{sino(int(plain))} 퍼센트"
    counter = next((c for c in NATIVE_COUNTERS if unit.startswith(c)), None)
    if counter and counter not in _SINO_ONLY:
        return f"{native(int(plain))} {unit}"
    if (
        "," not in whole
        and (len(plain) >= 5 or (len(plain) > 1 and plain[0] == "0"))
        and not unit.startswith("원")
    ):
        return f" {digits(plain)} {unit}"  # 단위 없는 긴 수는 식별자로 보고 한 자리씩 읽는다
    return f"{sino(int(plain))}{' ' if unit else ''}{unit}"


# 범위의 앞 수를 읽는다. 뒤따르는 단위가 세는 말이면 _number()와 같은 규칙으로 앞 수도 고유어로 읽고 세는 말을
# 한 번 더 붙인다: '2~3개' → '두 개에서 세 개'. 그 밖의 단위는 한자어 수 뒤에 바로 '에서'를 붙인다: '3-5일' →
# '삼에서 오 일'. 고유어 수의 세는 꼴('두')은 세는 말 없이 홀로 쓰지 않으므로 세는 말을 되풀이한다.
# 2026-10-13 전에는 앞 수를 늘 한자어로 읽어 '2~3개'가 '이에서 세 개'가 됐다. 커밋된 음성 기록의 고객 발화에는
# 세는 말 앞의 범위가 없어 기록의 말로 푼 글은 바뀌지 않는다(tests/test_voice.py, docs/experiments.md
# 5단계 끝).
def _range(match: re.Match[str]) -> str:
    first, second, unit = int(match[1]), match[2], match[3]
    counter = next((c for c in NATIVE_COUNTERS if unit.startswith(c)), None)
    if counter and counter not in _SINO_ONLY:
        return f"{native(first)} {counter}에서 {second}"
    return f"{sino(first)}에서 {second}"


# 글 → 말로 푼 글. NFKC 정규화 뒤 이메일 → 전화번호 → 식별자 → 연월일 → 숫자 묶음 번호 → 날짜(월일, 월) →
# 시각 → 범위 → 음수 → 수 → 남은 로마자 순서로 바꾸고, 말하지 않는 기호를 지운 뒤 공백을 정리한다.
# 호출 위치: voice/channel.py SpeechChannel.hear(), service/voice_frontend.py(답을 읽어 줄 때).
def verbalize(text: str) -> str:
    """`text`를 말로 푼 꼴: 한글, 공백, `.,?!~`만 남는다."""
    text = unicodedata.normalize("NFKC", text)
    text = _EMAIL.sub(_email, text)
    # 전화번호는 세 덩어리를 한 자리씩 읽고 쉼표로 끊는다: 010-0000-9103 → 공일공, 공공공공, 구일공삼. +82로
    # 시작해도 앞에 0을 붙여 010으로 읽는다.
    text = _PHONE.sub(lambda m: " " + ", ".join((digits("0" + m[1]), digits(m[2]), digits(m[3]))) + " ", text)
    text = _IDENT.sub(_identifier, text)
    # 도구 출력의 날짜 형식(2026-09-14)을 'N년 N월 N일'로 바꿔 아래 날짜 규칙과 수 규칙이 읽게 한다.
    text = _ISO_DATE.sub(lambda m: f"{m[1]}년 {int(m[3])}월 {int(m[4])}일", text)
    text = _DIGIT_GROUPS.sub(_digit_groups, text)
    text = _DATE.sub(lambda m: f"{_MONTHS.get(int(m[1]), sino(int(m[1])))}월 {sino(int(m[3]))}일", text)
    text = _MONTH.sub(lambda m: f"{_MONTHS.get(int(m[1]), sino(int(m[1])))}월", text)
    # 시각: 시는 고유어(열 시), 분은 한자어(삼십 분), 0분이면 분을 읽지 않는다.
    text = _CLOCK.sub(
        lambda m: f"{native(int(m[1]))} 시" + (f" {sino(int(m[2]))} 분" if int(m[2]) else ""), text
    )
    text = _RANGE.sub(_range, text)
    text = _MINUS.sub("마이너스 ", text)
    text = _NUMBER.sub(_number, text)
    text = _LATIN.sub(lambda m: " " + letters(m.group()) + " ", text)
    text = _KEEP.sub(" ", text)  # 따옴표, 괄호 같은 기호는 말하지 않는다
    # 치환하며 생긴 공백을 정리한다: 문장부호 앞의 공백을 지우고, 여러 공백을 하나로 줄인다.
    text = re.sub(r"\s+([.,?!~])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()
