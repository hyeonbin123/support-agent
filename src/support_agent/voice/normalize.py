"""V2: rule-based repair of what the recogniser wrote, before the agent reads it.

Every rule here comes from the development run of V1 (`analyze voice-worst` on its records): how synthesised
speech of the customers' utterances came back from the recogniser. Nothing was taken from the test tasks.

What the rules can repair is notation: the recogniser heard the right sounds and wrote them differently
("오 다시 구일공일공" -> "5-91010", "골뱅이 이그잼플 닷컴" -> "골뱅이그젠플.com"). What they cannot repair is
lost sound: a dropped digit ("010-0000-921"), a misheard name ("배성분"), a misspelt mail address.
"""

# [파일 역할] 음성 채널 V2의 정규화: 인식기가 적은 글에서 도구 인자가 되는 값의 '표기'만 고쳐 에이전트에게
# 넘긴다. 규칙은 네 개다: 주문 번호의 O, 이메일의 골뱅이와 도메인, 전화번호 구분, 만 단위 금액
# (docs/experiments.md 5단계 "V2의 정규화 규칙").
# [설계 이유] 규칙은 개발용 V1 기록에서 엔티티를 잃은 발화(analyze voice-worst)만 보고 만들고, 개발용 V2를
# 재기 전에 커밋했다. 시험용 기록은 보지 않았다. 소리 자체를 잃은 것(빠진 숫자, 잘못 들은 이름)은 고칠 수
# 없으므로 건드리지 않는다.
# [연결] run.py가 V2·V4 채널의 normalizer로, voice_real.py가 11단계의 두 받아쓰기에,
# service/voice_frontend.py가 서비스 음성 입력에(spoken.py보다 먼저) 쓴다.

from __future__ import annotations

# 모든 규칙이 정규식 치환이다. 모델 없이 같은 입력에 같은 출력을 내고, 테스트가 이 함수의 결과를 다시 넣어도
# 바뀌지 않는다(멱등)는 것을 확인한다.
import re

# "O-91010" is said "오 다시 구일공일공"; "오" is also the digit 5, and that is what the recogniser writes.
# 5 다음에 하이픈(앞뒤 공백 허용)과 숫자 4~6자리를 O-번호로 되돌린다. 앞에 숫자·영문·하이픈이 붙어 있으면 다른
# 값의 일부로 보고 건드리지 않고, 숫자 자리를 4~6자리로 묶어 '5-5 사이즈', '5-10일' 같은 말은 바꾸지 않는다
# (tests/test_voice.py).
_ORDER_ID = re.compile(r"(?<![\dA-Za-z-])5\s?-\s?(\d{4,6})(?!\d)")
# "5에서 9100": "다시" heard as a range
_ORDER_ID_RANGE = re.compile(r"(?<![\dA-Za-z-])5에서\s?(\d{4,6})(?!\d)")
# 010, 0000, 9103 / 010 0000 9103
# 휴대전화 번호의 세 덩어리 사이에 공백·쉼표·점·하이픈이 1~3자 섞인 것(예: 인식기가 적은 '010, 0000, 9103')을
# 010-0000-9103 형태로 맞춘다. 앞뒤에 숫자가 더 붙어 있으면 건드리지 않는다.
_PHONE = re.compile(r"(?<!\d)(01\d)[\s,.\-]{1,3}(\d{3,4})[\s,.\-]{1,3}(\d{4})(?!\d)")
# 12만 원, 9만 7,600원
# 'N만 원', 'N만 M원'(M은 쉼표 천 단위 또는 4자리까지)을 아라비아 숫자 금액으로 바꾼다. 앞에 숫자나 쉼표가
# 붙어 있으면 더 긴 금액의 일부로 보고 건드리지 않는다.
_MAN_WON = re.compile(r"(?<![\d,])(\d{1,4})만\s?(?:(\d{1,3}(?:,\d{3})*|\d{1,4})\s?)?원")
# The mail domain as it was heard: the at sign as a word, a hyphen or a dot, and the domain in Korean or in
# a spelling of its sound. The shop's customers all use example.com; a deployment would list its domains.
_DOMAIN = r"(?:e-)?(?:example|igsemple|이그잼플|이그젠플|그젠플)"
_TLD = r"(?:\s?\.\s?com|\s?닷컴|\s?점\s?컴)"
# 골뱅이(@)가 들린 모양: 한글 '골뱅이', 로마자 'golbenie', 기호 @, 또는 하이픈·점으로 바뀐 경우. 위 _DOMAIN,
# _TLD와 합쳐 개발용 V1 기록에 나온 변형(골뱅이그젠플.com, -golbenie-example.com, 닷컴 등)을 덮는다.
_AT = r"(?:\s?(?:골뱅이|golbenie|@)\s?|[-.]golbenie[-.]?|[-.])"
# 아이디(영문·숫자, 점이나 밑줄로 이어짐) + 골뱅이 + 도메인 + .com을 잡아 아이디만 살리고 뒤를
# @example.com으로 바꾼다. 대소문자를 가리지 않는다. 아이디 철자가 빠진 것은 고칠 수 없다.
_EMAIL = re.compile(rf"([A-Za-z0-9]+(?:[._][A-Za-z0-9]+)*?){_AT}{_DOMAIN}{_TLD}", re.IGNORECASE)


# 만 단위 금액 치환 함수: N × 10,000 + 나머지를 천 단위 쉼표로 적는다. 나머지가 10,000 이상이면 사람이 그렇게
# 말하지 않는 수라 원문 그대로 둔다.
def _won(match: re.Match[str]) -> str:
    rest = int(match[2].replace(",", "")) if match[2] else 0
    if rest >= 10_000:
        return match[0]  # "12만 34,000원" is not a number anybody says: leave it alone
    return f"{int(match[1]) * 10_000 + rest:,}원"


# 인식된 글에 네 규칙을 차례로 적용한다: 이메일 → 주문 번호(하이픈, '에서') → 전화번호 → 금액.
# 호출 위치: run.py(V2·V4 채널의 normalizer), voice_real.py measure()(깨끗한·열화한 받아쓰기 모두),
# service/voice_frontend.py(서비스의 음성 입력, 그 뒤 spoken_to_written).
# 효과: 같은 개발용 V1 기록에 적용하면 주문 번호 생존 0 → 306/364, 이메일 0 → 83/272가 됐다. 규칙을 만든
# 자료에 대한 수치라 성능 추정은 아니다 (docs/experiments.md 5단계).
def normalize_heard(text: str) -> str:
    # 이 쇼핑몰 고객의 도메인은 모두 example.com이라 도메인을 고정해서 되돌린다. 실제 서비스라면 자주 쓰는
    # 도메인 목록이 이 자리에 들어간다.
    text = _EMAIL.sub(lambda m: f"{m[1]}@example.com", text)
    text = _ORDER_ID.sub(r"O-\1", text)
    text = _ORDER_ID_RANGE.sub(r"O-\1", text)
    text = _PHONE.sub(r"\1-\2-\3", text)
    return _MAN_WON.sub(_won, text)
