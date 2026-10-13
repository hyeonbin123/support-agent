"""Numbers a person read aloud, back in writing. For a real microphone; the service uses it.

Synthesised speech comes back from the recogniser with its digits written as digits ("010-0000-9103"). A
person reading a phone number aloud came back as words: "공일공 공공공공 육일팔구입니다". This is the
inverse of `verbalize` for the two things a customer has to get across exactly, a mobile number and an
order number. It is not part of the measured V2 normaliser (voice/normalize.py): that one was built from
the development records, this one from a live microphone test of the service.
"""

# [파일 역할] 사람이 소리 내어 읽은 휴대전화 번호와 주문 번호를 숫자 표기로 되돌린다. 서비스의 음성 입력에서만
# 쓴다.
# [왜 필요한가] 실제 마이크로 서비스를 시험해 보니 사람이 부른 전화번호가 '공일공 공공공공 육일팔구'처럼
# 한글로 적혔다. 합성 음성에서는 010-0000-9103처럼 숫자로 적히던 것이다 (docs/voice.md "채팅 서비스에서 말로
# 상담하기").
# [설계 이유] 측정한 V2 정규화(normalize.py)와 일부러 분리했다. V2는 개발용 기록에서 정한 절차대로 만든 측정
# 대상이고, 이것은 실제 마이크 시험에서 나온 서비스용 보정이라 측정과 무관하다. 고객이 정확히 전해야 하는 두
# 값(휴대전화 번호, 주문 번호)만 되돌린다.
# [연결] service/voice_frontend.py가 normalize_heard() 다음에 spoken_to_written()을 부른다. 테스트는
# verbalize()로 풀어 쓴 글이 이 함수로 원래 글에 돌아오는지도 확인한다.

from __future__ import annotations

# 규칙은 모두 정규식이다. 모델 없이 같은 입력에 같은 출력을 낸다.
import re

# 한자어 숫자 낱말 → 숫자. 0은 '공'과 '영' 둘 다, 6은 '육'과 그 다른 표기 '륙'을 받는다.
_VALUE = {"공": "0", "영": "0", "일": "1", "이": "2", "삼": "3", "사": "4", "오": "5", "육": "6",
          "륙": "6", "칠": "7", "팔": "8", "구": "9"}  # fmt: skip
# _D: 숫자 한 자리(아라비아 숫자나 숫자 낱말). _GAP: 자리 사이에 끼어도 되는 공백·쉼표·점·하이픈.
_D = "[0-9공영일이삼사오육륙칠팔구]"
_GAP = r"[\s,.\-]*"
# 010 and exactly eight more digits. Exactly, because what follows is often a particle that is itself a digit
# word: "...육일팔구이고" ends with the particle "이고", not with a 2.
# 010(공일공/영일영 등) 뒤에 정확히 여덟 자리. 앞에 한글이나 숫자가 붙어 있으면 다른 낱말이나 더 긴 숫자의
# 일부로 보고 바꾸지 않는다. 한 자리 모자란 번호('공일공 공공공공 육일팔')는 적을 수 있는 번호가 아니라 그대로
# 둔다 (tests/test_voice.py).
_MOBILE = re.compile(rf"(?<![가-힣0-9])([0공영]{_GAP}[1일]{_GAP}[0공영])((?:{_GAP}{_D}){{8}})")
# "오 다시 구일공일공": the letter O, the hyphen said as "다시", five digits
# '오'·'5'·'O' 다음에 하이픈을 읽은 말('다시', '대시', '하이픈' 또는 -)과 다섯 자리. 주문 번호 O-XXXXX의
# 형태다. 앞에 한글·숫자·영문이 붙어 있으면 바꾸지 않는다.
_ORDER = re.compile(rf"(?<![가-힣0-9A-Za-z])[오5Oo]\s*(?:다시|대시|하이픈|-)\s*((?:{_D}\s*){{5}})")


# 글에서 숫자 자리만 골라 아라비아 숫자 문자열로 바꾼다 (공백·구분자는 버린다).
def _digits(text: str) -> str:
    return "".join(_VALUE.get(ch, ch) for ch in text if ch in _VALUE or ch.isdigit())


# 읽은 번호를 적힌 번호로 되돌린다: 휴대전화 번호 → 010-XXXX-XXXX, 주문 번호 → O-XXXXX. verbalize()의 역함수
# 역할이지만 이 두 값에 대해서만이다.
# 호출 위치: service/voice_frontend.py (normalize_heard를 먼저 거친 글에 적용).
def spoken_to_written(text: str) -> str:
    # 앞 덩어리는 정규식이 010의 여러 읽기만 받으므로 항상 '010'이고, 나머지 여덟 자리를 4-4로 나눈다.
    def mobile(match: re.Match[str]) -> str:
        rest = _digits(match[2])
        return f"010-{rest[:4]}-{rest[4:]}"

    text = _MOBILE.sub(mobile, text)
    # 다섯 자리를 잡는 부분이 끝의 공백까지 먹었으면 공백 하나를 되돌려 다음 낱말이 번호에 붙지 않게 한다.
    return _ORDER.sub(lambda m: f"O-{_digits(m[1])}" + (" " if m[1].endswith(" ") else ""), text)
