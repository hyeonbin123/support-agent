"""Numbers a person read aloud, back in writing. For a real microphone; the service uses it.

Synthesised speech comes back from the recogniser with its digits written as digits ("010-0000-9103"). A
person reading a phone number aloud came back as words: "공일공 공공공공 육일팔구입니다". This is the
inverse of `verbalize` for the two things a customer has to get across exactly, a mobile number and an
order number. It is not part of the measured V2 normaliser (voice/normalize.py): that one was built from
the development records, this one from a live microphone test of the service.
"""

from __future__ import annotations

import re

_VALUE = {"공": "0", "영": "0", "일": "1", "이": "2", "삼": "3", "사": "4", "오": "5", "육": "6",
          "륙": "6", "칠": "7", "팔": "8", "구": "9"}  # fmt: skip
_D = "[0-9공영일이삼사오육륙칠팔구]"
_GAP = r"[\s,.\-]*"
# 010 and exactly eight more digits. Exactly, because what follows is often a particle that is itself a digit
# word: "...육일팔구이고" ends with the particle "이고", not with a 2.
_MOBILE = re.compile(rf"(?<![가-힣0-9])([0공영]{_GAP}[1일]{_GAP}[0공영])((?:{_GAP}{_D}){{8}})")
# "오 다시 구일공일공": the letter O, the hyphen said as "다시", five digits
_ORDER = re.compile(rf"(?<![가-힣0-9A-Za-z])[오5Oo]\s*(?:다시|대시|하이픈|-)\s*((?:{_D}\s*){{5}})")


def _digits(text: str) -> str:
    return "".join(_VALUE.get(ch, ch) for ch in text if ch in _VALUE or ch.isdigit())


def spoken_to_written(text: str) -> str:
    def mobile(match: re.Match[str]) -> str:
        rest = _digits(match[2])
        return f"010-{rest[:4]}-{rest[4:]}"

    text = _MOBILE.sub(mobile, text)
    return _ORDER.sub(lambda m: f"O-{_digits(m[1])}" + (" " if m[1].endswith(" ") else ""), text)
