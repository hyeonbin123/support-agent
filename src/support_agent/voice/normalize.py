"""V2: rule-based repair of what the recogniser wrote, before the agent reads it.

Every rule here comes from the development run of V1 (`analyze voice-worst` on its records): how synthesised
speech of the customers' utterances came back from the recogniser. Nothing was taken from the test tasks.

What the rules can repair is notation: the recogniser heard the right sounds and wrote them differently
("오 다시 구일공일공" -> "5-91010", "골뱅이 이그잼플 닷컴" -> "골뱅이그젠플.com"). What they cannot repair is
lost sound: a dropped digit ("010-0000-921"), a misheard name ("배성분"), a misspelt mail address.
"""

from __future__ import annotations

import re

# "O-91010" is said "오 다시 구일공일공"; "오" is also the digit 5, and that is what the recogniser writes.
_ORDER_ID = re.compile(r"(?<![\dA-Za-z-])5\s?-\s?(\d{4,6})(?!\d)")
# "5에서 9100": "다시" heard as a range
_ORDER_ID_RANGE = re.compile(r"(?<![\dA-Za-z-])5에서\s?(\d{4,6})(?!\d)")
# 010, 0000, 9103 / 010 0000 9103
_PHONE = re.compile(r"(?<!\d)(01\d)[\s,.\-]{1,3}(\d{3,4})[\s,.\-]{1,3}(\d{4})(?!\d)")
# 12만 원, 9만 7,600원
_MAN_WON = re.compile(r"(?<![\d,])(\d{1,4})만\s?(?:(\d{1,3}(?:,\d{3})*|\d{1,4})\s?)?원")
# The mail domain as it was heard: the at sign as a word, a hyphen or a dot, and the domain in Korean or in
# a spelling of its sound. The shop's customers all use example.com; a deployment would list its domains.
_DOMAIN = r"(?:e-)?(?:example|igsemple|이그잼플|이그젠플|그젠플)"
_TLD = r"(?:\s?\.\s?com|\s?닷컴|\s?점\s?컴)"
_AT = r"(?:\s?(?:골뱅이|golbenie|@)\s?|[-.]golbenie[-.]?|[-.])"
_EMAIL = re.compile(rf"([A-Za-z0-9]+(?:[._][A-Za-z0-9]+)*?){_AT}{_DOMAIN}{_TLD}", re.IGNORECASE)


def _won(match: re.Match[str]) -> str:
    rest = int(match[2].replace(",", "")) if match[2] else 0
    if rest >= 10_000:
        return match[0]  # "12만 34,000원" is not a number anybody says: leave it alone
    return f"{int(match[1]) * 10_000 + rest:,}원"


def normalize_heard(text: str) -> str:
    text = _EMAIL.sub(lambda m: f"{m[1]}@example.com", text)
    text = _ORDER_ID.sub(r"O-\1", text)
    text = _ORDER_ID_RANGE.sub(r"O-\1", text)
    text = _PHONE.sub(r"\1-\2-\3", text)
    return _MAN_WON.sub(_won, text)
