"""5단계: 말로 푼 꼴, 음성 채널(가짜 음성 모델로), 그 지표."""

# [파일 역할] 5단계 음성 채널과 그 주변을 검증하는 테스트 묶음이다. 구역은 여섯 개다: 말로 풀어
# 쓰기(verbalize), 채널(풀어 쓰기 → 합성 → 인식 → 정규화, 캐시), 서비스의 업로드 디코딩 길이 제한, 에피소드
# 안에서의 채널 위치, 채널 지표(metrics), V2 정규화와 서비스용 '부른 번호 되돌리기'(spoken).
# [왜 중요한가] 5단계는 고객 → 에이전트 방향만 음성을 거치게 해 '에이전트가 잘못 들었다'에서 오는 차이만
# 잰다(docs/voice.md "정한 것과 이유"). 채널이 시뮬레이터 쪽 기록까지 바꾸거나, 같은 입력에서 다른 소리를
# 내거나, 정규화가 표기 아닌 것까지 고치면 측정이 무너진다.
# [방법] 실제 MeloTTS·faster-whisper 대신 가짜 합성기·인식기를 쓴다. 'O'를 숫자 5로 잘못 듣는 실제 인식 오류를
# 일부러 흉내 낸다. numpy·PyAV·faster_whisper가 필요한 테스트는 importorskip으로 건너뛰어, voice 의존성 그룹
# 없이 도는 CI에서도 묶음 전체가 통과한다.

from __future__ import annotations

# tracemalloc: 긴 녹음을 거절할 때 파이썬 쪽 메모리 최고치를 잰다. io: 인코딩한 녹음을 메모리 파일로 다룬다.
import io
import json
import re
import tracemalloc
from pathlib import Path

# NOW는 고정 시각, TOY_REGISTRY는 장난감 도구 등록부다(에피소드 구역에서 쓴다).
import pytest
from conftest import NOW
from toy_tools import TOY_REGISTRY

# episode의 _converse는 고객과 에이전트의 대화 순환이고, NOTHING_HEARD는 인식된 말이 없을 때 에이전트가 받는
# 문장이다. REPORTS는 커밋된 측정 기록 폴더로, 풀어 쓰기 테스트의 말뭉치로 쓴다. voice 패키지의 각 모듈이 검증
# 대상이다.
from support_agent import db
from support_agent.agent import agent_turn, new_state
from support_agent.chat import ScriptedProvider, ToolCall
from support_agent.config import RunConfig
from support_agent.episode import NOTHING_HEARD, _converse
from support_agent.paths import REPORTS
from support_agent.toolkit import ToolContext, execute
from support_agent.user_sim import ScriptedUser
from support_agent.voice import metrics, speech
from support_agent.voice.channel import SpeechChannel
from support_agent.voice.normalize import normalize_heard
from support_agent.voice.speech import LISTEN_RATE, Audio, AudioTooLongError, WhisperListener, wav_bytes
from support_agent.voice.spoken import spoken_to_written
from support_agent.voice.verbalize import native, sino, verbalize

# 말로 풀어 쓰기: 합성기에 'O-91010'이나 '38,900원'을 그대로 주면 아무도 그렇게 말하지 않는 소리가 나오므로,
# 합성 전에 한글로만 된 글로 바꾼다(docs/voice.md "말로 풀어 쓰기").
# -------------------------------------------------------------------- 말로 풀어 쓰기


# 한자어 수 읽기(sino)를 본다. 만 단위로 끊어 띄우고, '일만'이 아니라 '만', '일십'이 아니라 '십'처럼 맨 앞의
# 1을 읽지 않는 한국어 관습을 따르는지 확인한다.
@pytest.mark.parametrize(
    ("number", "words"),
    [
        (0, "영"),
        (10, "십"),
        (11, "십일"),
        (110, "백십"),
        (1_500, "천오백"),
        (10_000, "만"),
        (10_001, "만 일"),
        (38_900, "삼만 팔천구백"),
        (120_000, "십이만"),
        (100_000_000, "일억"),
        (123_456_789, "일억 이천삼백사십오만 육천칠백팔십구"),
    ],
)
def test_sino_korean_numbers(number, words):
    assert sino(number) == words


# 고유어 수(native)를 본다. 세는 말 앞에서 쓰는 꼴(한, 두, 세, 네, 스무)을 내고, 99를 넘으면 한자어로
# 넘어간다.
def test_native_numerals_up_to_99():
    assert [native(n) for n in (1, 2, 3, 4, 10, 20, 21, 37, 99)] == [
        "한", "두", "세", "네", "열", "스무", "스물한", "서른일곱", "아흔아홉",
    ]  # fmt: skip
    assert native(100) == "백"


# 고객 발화에 나오는 표기마다 사람이 말하는 꼴로 바뀌는지 본다: 전화번호는 한 자리씩 끊어, 주문·접수 번호의
# 하이픈은 '다시', 금액은 한자어 수 + '원', 날짜(유월·시월 같은 예외 포함), 세는 말 앞의 고유어,
# 범위·시각·백분율·단위·영문 약자·이메일, 괄호와 따옴표 제거. docs/voice.md의 풀어 쓰기 표와 같은 사례들이다.
@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("010-0000-9103입니다", "공일공, 공공공공, 구일공삼 입니다"),
        ("01000009103", "공일공, 공공공공, 구일공삼"),
        ("주문 번호 O-91010 맞습니다.", "주문 번호 오 다시 구일공일공 맞습니다."),
        ("RT-O-91009-1이에요", "알 티 다시 오 다시 구일공공구 다시 일 이에요"),
        ("38,900원 맞죠?", "삼만 팔천구백 원 맞죠?"),
        ("10,000원 쿠폰", "만 원 쿠폰"),
        ("9월 14일, 6월 3일, 10월 2일", "구월 십사일, 유월 삼일, 시월 이일"),
        ("2개 주문했고 20개, 37개", "두 개 주문했고 스무 개, 서른일곱 개"),
        ("7일 지났나요? 3개월 전, 2026년", "칠 일 지났나요? 삼 개월 전, 이천이십육 년"),
        ("3-5일 안에", "삼에서 오 일 안에"),
        ("2026-09-13 11:10에 주문", "이천이십육 년 구월 십삼일 열한 시 십 분에 주문"),
        ("10% 할인, 1.5배", "십 퍼센트 할인, 일 점 오배"),
        ("10:30에 오전 10:00 도착", "열 시 삼십 분에 오전 열 시 도착"),
        ("250mm CJ택배", "이백오십 밀리미터 씨 제이 택배"),
        ("hajun@example.com이에요", "에이치 에이 제이 유 엔 골뱅이 이그잼플 닷컴 이에요"),
        ("'기본형(화이트)'로요", "기본형 화이트 로요"),
    ],
)
def test_written_forms_are_spelled_out_the_way_they_are_said(written, spoken):
    assert verbalize(written) == spoken


# 범위의 앞 수도 뒤 수와 같은 수 체계로 읽는지 본다. 세는 말(개, 시간 등) 앞이면 앞 수도 고유어로 읽고 세는
# 말을 한 번 더 붙인다("2~3개" → "두 개에서 세 개"). 고치기 전에는 앞 수를 늘 한자어로 읽어 "이에서 세 개"가
# 됐다. 한자어 단위(일, 개월, %)의 범위는 그대로 "삼에서 오 일"이다.
@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("2~3개 남았어요", "두 개에서 세 개 남았어요"),
        ("1-2시간 걸려요", "한 시간에서 두 시간 걸려요"),
        ("3 ~ 4명이요", "세 명에서 네 명이요"),
        ("10~20개씩", "열 개에서 스무 개씩"),
        ("3-5일 안에", "삼에서 오 일 안에"),
        ("2~3개월 전", "이에서 삼 개월 전"),
        ("10~20% 할인", "십에서 이십 퍼센트 할인"),
    ],
)
def test_a_range_reads_both_numbers_the_same_way(written, spoken):
    assert verbalize(written) == spoken


# 숫자만으로 된 묶음이 하이픈으로 셋 이상 이어지면(송장 번호 5551-0914-9121 같은 것) 범위가 아니라 번호다.
# 전화번호와 같이 묶음마다 한 자리씩 읽고 쉼표로 끊는지 본다. 고치기 전에는 뒤 두 묶음을 범위 규칙이 잡아
# "오천오백오십일 구백십사에서 구천백이십일 입니다"가 됐다. 묶음이 둘인 진짜 범위와 날짜는 그대로다.
@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("송장 번호는 5551-0914-9121입니다.", "송장 번호는 오오오일, 공구일사, 구일이일 입니다."),
        ("5552-1914-9230이고, 택배사는", "오오오이, 일구일사, 구이삼공 이고, 택배사는"),
        ("02-1234-5678로 전화", "공이, 일이삼사, 오육칠팔 로 전화"),
        ("1234-5678-9012-3456 맞아요?", "일이삼사, 오육칠팔, 구공일이, 삼사오육 맞아요?"),
        ("3-5일 안에", "삼에서 오 일 안에"),
        ("2~3개 남았어요", "두 개에서 세 개 남았어요"),
        ("2026-09-13에 주문", "이천이십육 년 구월 십삼일에 주문"),
    ],
)
def test_digit_only_numbers_in_three_or_more_groups_are_read_like_phone_numbers(written, spoken):
    assert verbalize(written) == spoken


# 커밋된 개발용 기준선 기록(*-baseline3)의 고객 발화 400개 이상을 모두 풀어 썼을 때 한글·공백·문장부호만
# 남는지 본다. 숫자나 영문이 하나라도 남으면 합성기가 읽는 방식이 정해지지 않으므로, 실제 측정 말뭉치 전체로
# 빠진 표기가 없는지 확인한다. 하니스 안내문(harness)은 고객 발화가 아니라서 뺀다.
def test_every_recorded_customer_utterance_becomes_pronounceable():
    """개발용 기준선 기록에는 시뮬레이터 고객의 서로 다른 발화가 500개쯤 있다."""
    records = sorted(REPORTS.glob("*-baseline3/episodes.jsonl"))
    assert records, "the committed baseline records are the corpus of this test"
    said = {
        m["content"]
        for line in records[0].read_text(encoding="utf-8").splitlines()
        for m in json.loads(line)["messages"]
        if m["role"] == "user" and not m.get("harness")
    }
    assert len(said) > 400
    for text in said:
        spoken = verbalize(text)
        assert re.fullmatch(r"[가-힣 .,?!~]*", spoken), (text, spoken)
        assert not re.search(r"\d|[A-Za-z]", spoken)


# 송장 번호를 범위로 읽던 풀어 쓰기로 합성된 기록의 발화 21개: (실행, 과제, 시도, 발화 순번). 5단계 1, 9단계
# 12, 11단계 8(9단계 V4 개발용과 같은 발화를 다시 합성한 것)이다. 이 기록들의 수치는 수정 전 코드로 낸
# 값이고, 이 발화가 든 에피소드만 다시 재는 계획이 docs/experiments.md 9단계 끝에 있다. 다시 잰 기록은 지금
# 코드로 풀어 쓴 것이라 이 목록에 들지 않는다. 원래 기록은 지우지 않으므로 목록도 그대로 남는다.
_V2_DEV_5 = "20260921-145634-qwen2.5-7b-instruct-P0-R0-G0-F0-V2-v2"
_V2_DEV_9 = "20261004-131932-qwen3.5-4b-P0-R0-G0-F0-V2-v2-4b-stage9"
_V4_DEV_9 = "20261004-154016-qwen3.5-4b-P0-R0-G0-F0-V4-v4-k0"
_V2_TEST_9 = "20261009-101009-qwen3.5-4b-P0-R0-G0-F0-V2-test-v2-4b-stage9"
_V4R_A_11 = "20261008-145519-v4r-a-dev"
_V4_DEV_UTTERANCES = [
    ("dev-002", 0, 3), ("dev-002", 2, 6), ("dev-002", 3, 3),
    ("dev-018", 0, 6), ("dev-018", 0, 9), ("dev-018", 0, 12), ("dev-018", 0, 15), ("dev-018", 0, 18),
]  # fmt: skip
SPELLED_BEFORE_THE_DIGIT_GROUP_FIX = sorted(
    [
        (_V2_DEV_5, "dev-018", 1, 7),
        (_V2_DEV_9, "dev-018", 1, 16),
        (_V2_DEV_9, "dev-018", 2, 9),
        (_V2_DEV_9, "dev-018", 2, 10),
        *[(_V4_DEV_9, *where) for where in _V4_DEV_UTTERANCES],
        *[(_V4R_A_11, *where) for where in _V4_DEV_UTTERANCES],
        (_V2_TEST_9, "test-025", 2, 16),
    ]
)


# 커밋된 음성 측정 기록(5·9단계의 episodes.jsonl voice 항목, 11단계의 utterances.jsonl)의 고객 발화를 지금
# 코드로 다시 풀어 쓰면, 위 21개를 빼고는 기록의 spoken과 모두 같아야 한다. 같으면 그 실행을 지금 코드로 다시
# 돌려도 합성기에 들어가는 글이 같아, 풀어 쓰기를 고쳐도 그 발화의 결과가 바뀌지 않는다는 뜻이다. 2026-10-13
# 범위 규칙을 고칠 때(docs/experiments.md 5단계 끝)는 바뀌는 발화가 0개였고, 같은 날 숫자 묶음 번호(송장 번호)
# 규칙을 더해 21개가 바뀌었다. 앞으로 풀어 쓰기를 고쳐 이 목록 밖의 발화가 바뀌면 이 테스트가 실패하므로, 그
# 고침이 바꾸는 기록과 수치를 먼저 적어야 한다. 시험용 기록도 읽지만 규칙을 만드는 데 쓰지 않도록 실패
# 메시지에는 발화 글 없이 위치만 낸다.
def test_recorded_voice_utterances_are_spelled_out_as_they_were():
    mismatched, seen = [], 0
    for path in sorted([*REPORTS.glob("*/episodes.jsonl"), *REPORTS.glob("*/utterances.jsonl")]):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            items = record.get("voice") or [] if path.name == "episodes.jsonl" else [record]
            for index, item in enumerate(items):
                seen += 1
                if verbalize(item["said"]) != item["spoken"]:
                    where = item.get("index", index)  # utterances.jsonl은 턴 번호를 담고 있다
                    mismatched.append((path.parent.name, record["task_id"], record["trial"], where))
    assert seen > 14_000, "the committed voice records are the corpus of this test"
    assert sorted(mismatched) == SPELLED_BEFORE_THE_DIGIT_GROUP_FIX


# 채널: 풀어 쓰기 → 합성 → 인식 → (정규화)를 한 번에 하고, 같은 입력의 결과는 캐시한다.
# -------------------------------------------------------------------- 채널


# 가짜 합성기: 풀어 쓴 글을 그대로 '소리' 바이트로 돌려주고, 어떤 글과 seed로 불렸는지 기록한다. 길이는 글자
# 수로 정한 가짜 값이다.
class FakeSpeaker:
    """'소리'는 말로 푼 글 그 자체라서, 가짜 인식기가 정해진 방식으로 잘못 들을 수 있다."""

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def describe(self):
        return {"tts": "fake"}

    def synthesize(self, spoken_text: str, seed: int = 0) -> Audio:
        self.calls.append((spoken_text, seed))
        return Audio(spoken_text.encode("utf-8"), seconds=len(spoken_text) / 8)


# 가짜 인식기: 실제 인식기처럼 금액을 숫자로 적고, '오 다시 일'(O-1)을 '5-1'로 잘못 적는다. 5단계 V1에서
# 'O-91010'이 매번 '5-91010'으로 적힌 것(README 5단계)을 흉내 내 정규화와 채널 동작을 결정적으로 시험한다.
# 소리가 비면 빈 글을 낸다.
class FakeListener:
    """인식기처럼 숫자를 숫자로 적고, 글자 O를 숫자 5로 잘못 듣는다."""

    def describe(self):
        return {"stt": "fake"}

    def transcribe(self, audio: bytes) -> str:
        text = audio.decode("utf-8")
        if not text.strip(" .,?!~"):
            return ""
        return (
            text.replace("오 다시 일, ", "5-1 ")
            .replace("오 다시 일", "5-1")
            .replace("삼만 팔천구백 원", "38,900원")
        )


# 채널 한 번의 결과(풀어 쓴 글, 인식된 글, 에이전트가 받는 글)와 캐시 동작을 본다.
# - 같은 글·같은 seed는 모델을 다시 부르지 않고 캐시에서 온다. seed가 다르면 다른 녹음으로 본다.
# - 캐시 파일을 새 채널이 읽으면 다음 실행도 캐시를 쓰고, 다른 캐시 파일은 공유되지 않는다.
# 합성 잡음을 seed로 고정하고 인식은 temperature 0만 쓰므로 캐시해도 측정값이 달라지지 않는다는
# 설계(docs/voice.md)의 전제다.
def test_the_channel_verbalises_synthesises_recognises_and_remembers(tmp_path):
    speaker = FakeSpeaker()
    cache = tmp_path / "cache" / "roundtrips.jsonl"
    channel = SpeechChannel(speaker, FakeListener(), cache_path=cache)
    heard = channel.hear("O-1 주문 38,900원 맞나요?", seed=3)
    assert heard.spoken == "오 다시 일 주문 삼만 팔천구백 원 맞나요?"
    assert heard.heard == heard.text == "5-1 주문 38,900원 맞나요?"
    assert (heard.cached, speaker.calls) == (False, [(heard.spoken, 3)])

    assert channel.hear("O-1 주문 38,900원 맞나요?", seed=3).cached  # 같은 말, 같은 seed: 모델 호출 없음
    assert not channel.hear("O-1 주문 38,900원 맞나요?", seed=4).cached  # seed가 다르면 다른 녹음이다
    assert len(speaker.calls) == 2

    again = SpeechChannel(FakeSpeaker(), FakeListener(), cache_path=cache)  # 나중 실행은 파일을 읽는다
    assert again.hear("O-1 주문 38,900원 맞나요?", seed=3).cached
    other = SpeechChannel(FakeSpeaker(), FakeListener(), cache_path=tmp_path / "other.jsonl")
    assert not other.hear("O-1 주문 38,900원 맞나요?", seed=3).cached


# 정규화는 에이전트가 받는 글(text)만 바꾸고 인식 원문(heard)은 그대로 남기는지, 채널 설명에 정규화 함수
# 이름이 남는지 본다. 원문이 남아야 V1과 V2의 인식 품질을 같은 기록에서 비교할 수 있다.
def test_the_normaliser_changes_only_what_the_agent_receives():
    channel = SpeechChannel(FakeSpeaker(), FakeListener(), normalizer=lambda text: text.replace("5-1", "O-1"))
    heard = channel.hear("O-1 취소해 주세요", seed=0)
    assert (heard.heard, heard.text) == ("5-1 취소해 주세요", "O-1 취소해 주세요")
    assert channel.describe()["normalizer"] == "<lambda>"


# '...'처럼 소리 낼 글자가 없는 발화는 모델을 부르지 않고 '들린 것 없음'(빈 글, 음성 길이 0)이 되는지 본다.
def test_nothing_pronounceable_means_nothing_heard():
    heard = SpeechChannel(FakeSpeaker(), FakeListener()).hear("...", seed=0)
    assert (heard.heard, heard.text, heard.audio_seconds) == ("", "", 0.0)


# 합성 결과 표본을 WAV 바이트로 싸는 wav_bytes가 RIFF 머리와 올바른 길이를 내고, 빈 표본은 오류로 거절하는지
# 본다. numpy는 voice 그룹에만 있어 없으면 건너뛴다.
def test_wav_bytes_is_a_wav_file():
    pytest.importorskip("numpy")  # 그것을 쓰는 모델처럼 `voice` 의존성 그룹에 들어 있다
    audio = wav_bytes([0.0, 0.5, -0.5, 1.5], 16_000)
    assert audio.wav[:4] == b"RIFF" and audio.seconds == pytest.approx(4 / 16_000)
    with pytest.raises(RuntimeError):
        wav_bytes([], 16_000)


# 서비스의 업로드 디코딩: 브라우저가 보낸 녹음을 60초 상한까지만 풀고 거절한다. 몇 kB의 FLAC가 몇 시간의
# 무음을 담을 수 있어, 다 푼 뒤에 길이를 재면 메모리가 크게 튄다(커밋 b747c1f, 93f2c64).
# -------------------------------------------------------------------- 업로드 디코딩 (서비스)


# PyAV로 시험용 녹음을 만든다: 무음, 또는 440 Hz 사인파에 잡음을 얹은 소리. 형식·코덱·채널·프레임 크기를 바꿔
# 실제 브라우저 업로드(webm/opus, wav)와 공격용 파일(아주 긴 무음 FLAC)을 모두 만든다. 잡음은 고정 seed(0)
# 난수라 매번 같은 파일이 나온다.
def encoded(seconds, rate, *, fmt="flac", codec="flac", layout="mono", tone=False, step=4096, options=None):
    """PyAV(`voice` 그룹)로 만든 녹음: 무음, 또는 잡음을 조금 얹은 440 Hz 사인파."""
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    buffer, total, noise = io.BytesIO(), int(seconds * rate), np.random.default_rng(0)
    with av.open(buffer, mode="w", format=fmt) as container:
        stream = container.add_stream(codec, rate=rate, options=options or {})
        stream.layout = layout
        for start in range(0, total, step):
            n = min(step, total - start)
            x = np.zeros(n)
            if tone:
                x = np.sin(2 * np.pi * 440 * np.arange(start, start + n) / rate) * 8000 + noise.normal(
                    0, 500, n
                )
            x = np.repeat(x, 2 if layout == "stereo" else 1).astype(np.int16).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(x, format="s16", layout=layout)
            frame.sample_rate, frame.pts = rate, start
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return buffer.getvalue()


# faster-whisper 모델 대역: 받은 표본 길이만 적고 빈 결과를 돌려준다. 모델 없이 '디코딩된 표본이 모델까지
# 갔는가'를 확인한다.
class StubWhisper:
    def __init__(self):
        self.lengths = []

    def transcribe(self, samples, **options):
        self.lengths.append(len(samples))
        return [], None


# 생성자(모델 적재, CUDA 경로 설정)를 건너뛰고 WhisperListener를 만든다. 서비스 설정처럼 max_seconds=60, VAD
# 켬으로 두고 모델만 대역으로 바꾼다.
def listener_without_a_model(model) -> WhisperListener:
    listener = object.__new__(WhisperListener)
    listener.max_seconds, listener.beam_size, listener.vad_filter, listener._model = 60, 5, True, model
    return listener


# 19 kB 안의 1시간 무음 FLAC가 전체 디코딩(decode_audio) 없이 AudioTooLongError로 거절되고, 파이썬 쪽 메모리
# 최고치가 50 MB 아래이며 모델에는 아무것도 가지 않는지 본다. decode_audio를 실패로 바꿔치기해 전체 디코딩
# 경로를 타면 바로 실패하게 했다. 다 풀면 약 580 MB가 든다(아래 줄 주석, 커밋 b747c1f).
def test_a_long_recording_is_refused_before_it_is_decoded(monkeypatch):
    """19 kB의 FLAC에 1시간 무음이 든다: 서비스는 다 풀지 않고 한도에서 멈춘다. tracemalloc은 파이썬에
    남은 표본만 보고 FFmpeg 자체 버퍼는 보지 못한다. 그것은 다음 테스트가 맡는다."""
    whisper_audio = pytest.importorskip("faster_whisper.audio")
    hour = encoded(3600, 8000, step=32768, options={"frame_size": "32768"})
    assert len(hour) < 50_000
    monkeypatch.setattr(
        whisper_audio, "decode_audio", lambda *a, **k: pytest.fail("decoded the whole upload")
    )
    model = StubWhisper()
    tracemalloc.start()
    try:
        with pytest.raises(AudioTooLongError):
            listener_without_a_model(model).transcribe(hour)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 50_000_000 and model.lengths == []  # 다 풀면 약 580 MB가 든다


# 표본 65,535개짜리 FLAC 프레임 하나를 아주 낮은 표본화율(1 Hz, 5 Hz)로 만들면 몇 시간짜리가 된다. 이 프레임이
# 리샘플러에 들어가기 전에 길이를 세어 거절하는지 본다. 리샘플러를 '불리면 실패'로 바꿔치기해 확인한다. 고치기
# 전에는 리샘플러 안에서 GB 단위 메모리를 썼고, tracemalloc은 FFmpeg 버퍼를 보지 못해 앞 테스트로는 잡히지
# 않았다(docstring, 커밋 93f2c64).
@pytest.mark.parametrize("rate", [1, 5])
def test_an_hours_long_frame_is_refused_before_it_is_resampled(monkeypatch, rate):
    """5 Hz 표본 65,535개짜리 8 kB FLAC 프레임 하나는 13,107초다: 16 kHz로 리샘플하면 FFmpeg 안에서 4 GB가
    들었다(1 Hz면 2 GB였고 아무것도 나오지 않아 받아들여졌다). 그래서 길이를 먼저 센다."""
    av = pytest.importorskip("av")
    data = encoded(65_535 / rate, rate, step=65_535, options={"frame_size": "65535"})
    assert len(data) < 10_000

    class NoResampling:
        def __init__(self, **options):
            pass

        def resample(self, frame):
            pytest.fail("a frame went to the resampler")

    monkeypatch.setattr(av.audio.resampler, "AudioResampler", NoResampling)
    with pytest.raises(AudioTooLongError):
        speech.decode_limited(data, 60)


# 상한 안의 정상 녹음(44.1 kHz 스테레오 FLAC, 48 kHz WebM/Opus, 상한과 같은 60초 WAV)에서 길이 제한 디코더가
# faster-whisper의 decode_audio와 표본까지 같은 결과를 내는지 본다. 제한을 넣으면서 인식 입력이 달라지지
# 않았다는 확인이다.
@pytest.mark.parametrize(
    "recording",
    [
        {"seconds": 20, "rate": 44_100, "layout": "stereo", "tone": True},
        {"seconds": 20, "rate": 48_000, "fmt": "webm", "codec": "libopus", "tone": True, "step": 960},
        {"seconds": 60, "rate": 16_000, "fmt": "wav", "codec": "pcm_s16le", "tone": True},  # 딱 한도
    ],
    ids=["flac-stereo-44k", "webm-opus-48k", "wav-60s"],
)
def test_the_limited_decoder_gives_what_decode_audio_gives(recording):
    np = pytest.importorskip("numpy")
    decode_audio = pytest.importorskip("faster_whisper.audio").decode_audio
    data = encoded(**recording)
    expected = decode_audio(io.BytesIO(data), sampling_rate=LISTEN_RATE)
    assert np.array_equal(speech.decode_limited(data, 60), expected)


# 상한 안의 2초 녹음은 16 kHz로 디코딩되어 모델에 2초 × 16,000개 표본이 그대로 가는지 본다.
def test_a_recording_under_the_limit_reaches_the_model():
    model = StubWhisper()
    assert listener_without_a_model(model).transcribe(encoded(2, 16_000, tone=True)) == ""
    assert model.lengths == [2 * LISTEN_RATE]


# 에피소드 안에서: 채널은 고객 → 에이전트 방향에만 끼어든다.
# -------------------------------------------------------------------- 에피소드 안에서


# V1 대화 순환(_converse)에서 에이전트만 잘못 들은 글('5-1 취소해 주세요')을 받고, 시뮬레이터의 기록에는
# 하려던 말('O-1 취소해 주세요')이 남는지 본다. 소리 낼 것이 없는 발화('...')는 NOTHING_HEARD로 바뀌어
# 에이전트에 가고, 음성 기록(log)에는 발화마다 원래 글과 받은 글이 남는다.
# 잘못 들은 번호(5-1)로 한 취소는 없는 주문이라 실패해 DB가 그대로인 것까지 확인한다. 시뮬레이터가 자기 말을
# 기억해야 '에이전트가 잘못 들었다'는 차이만 남는다(docs/voice.md).
def test_only_the_agent_hears_through_the_channel(tiny_engine):
    engine = db.memory_engine(tiny_engine)
    ctx = ToolContext(now=NOW)
    state = new_state("system prompt")
    provider = ScriptedProvider(
        [
            ToolCall("toy_cancel_order", {"order_id": "5-1"}),
            "5-1 주문은 찾을 수 없습니다. 번호를 다시 알려 주세요.",
            "잘 들리지 않았습니다. 다시 말씀해 주세요.",
        ]
    )
    user = ScriptedUser(["O-1 취소해 주세요", "...", "###STOP###"])
    log: list[dict] = []
    # _converse는 과제에서 id(seed 유도용)만 쓰므로 id만 가진 가짜 과제 객체로 충분하다.
    task = type("T", (), {"id": "toy-1"})()
    ending = _converse(
        state,
        user,
        task,
        0,
        RunConfig(voice="V1"),
        provider,
        TOY_REGISTRY,
        lambda name, args: execute(TOY_REGISTRY, engine, ctx, name, args),
        SpeechChannel(FakeSpeaker(), FakeListener()),
        log,
    )
    assert ending == "user_stop"
    agent_heard = [m.content for m in state.messages if m.role == "user"]
    assert agent_heard == ["5-1 취소해 주세요", NOTHING_HEARD]
    assert [m.content for m in user.messages if m.role == "assistant"][
        0
    ] == "O-1 취소해 주세요"  # 고객이 뜻한 그대로 남는다
    assert [(u["said"], u["text"]) for u in log] == [("O-1 취소해 주세요", "5-1 취소해 주세요"), ("...", "")]
    assert db.dump_db(engine)["orders"][0]["status"] == "paid"  # 잘못 들은 번호로는 아무것도 취소되지 않았다


# 기본값은 V0(글)이고, 글 실행의 한 턴은 채널 없이 그대로 도는지 본다. 기존 글 측정과 비교 가능성을 지키는
# 확인이다.
def test_text_runs_are_untouched(tiny_engine):
    state = new_state("system prompt")
    result = agent_turn(
        state,
        "O-1 취소해 주세요",
        provider=ScriptedProvider(["네."]),
        registry=TOY_REGISTRY,
        run_tool=lambda name, args: None,
        config=RunConfig(),
    )
    assert RunConfig().voice == "V0" and result.reply == "네."


# 지표: 기록에서 글자 오류율, 엔티티 생존율, 본인 확인 성공률과 짝지은 구간을 다시 계산한다(모델 없음).
# -------------------------------------------------------------------- metrics


# 고객 발화에서 도구 인자가 될 엔티티(이메일, 전화번호, 주문 번호, 금액, 날짜)를 뽑고, 인식된 글에서 각각이
# 도구가 쓸 수 있는 꼴로 살아남았는지 본다. 전화번호는 숫자 묶음이 달라도(010 0000 9103) 살아남은 것으로,
# 금액은 쉼표·공백 차이를 무시한다. 이메일의 골뱅이가 글자로 적히거나 O가 5로 적히면 잃은 것이다.
def test_entities_and_their_survival():
    said = "엄다현이고 010-0000-9103, dahyun.eom@example.com입니다. O-91010 주문 38,900원, 9월 14일이요."
    found = metrics.entities(said)
    assert found == [
        ("email", "dahyun.eom@example.com"),
        ("phone", "010-0000-9103"),
        ("id", "O-91010"),
        ("amount", "38,900원"),
        ("date", "9월 14일"),
    ]
    heard = "엄다현이고 010 0000 9103 dahyun.eom골뱅이그젠플 닷컴입니다 5-91010 주문 38900 원 9월14일이요"
    assert [metrics.survived(kind, entity, heard) for kind, entity in found] == [
        False,
        True,
        False,
        True,
        True,
    ]


# 글자 오류율은 띄어쓰기·문장부호를 지운 꼴(squeeze)로 비교하고, 편집 거리는 글자 단위 레벤슈타인 거리인지
# 본다. 띄어쓰기 차이까지 오류로 세면 인식 품질과 무관한 차이가 지표를 부풀린다.
def test_cer_ignores_spacing_and_punctuation():
    assert metrics.squeeze("네, 진행해 주세요.") == metrics.squeeze("네 진행해주세요")
    assert metrics.edit_distance("주문번호", "주문번오") == 1 and metrics.edit_distance("", "가나") == 2


# 지표 테스트용 최소 에피소드 기록: 본인 확인 도구(find_customer)의 성공 여부와 음성 발화 기록만 담는다.
def episode(task_id, identified, voice=()):
    calls = [{"name": "find_customer", "ok": identified}, {"name": "get_order", "ok": True}]
    return {"task_id": task_id, "status": "completed", "tool_calls": calls, "voice": list(voice)}


# analyze voice가 내는 표(metrics.voice_table)를 본다: 글 실행은 '(텍스트)'로 표시하고, 음성 실행은 발화
# 수·그대로 전달 비율·엔티티 생존(0/2)을 내며, 본인 확인 성공률을 첫 실행과 과제별로 짝지어 차이와 구간을
# 낸다. worst는 엔티티를 잃은 발화를 중복 없이 모아 정규화 규칙을 만드는 재료로 보여 준다. 과제 집합이 다르면
# 짝지은 차이를 내지 않는다.
def test_the_voice_table_pairs_runs_by_task():
    heard = {"said": "O-1 취소", "heard": "5-1 취소", "text": "5-1 취소", "audio_seconds": 2.0,
             "tts_ms": 700.0, "stt_ms": 900.0, "cached": False}  # fmt: skip
    clean = {**heard, "said": "네", "heard": "네.", "text": "네."}
    runs = {
        "text": [episode("a", True), episode("a", True), episode("b", True)],
        "voice": [
            episode("a", False, [heard, clean]),
            episode("a", True, [clean]),
            episode("b", False, [heard]),
        ],
    }
    table = metrics.voice_table(runs)
    assert "| text | (텍스트) |" in table
    assert "| voice | 4 | 2 (50.0%) | 0 |" in table and "0/2 (0%)" in table
    assert "| text (기준) | 100.0% | |" in table and "| voice | 25.0% | -75.0%p [" in table
    assert "lost : ['O-1']" in metrics.worst(runs["voice"]) and "(1 distinct" in metrics.worst(runs["voice"])
    with pytest.raises(ValueError, match="same tasks"):
        metrics.paired_mean_difference({"a": 1.0}, {"b": 1.0})


# speech.py의 맨 위 import에 torch·melo·faster_whisper·numpy·av가 없는지 본다. 이 라이브러리들은 선택 의존성
# 그룹 voice에만 있고, CI와 본체 테스트는 그 그룹 없이 돈다. 음성 그룹이 깔린 환경에서만 테스트하다 CI가
# 실제로 깨진 적이 있어(커밋 11a148e) 무거운 import는 함수 안에서만 하게 지킨다.
def test_importing_the_voice_package_needs_no_speech_library():
    source = Path(metrics.__file__).with_name("speech.py").read_text(encoding="utf-8")
    top_level = [line for line in source.splitlines() if re.match(r"(import|from) ", line)]
    assert not [line for line in top_level if re.search(r"torch|melo|faster_whisper|numpy|\bav\b", line)]


# V2 정규화: 인식된 글의 '표기'만 고치는 규칙 네 개. 개발용 V1 기록만 보고 만들었다(docs/experiments.md
# 5단계).
# -------------------------------------------------------------------- V2: 정규화


# 개발용 V1의 실제 받아쓰기(왼쪽)가 고쳐지는 꼴(오른쪽)을 본다: 5-번호 → O-번호, 하이픈·글자·점으로 적힌
# 골뱅이와 도메인 → @example.com, 쉼표로 끊긴 전화번호 → 하이픈, '12만 원'·'9만 7,600원' → 아라비아 숫자 금액.
# docs/experiments.md 5단계 "V2의 정규화 규칙" 표의 네 규칙이다. 이메일 규칙은 이 가상 쇼핑몰 고객의 도메인이
# 모두 example.com이라는 데 기댄다(같은 절).
@pytest.mark.parametrize(
    ("heard", "repaired"),
    [  # 왼쪽은 모두 V1 개발용 실행의 받아쓰기다
        ("주문번호는 5-91001이에요.", "주문번호는 O-91001이에요."),
        ("주문번호가 5에서 9100이 아니라", "주문번호가 O-9100이 아니라"),
        ("이메일은 yunseo.heo-example.com입니다.", "이메일은 yunseo.heo@example.com입니다."),
        ("sunghon.pae골뱅이그젠플.com입니다.", "sunghon.pae@example.com입니다."),
        ("yunseo.heo-golbenie-example.com입니다.", "yunseo.heo@example.com입니다."),
        ("minseo.chae-e-example.com입니다.", "minseo.chae@example.com입니다."),
        ("resunghon.pae 골뱅이 이그젠플 닷컴입니다.", "resunghon.pae@example.com입니다."),
        ("junghon.byun.example.com입니다. 5-91019죠.", "junghon.byun@example.com입니다. O-91019죠."),
        ("010, 0000, 9103으로 가입했어요.", "010-0000-9103으로 가입했어요."),
        ("결제 금액 12만 원을 받아 적었습니다.", "결제 금액 120,000원을 받아 적었습니다."),
        ("9만 7,600원이 맞나요?", "97,600원이 맞나요?"),
    ],
)
def test_the_normaliser_repairs_notation(heard, repaired):
    assert normalize_heard(heard) == repaired


# 고치면 안 되는 글은 그대로인지 본다: 평범한 말, 주문 번호가 아닌 '5-5 사이즈'·'5-10일', 숫자 하나를 잃은
# 전화번호, 잘못 들은 이름, 이미 쓴 꼴인 값, 아무도 그렇게 말하지 않는 금액. 소리에서 잃은 것은 규칙으로
# 되찾을 수 없으므로 추측해 고치지 않는다(normalize.py docstring).
@pytest.mark.parametrize(
    "heard",
    [
        "네, 진행해 주세요.",
        "5-5 사이즈로 5개 주문했고 5-10일 안에 받고 싶어요.",  # 주문 번호가 아니다
        "010-0000-921입니다.",  # 소리에서 숫자 하나를 잃었다: 되살릴 근거가 없다
        "이름은 배성분입니다.",  # 잘못 들은 이름은 잘못 들은 채로 남는다
        "주문번호는 O-91001이고 38,900원입니다.",  # 이미 글 표기다
        "12만 34,000원",  # 아무도 그렇게 말하지 않는 수
    ],
)
def test_the_normaliser_leaves_the_rest_alone(heard):
    assert normalize_heard(heard) == heard


# 정규화를 두 번 해도 결과가 같은지(멱등) 본다. 서비스처럼 고친 글이 다시 들어와도 값이 망가지지 않아야 한다.
def test_the_normaliser_is_idempotent_on_its_own_output():
    once = normalize_heard("junghon.byun.example.com입니다. 5-91019죠. 010, 0000, 9103")
    assert normalize_heard(once) == once


# 실제 마이크: 사람이 소리 내어 부른 번호를 숫자로 되돌린다(spoken.py). 서비스에서만 쓰고 측정한 V2와는
# 별개다.
# -------------------------------------------------------------------- 실제 마이크: 소리 내어 부른 번호


# 실제 마이크 시험에서 나온 받아쓰기(첫 줄)처럼 한글로 적힌 휴대전화 번호와 '오 다시 …' 주문 번호가 숫자
# 표기로 돌아오는지 본다. 조사 '이고'를 숫자 2로 읽지 않도록 자릿수를 정확히 세고, 한 자리가 모자란 번호나
# 일상 문장('이사 오고 일이 많아서')은 바꾸지 않는다.
@pytest.mark.parametrize(
    ("heard", "written"),
    [
        ("강정월구 공일공 공공공공 육일팔구입니다", "강정월구 010-0000-6189입니다"),  # 실제 시험에서 나온 것
        (
            "전화는 공일공공공공공육일팔구이고요",
            "전화는 010-0000-6189이고요",
        ),  # "이고"는 2가 아니라 조사다
        ("010 공공공공 6189요", "010-0000-6189요"),
        ("주문은 오 다시 구일공일공이에요", "주문은 O-91010이에요"),
        ("주문번호는 O-91010입니다", "주문번호는 O-91010입니다"),
        ("공일공 공공공공 육일팔", "공일공 공공공공 육일팔"),  # 한 자리 모자람: 적을 수 있는 번호가 아니다
        ("이사 오고 일이 많아서 구이를 먹었어요", "이사 오고 일이 많아서 구이를 먹었어요"),
    ],
)
def test_numbers_read_aloud_are_written_down(heard, written):
    assert spoken_to_written(heard) == written


# verbalize로 풀어 쓴 번호를 spoken_to_written이 원래 글로 되돌리는지(왕복) 본다. 두 함수가 같은 읽기 규칙을
# 공유한다는 확인이다.
@pytest.mark.parametrize("written", ["010-0000-6189", "010-1234-5678", "O-91010", "O-10097"])
def test_what_verbalize_spells_out_comes_back(written):
    assert spoken_to_written(verbalize(written)) == written
