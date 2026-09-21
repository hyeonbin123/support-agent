"""Stage 5: spoken forms, the speech channel (with fake speech models) and its metrics."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from conftest import NOW
from toy_tools import TOY_REGISTRY

from support_agent import db
from support_agent.agent import agent_turn, new_state
from support_agent.chat import ScriptedProvider, ToolCall
from support_agent.config import RunConfig
from support_agent.episode import NOTHING_HEARD, _converse
from support_agent.paths import REPORTS
from support_agent.toolkit import ToolContext, execute
from support_agent.user_sim import ScriptedUser
from support_agent.voice import metrics
from support_agent.voice.channel import SpeechChannel
from support_agent.voice.speech import Audio, wav_bytes
from support_agent.voice.verbalize import native, sino, verbalize

# -------------------------------------------------------------------- spoken forms


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


def test_native_numerals_up_to_99():
    assert [native(n) for n in (1, 2, 3, 4, 10, 20, 21, 37, 99)] == [
        "한", "두", "세", "네", "열", "스무", "스물한", "서른일곱", "아흔아홉",
    ]  # fmt: skip
    assert native(100) == "백"


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


def test_every_recorded_customer_utterance_becomes_pronounceable():
    """The development baseline holds about 500 distinct utterances of the simulated customers."""
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


# -------------------------------------------------------------------- the channel


class FakeSpeaker:
    """The 'audio' is the spoken text itself, so the fake listener can mishear it in a known way."""

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def describe(self):
        return {"tts": "fake"}

    def synthesize(self, spoken_text: str, seed: int = 0) -> Audio:
        self.calls.append((spoken_text, seed))
        return Audio(spoken_text.encode("utf-8"), seconds=len(spoken_text) / 8)


class FakeListener:
    """Writes numbers the way a recogniser does, and mishears the letter O as the digit 5."""

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


def test_the_channel_verbalises_synthesises_recognises_and_remembers(tmp_path):
    speaker = FakeSpeaker()
    cache = tmp_path / "cache" / "roundtrips.jsonl"
    channel = SpeechChannel(speaker, FakeListener(), cache_path=cache)
    heard = channel.hear("O-1 주문 38,900원 맞나요?", seed=3)
    assert heard.spoken == "오 다시 일 주문 삼만 팔천구백 원 맞나요?"
    assert heard.heard == heard.text == "5-1 주문 38,900원 맞나요?"
    assert (heard.cached, speaker.calls) == (False, [(heard.spoken, 3)])

    assert channel.hear("O-1 주문 38,900원 맞나요?", seed=3).cached  # same words, same seed: no model call
    assert not channel.hear("O-1 주문 38,900원 맞나요?", seed=4).cached  # another seed is another recording
    assert len(speaker.calls) == 2

    again = SpeechChannel(FakeSpeaker(), FakeListener(), cache_path=cache)  # a later run reads the file
    assert again.hear("O-1 주문 38,900원 맞나요?", seed=3).cached
    other = SpeechChannel(FakeSpeaker(), FakeListener(), cache_path=tmp_path / "other.jsonl")
    assert not other.hear("O-1 주문 38,900원 맞나요?", seed=3).cached


def test_the_normaliser_changes_only_what_the_agent_receives():
    channel = SpeechChannel(FakeSpeaker(), FakeListener(), normalizer=lambda text: text.replace("5-1", "O-1"))
    heard = channel.hear("O-1 취소해 주세요", seed=0)
    assert (heard.heard, heard.text) == ("5-1 취소해 주세요", "O-1 취소해 주세요")
    assert channel.describe()["normalizer"] == "<lambda>"


def test_nothing_pronounceable_means_nothing_heard():
    heard = SpeechChannel(FakeSpeaker(), FakeListener()).hear("...", seed=0)
    assert (heard.heard, heard.text, heard.audio_seconds) == ("", "", 0.0)


def test_wav_bytes_is_a_wav_file():
    pytest.importorskip("numpy")  # comes with the `voice` dependency group, like the models that use it
    audio = wav_bytes([0.0, 0.5, -0.5, 1.5], 16_000)
    assert audio.wav[:4] == b"RIFF" and audio.seconds == pytest.approx(4 / 16_000)
    with pytest.raises(RuntimeError):
        wav_bytes([], 16_000)


# -------------------------------------------------------------------- in an episode


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
    ] == "O-1 취소해 주세요"  # as it was meant
    assert [(u["said"], u["text"]) for u in log] == [("O-1 취소해 주세요", "5-1 취소해 주세요"), ("...", "")]
    assert db.dump_db(engine)["orders"][0]["status"] == "paid"  # the misheard number cancelled nothing


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


# -------------------------------------------------------------------- metrics


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


def test_cer_ignores_spacing_and_punctuation():
    assert metrics.squeeze("네, 진행해 주세요.") == metrics.squeeze("네 진행해주세요")
    assert metrics.edit_distance("주문번호", "주문번오") == 1 and metrics.edit_distance("", "가나") == 2


def episode(task_id, identified, voice=()):
    calls = [{"name": "find_customer", "ok": identified}, {"name": "get_order", "ok": True}]
    return {"task_id": task_id, "status": "completed", "tool_calls": calls, "voice": list(voice)}


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


def test_importing_the_voice_package_needs_no_speech_library():
    source = Path(metrics.__file__).with_name("speech.py").read_text(encoding="utf-8")
    top_level = [line for line in source.splitlines() if re.match(r"(import|from) ", line)]
    assert not [line for line in top_level if re.search(r"torch|melo|faster_whisper|numpy", line)]
