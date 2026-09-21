"""The runner end to end with --voice V1: arguments, channel, run id, manifest and the episode record.

The model and the speech models are fakes; everything between them is the real code path of a measurement.
"""

from __future__ import annotations

import json

from test_voice import FakeListener, FakeSpeaker

from support_agent import run
from support_agent.chat import ChatProvider, ChatResponse
from support_agent.voice.channel import SpeechChannel


class FakeOllama(ChatProvider):
    """The customer (called without tools) asks about an order and then stops; the agent only acknowledges."""

    def __init__(self, model: str, **_options):
        self.model = model
        self.customer_lines = ["O-1 주문 38,900원 맞나요?", "###STOP###"]

    def preload(self) -> float:
        return 0.0

    def describe(self):
        return {"provider": "fake", "model": self.model}

    def chat(self, messages, tools=(), *, temperature=0.0, seed=None, max_tokens=1024):
        if tools:
            return ChatResponse("네, 확인해 보겠습니다. 성함을 알려 주세요.")
        return ChatResponse(self.customer_lines.pop(0))


def test_a_voice_run_records_what_was_said_and_what_was_heard(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run, "OllamaProvider", FakeOllama)
    monkeypatch.setattr(run, "gpu_used_by_others_mib", lambda provider: None)
    monkeypatch.setattr(run, "OUTPUTS", tmp_path)
    built = []

    def build_channel(config, tts_device, stt_device):
        built.append((config.voice, tts_device, stt_device))
        return SpeechChannel(FakeSpeaker(), FakeListener(), cache_path=tmp_path / "cache.jsonl")

    monkeypatch.setattr(run, "build_channel", build_channel)
    argv = ["run", "--tasks", "smoke", "--task-id", "smoke-lookup-01", "--voice", "V1", "--tts-device", "cpu"]
    monkeypatch.setattr("sys.argv", [*argv, "--label", "unit"])
    run.main()

    assert built == [("V1", "cpu", "cuda")]
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert run_dir.name.endswith("-P0-R0-G0-F0-V1-unit")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["voice"] == "V1"
    assert manifest["voice_channel"] == {"tts": "fake", "stt": "fake", "normalizer": None}
    (line,) = (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    episode = json.loads(line)
    assert episode["termination"] == "user_stop"
    (utterance,) = episode["voice"]
    assert utterance["said"] == "O-1 주문 38,900원 맞나요?"
    assert utterance["spoken"] == "오 다시 일 주문 삼만 팔천구백 원 맞나요?"
    assert utterance["text"] == "5-1 주문 38,900원 맞나요?"  # the fake listener mishears O as 5
    heard_by_agent = [m["content"] for m in episode["messages"] if m["role"] == "user"]
    meant_by_customer = [m["content"] for m in episode["user_messages"] if m["role"] == "assistant"]
    assert heard_by_agent == [utterance["text"]] and meant_by_customer[0] == utterance["said"]
    assert "smoke-lookup-01 #0" in capsys.readouterr().out


def test_a_text_run_has_no_channel_and_keeps_its_old_name(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OllamaProvider", FakeOllama)
    monkeypatch.setattr(run, "gpu_used_by_others_mib", lambda provider: None)
    monkeypatch.setattr(run, "OUTPUTS", tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["run", "--tasks", "smoke", "--task-id", "smoke-lookup-01", "--label", "unit"]
    )
    run.main()
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert run_dir.name.endswith("-P0-R0-G0-F0-unit")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["voice_channel"] is None and manifest["config"]["voice"] == "V0"
    episode = json.loads((run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert episode["voice"] == []
