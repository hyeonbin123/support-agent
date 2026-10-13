"""The runner end to end with --voice V1: arguments, channel, run id, manifest and the episode record.

The model and the speech models are fakes; everything between them is the real code path of a measurement.
"""

# [테스트 묶음] 실행기(run.main)를 --voice V1로 처음부터 끝까지 돌려, 명령줄 인자 → 음성 채널 만들기 → 실행 id
# → manifest → 에피소드 기록이 이어지는지 확인한다. 텍스트 실행이 예전 그대로인지도 본다.
# [왜 중요한가] 5단계 측정은 이 경로로 기록을 남긴다. 고객이 하려던 말(said), 말로 푼 글(spoken), 에이전트가
# 받은 글(text)이 기록에 남아야 analyze voice가 글자 오류율과 엔티티 생존을 다시 계산할 수 있다. 에이전트만
# 잘못 듣고 시뮬레이터는 자기가 쓴 글을 기억한다는 설계(docs/voice.md)도 기록으로 확인한다.
# [방법] Ollama와 음성 모델은 가짜로 바꾸고(monkeypatch), 그 사이의 실행기 코드는 실제 측정과 같은 경로를
# 탄다. GPU 확인과 출력 폴더도 테스트용으로 바꾼다.

from __future__ import annotations

# test_voice의 가짜 합성기·인식기(인식기는 O를 5로 잘못 듣는다)와 실제 음성 채널(SpeechChannel)을 쓴다.
import json

from test_voice import FakeListener, FakeSpeaker

from support_agent import run
from support_agent.chat import ChatProvider, ChatResponse
from support_agent.voice.channel import SpeechChannel


# 실행기가 만드는 OllamaProvider 자리에 넣는 가짜. 도구 목록 없이 불리면 고객(시뮬레이터)이라 주문 금액을 묻고
# 다음에 ###STOP###으로 끝내고, 도구와 함께 불리면 상담원이라 확인하겠다고만 답한다. preload와 describe도
# 실행기가 부르므로 흉내 낸다. tests/test_stage9.py도 이 클래스를 가져다 쓴다.
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


# V1 실행: 채널은 설정한 장치로 한 번 만들어지고(합성 cpu, 인식 기본 cuda), 실행 id 끝에 구성
# 이름(-P0-R0-G0-F0-V1-라벨)이 붙고, manifest에 음성 조건과 채널 정보가 남는다. 에피소드 기록의 voice 항목에는
# 고객이 쓴 글, 말로 푼 글, 인식된 글이 남는다.
# 에이전트가 받은 고객 말은 인식된 글('5-1 …')이고, 시뮬레이터 쪽 기록에는 자기가 쓴 원래 글이 그대로 남아야
# 한다.
def test_a_voice_run_records_what_was_said_and_what_was_heard(tmp_path, monkeypatch, capsys):
    # 실제 모델과 GPU 없이 돌리려고 바꾸는 것: Ollama 제공자, 다른 프로그램의 GPU 사용량 확인(nvidia-smi),
    # 출력 폴더, 음성 채널 만들기.
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


# 텍스트 실행(V0)은 채널이 없고(voice_channel None), 실행 id에 음성 조건이 붙지 않아 5단계 이전 기록과 이름
# 형식이 같다. 기록의 voice 항목은 빈 목록이다. 예전 기록과 짝지어 비교할 수 있게 하려는 것이다.
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
