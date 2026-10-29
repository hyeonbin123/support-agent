"""--voice V1로 실행기를 처음부터 끝까지 돌린다: 인자, 채널, 실행 id, manifest, 에피소드 기록.

LLM과 음성 모델은 가짜이고, 그 사이의 모든 것은 측정의 실제 코드 경로다.
"""

# [테스트 묶음] 실행기(run.main)를 --voice V1로 처음부터 끝까지 돌려, 명령줄 인자 → 음성 채널 만들기 → 실행 id
# → manifest → 에피소드 기록이 이어지는지 확인한다. 텍스트 실행이 예전 그대로인지도 본다.
# [왜 중요한가] 5단계 측정은 이 경로로 기록을 남긴다. 고객이 하려던 말(said), 말로 푼 글(spoken), 에이전트가
# 받은 글(text)이 기록에 남아야 analyze voice가 글자 오류율과 엔티티 생존을 다시 계산할 수 있다. 에이전트만
# 잘못 듣고 시뮬레이터는 자기가 쓴 글을 기억한다는 설계(docs/voice.md)도 기록으로 확인한다.
# [방법] Ollama와 음성 모델은 가짜로 바꾸고(monkeypatch), 그 사이의 실행기 코드는 실제 측정과 같은 경로를
# 탄다. GPU 확인과 출력 폴더도 테스트용으로 바꾼다.
# 2026-10-29부터는 음성 실행 manifest의 voice_extra(짝짓기 검사에 쓰지 않는 패키지 버전·음성 모델 스냅숏)도
# 본다. 패키지 메타데이터와 Hugging Face 캐시는 대역이라 음성 그룹이 없는 환경(CI)에서도 돈다.

from __future__ import annotations

# test_voice의 가짜 합성기·인식기(인식기는 O를 5로 잘못 듣는다)와 실제 음성 채널(SpeechChannel)을 쓴다.
# importlib.util·inspect: 음성 라이브러리를 불러오지 않고 소스 위치만 찾기, WhisperListener의 기본 모델 읽기.
# sys·types: huggingface_hub 자리에 대역 모듈을 넣기.
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest
from test_voice import FakeListener, FakeSpeaker

from support_agent import run
from support_agent.chat import ChatProvider, ChatResponse
from support_agent.voice import speech
from support_agent.voice.channel import SpeechChannel


# 실행기가 만드는 OllamaProvider 자리에 넣는 가짜. 도구 목록 없이 불리면 고객(시뮬레이터)이라 주문 금액을 묻고
# 다음에 ###STOP###으로 끝내고, 도구와 함께 불리면 상담원이라 확인하겠다고만 답한다. preload와 describe도
# 실행기가 부르므로 흉내 낸다. tests/test_stage9.py도 이 클래스를 가져다 쓴다.
class FakeOllama(ChatProvider):
    """고객(도구 없이 불림)은 주문을 묻고 끝내며, 상담원은 확인하겠다고만 답한다."""

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
    assert utterance["text"] == "5-1 주문 38,900원 맞나요?"  # 가짜 인식기가 O를 5로 잘못 듣는다
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
    assert manifest["voice_extra"] is None  # 음성 채널이 없으면 음성 쪽 기록도 없다
    episode = json.loads((run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert episode["voice"] == []


# ==================================================================== 비교하지 않는 음성 기록(voice_extra)

# 대역 HF 캐시가 저장소마다 돌려줄 스냅숏(커밋) 이름. 키는 음성 측정이 받는 세 저장소다: MeloTTS 한국어
# 체크포인트, melo 한국어 앞단의 BERT, faster-whisper large-v3-turbo. tests/test_stage11.py도 쓴다.
FAKE_REVISIONS = {
    "myshell-ai/MeloTTS-Korean": "1" * 40,
    "kykim/bert-kor-base": "2" * 40,
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo": "3" * 40,
}


# 가짜 음성 메타데이터를 깐다: 설치된 패키지 버전은 모두 '<이름>-fake', huggingface_hub는 캐시
# 조회(try_to_load_from_cache)만 있는 대역 모듈(…/snapshots/<FAKE_REVISIONS>/<파일> 경로를 돌려줌)이다.
# 음성 그룹이 없는 환경에는 huggingface_hub가 없으므로 sys.modules에 직접 넣는다(monkeypatch가 테스트 뒤
# 되돌림). 돌려주는 값은 manifest의 voice_extra로 기대하는 것이다. 패키지 다섯은 코드에서 가져오지 않고
# 여기에 적어 둔다(목록이 바뀌면 이 테스트가 먼저 깨진다): MeloTTS가 정확히 고정하는 transformers, 그와
# 함께 묶이는 tokenizers·huggingface-hub, numpy·librosa.
def fake_voice_metadata(monkeypatch, cache_root: Path) -> dict:
    monkeypatch.setattr(speech, "version", lambda name: f"{name}-fake")
    hub = types.ModuleType("huggingface_hub")

    def try_to_load_from_cache(repo_id, filename, cache_dir=None, revision=None, repo_type=None):
        repo = cache_root / ("models--" + repo_id.replace("/", "--"))
        return str(repo / "snapshots" / FAKE_REVISIONS[repo_id] / filename)

    hub.try_to_load_from_cache = try_to_load_from_cache  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    packages = ("transformers", "tokenizers", "huggingface-hub", "numpy", "librosa")
    return {"versions": {name: f"{name}-fake" for name in packages}, "model_revisions": dict(FAKE_REVISIONS)}


# 음성 실행의 manifest에 voice_extra(묶인 패키지 버전과 음성 모델 스냅숏)가 남고, 짝짓기 검사가 비교하는
# voice_channel과 versions는 예전 그대로인지 본다. transformers 같은 이름이 versions나 채널 설명에 들어가면 그
# 키가 없는 기존 음성 기록 전부와 '다름'이 되므로, 두 항목의 모양을 여기서 고정한다.
def test_a_voice_run_records_the_unpaired_versions_and_model_revisions(tmp_path, monkeypatch):
    expected = fake_voice_metadata(monkeypatch, tmp_path / "hf")
    monkeypatch.setattr(run, "OllamaProvider", FakeOllama)
    monkeypatch.setattr(run, "gpu_used_by_others_mib", lambda provider: None)
    monkeypatch.setattr(run, "OUTPUTS", tmp_path)
    monkeypatch.setattr(
        run, "build_channel", lambda config, tts, stt: SpeechChannel(FakeSpeaker(), FakeListener())
    )
    argv = ["run", "--tasks", "smoke", "--task-id", "smoke-lookup-01", "--voice", "V1", "--label", "unit"]
    monkeypatch.setattr("sys.argv", argv)
    run.main()
    (run_dir,) = (tmp_path / "runs").iterdir()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["voice_extra"] == expected
    assert manifest["voice_channel"] == {"tts": "fake", "stt": "fake", "normalizer": None}
    base = {"python", "sqlalchemy", "pydantic", "httpx", "pyyaml"}
    assert set(manifest["versions"]) == base | {"faster-whisper", "ctranslate2", "melotts", "torch"}


# 진짜 huggingface_hub로, 가짜 캐시 폴더에서 저장소마다 main이 가리키는 스냅숏 이름을 읽는지 본다(음성 그룹이
# 있는 환경에서만). MeloTTS: 스냅숏이 둘이면 refs/main이 가리키는 쪽. BERT: 첫 합성 전처럼 토크나이저
# 파일(vocab.txt)만 받은 스냅숏도 그 커밋으로 본다. 인식 모델: main이 가리키는 스냅숏에 확인할 파일
# (model.bin)이 없으면 모름. 저장소마다 확인하는 파일 이름도 이것으로 고정된다.
def test_model_revisions_name_the_snapshot_that_main_points_to(tmp_path):
    pytest.importorskip("huggingface_hub")

    def snapshot(repo_id: str, commit: str, filename: str) -> Path:
        repo = tmp_path / ("models--" + repo_id.replace("/", "--"))
        (repo / "snapshots" / commit).mkdir(parents=True, exist_ok=True)
        (repo / "snapshots" / commit / filename).write_bytes(b"")
        return repo

    def point_main(repo: Path, commit: str) -> None:
        (repo / "refs").mkdir(exist_ok=True)
        (repo / "refs" / "main").write_text(commit, encoding="utf-8")

    snapshot("myshell-ai/MeloTTS-Korean", "a" * 40, "checkpoint.pth")
    point_main(snapshot("myshell-ai/MeloTTS-Korean", "b" * 40, "checkpoint.pth"), "b" * 40)
    point_main(snapshot("kykim/bert-kor-base", "c" * 40, "vocab.txt"), "c" * 40)
    point_main(snapshot("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "d" * 40, "config.json"), "d" * 40)
    assert speech.model_revisions(cache_dir=tmp_path) == {
        "myshell-ai/MeloTTS-Korean": "b" * 40,
        "kykim/bert-kor-base": "c" * 40,
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo": "?",
    }


# huggingface_hub가 없는 환경(음성 그룹 없음), 캐시에 없는 파일(None), 허브에 없다고 캐시된 파일(경로가
# 아닌 표시 객체)은 모두 '?'로 적는지 본다. versions()가 설치되지 않은 패키지를 '?'로 적는 것과 같은 표기다.
def test_model_revisions_are_unknown_without_the_hub_library_or_the_file(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # import가 ImportError를 낸다
    assert speech.model_revisions() == dict.fromkeys(FAKE_REVISIONS, "?")
    hub = types.ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = lambda repo_id, filename, cache_dir=None: (  # type: ignore[attr-defined]
        None if "Melo" in repo_id else object()
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert speech.model_revisions() == dict.fromkeys(FAKE_REVISIONS, "?")


# 기록하는 세 저장소가 음성 라이브러리가 실제로 받는 저장소인지, 라이브러리를 불러오지 않고 소스 글로
# 본다(melo를 불러오면 한국어 앞단이 토크나이저를 허브에서 받으므로 불러오지 않는다). 음성 그룹이 없으면
# 소스 대조는 건너뛴다. 인식 저장소는 WhisperListener의 기본 모델(large-v3-turbo, 측정과 서비스가 쓰는
# 것)의 것이다. docs/voice.md의 리비전 표에도 세 저장소가 모두 있어야 한다(코드와 문서가 따로 놀지 않게).
def test_the_recorded_repositories_are_the_ones_the_speech_libraries_download():
    assert set(speech.SPEECH_MODEL_FILES) == set(FAKE_REVISIONS)
    assert inspect.signature(speech.WhisperListener).parameters["model"].default == "large-v3-turbo"
    doc = (Path(__file__).resolve().parents[1] / "docs" / "voice.md").read_text(encoding="utf-8")
    table = doc[doc.index("### 음성 모델의 리비전") :]
    assert all(f"| `{repo_id}` |" in table for repo_id in speech.SPEECH_MODEL_FILES)

    def source(package: str, relative: str) -> str:
        spec = importlib.util.find_spec(package)  # 최상위 패키지라 불러오지(실행하지) 않는다
        if spec is None or not spec.submodule_search_locations:
            pytest.skip(f"{package} is not installed (voice group)")
        return (Path(list(spec.submodule_search_locations)[0]) / relative).read_text(encoding="utf-8")

    download_utils = source("melo", "download_utils.py")
    assert (
        "'KR': 'myshell-ai/MeloTTS-Korean'" in download_utils
        and 'filename="checkpoint.pth"' in download_utils
    )
    assert "model_id = 'kykim/bert-kor-base'" in source("melo", "text/korean.py")
    whisper_utils = source("faster_whisper", "utils.py")
    assert '"large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"' in whisper_utils
    assert '"model.bin"' in whisper_utils
