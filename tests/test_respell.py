"""말로 푼 꼴을 고친 뒤 다시 재기: run --episode와 analyze respelled/stitch."""

# [테스트 묶음] 말로 풀어 쓰기를 고친 뒤 바뀐 발화가 든 에피소드만 다시 재는 도구를 확인한다: 실행기의
# --episode(고른 에피소드만, 전체 실행의 seed와 manifest로), respell.py의 바뀐 에피소드 찾기와 끼워 넣기.
# [왜 중요한가] 재측정 계획(docs/experiments.md 9단계 끝)은 바뀐 에피소드만 다시 재고 나머지 기록을 그대로
# 쓴다. 다시 잰 에피소드가 원래 실행과 다른 설정으로 돌았거나, 바꿀 에피소드를 잘못 고르면 수치가 조용히
# 섞인다. 그래서 끼워 넣기는 설정이 하나라도 다르면 거절해야 한다.
# [방법] 실행기는 가짜 Ollama로 개발용 과제 파일에서 돌린다. 끼워 넣기는 tmp_path에 만든 작은 기록으로 보고,
# 커밋된 기록에서는 바뀐 에피소드 목록이 계획과 같은지 본다.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from support_agent import analyze, respell, run
from support_agent.chat import ChatProvider, ChatResponse
from support_agent.paths import REPORTS
from support_agent.voice.verbalize import verbalize

# ==================================================================== 실행기: --episode


# 고객(도구 없이 불림)은 바로 끝내고, 상담원은 인사만 한다. 에피소드마다 같은 답을 낸다.
class StoppingOllama(ChatProvider):
    def __init__(self, model: str, **_options):
        self.model = model

    def preload(self) -> float:
        return 0.0

    def describe(self):
        return {"provider": "fake", "model": self.model}

    def chat(self, messages, tools=(), *, temperature=0.0, seed=None, max_tokens=1024):
        return ChatResponse("네, 확인해 보겠습니다." if tools else "###STOP###")


def run_main(tmp_path, monkeypatch, *argv: str) -> Path:
    monkeypatch.setattr(run, "OllamaProvider", StoppingOllama)
    monkeypatch.setattr(run, "gpu_used_by_others_mib", lambda provider: None)
    monkeypatch.setattr(run, "OUTPUTS", tmp_path)
    monkeypatch.setattr("sys.argv", ["run", *argv, "--label", "unit"])
    run.main()
    (run_dir,) = (tmp_path / "runs").iterdir()
    return run_dir


# --episode는 고른 (과제, 시도)만 과제 파일 순서대로 돌고, 시도 번호(곧 seed)는 전체 실행의 것이다. manifest의
# 과제 해시·프롬프트 해시·시도 수는 전체 실행과 같아, 9단계 V2 개발용 기록의 manifest와 그대로 맞는다(V0~V2의
# 프롬프트는 같다).
def test_episode_runs_only_the_chosen_episodes_and_describes_the_full_run(tmp_path, monkeypatch):
    run_dir = run_main(
        tmp_path,
        monkeypatch,
        "--tasks",
        "dev",
        "--trials",
        "4",
        "--episode",
        "dev-018:1",
        "--episode",
        "dev-002:3",
    )
    lines = (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    assert [(e["task_id"], e["trial"]) for e in map(json.loads, lines)] == [("dev-002", 3), ("dev-018", 1)]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    measured = json.loads((REPORTS / _V2_DEV_9 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["only_episodes"] == [["dev-002", 3], ["dev-018", 1]]
    assert manifest["trials"] == 4
    assert manifest["task_sha256"] == measured["task_sha256"]  # 돌린 두 과제가 아니라 24과제 전체
    assert manifest["prompt_sha256"] == measured["prompt_sha256"]


# --episode가 없으면 manifest에 only_episodes가 없고 고른 과제의 모든 시도를 돈다(지금까지의 실행 그대로).
def test_without_episode_every_trial_runs_and_the_manifest_is_as_before(tmp_path, monkeypatch):
    run_dir = run_main(
        tmp_path, monkeypatch, "--tasks", "smoke", "--task-id", "smoke-lookup-01", "--trials", "2"
    )
    lines = (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["trial"] for line in lines] == [0, 1]
    assert "only_episodes" not in json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


# 잘못된 --episode 값은 모델을 올리기 전에 명령줄 오류로 멈춘다.
@pytest.mark.parametrize(
    "argv",
    [
        ["--episode", "dev-018:4"],  # --trials 4에서 시도 번호 4(0부터 세므로 없음)
        ["--episode", "dev-999:0"],  # 없는 과제
        ["--episode", "dev-018"],  # 시도 번호 없음
        ["--episode", "dev-018:1", "--task-id", "dev-018"],  # --episode가 과제를 스스로 고른다
    ],
)
def test_a_wrong_episode_stops_before_any_model(tmp_path, monkeypatch, argv):
    with pytest.raises(SystemExit):
        run_main(tmp_path, monkeypatch, "--tasks", "dev", "--trials", "4", *argv)
    assert not (tmp_path / "runs").exists()


def test_parse_episodes():
    assert run.parse_episodes(["dev-002:0", "dev-002:3", "dev-002:0"], ["dev-002"], 4) == {
        ("dev-002", 0),
        ("dev-002", 3),
    }
    with pytest.raises(ValueError, match="TASK:TRIAL"):
        run.parse_episodes(["dev-002:x"], ["dev-002"], 4)


# ==================================================================== respelled, stitch
# (바뀐 에피소드 찾기와 끼워 넣기)

SAID = "송장 번호는 5551-0914-9121입니다."
BEFORE = "송장 번호는 오천오백오십일 구백십사에서 구천백이십일 입니다."  # 고치기 전의 말로 푼 꼴


def item(said: str, spoken: str | None = None) -> dict:
    return {"said": said, "spoken": verbalize(said) if spoken is None else spoken, "heard": "", "text": ""}


def episode(task: str, trial: int, *items: dict, status: str = "ok") -> dict:
    return {"task_id": task, "trial": trial, "status": status, "voice": list(items)}


MANIFEST = {
    "commit": "c0",
    "config": {"voice": "V2"},
    "trials": 2,
    "tasks_file": "dev",
    "task_sha256": {"dev-001": "a", "dev-002": "b"},
    "seed_hash": "s",
    "prompt_sha256": {"policy": "p"},
    "agent_provider": {"model": "qwen3.5:4b", "ollama_version": "0.35.1"},
    "user_provider": {"model": "qwen2.5:7b-instruct", "num_gpu": 24},
    "voice_channel": {"tts": "melotts/KR"},
    "caller_id": None,
    "versions": {"python": "3.11.9"},
}


def write_run(path: Path, episodes: list[dict], **manifest) -> Path:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps(MANIFEST | manifest, ensure_ascii=False), encoding="utf-8")
    lines = [json.dumps(e, ensure_ascii=False) for e in episodes]
    (path / "episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


# 원래 기록: dev-001 #1의 한 발화만 고치기 전의 글로 합성됐다. 다시 잰 실행: 그 에피소드만 지금 코드로.
def measured_and_rerun(tmp_path: Path, rerun_episode: dict | None = None, **rerun_manifest):
    original = write_run(
        tmp_path / "measured",
        [
            episode("dev-001", 0, item("네")),
            episode("dev-001", 1, item("네"), item(SAID, BEFORE)),
            episode("dev-002", 0, item("2~3개요")),
            episode("dev-002", 1, item("O-91010이요")),
        ],
    )
    fresh = episode("dev-001", 1, item("네"), item(SAID), item("고마워요"))
    rerun = write_run(
        tmp_path / "rerun",
        [rerun_episode or fresh],
        **({"commit": "c1", "only_episodes": [["dev-001", 1]]} | rerun_manifest),
    )
    return original, rerun


def test_changed_episodes_are_those_with_an_utterance_spelled_differently_now(tmp_path):
    original, _ = measured_and_rerun(tmp_path)
    assert respell.changed_episodes(original) == [("dev-001", 1)]
    assert "| measured | 5 | 1 | dev-001#1 |" in respell.respelled_table([original])


# 끼워 넣기: 원래 기록의 줄 순서와 바이트는 그대로, 바뀐 에피소드의 줄만 다시 잰 실행의 것이다. manifest는
# 원래 것에 stitched가 붙어, 원래 실행과 같은 설정으로 짝지어지고(same-setup), 지금 코드로 바뀌는 발화가
# 없다.
def test_stitch_replaces_only_the_changed_episodes(tmp_path):
    original, rerun = measured_and_rerun(tmp_path)
    out = tmp_path / "measured-respelled"
    assert respell.stitch(original, rerun, out) == [("dev-001", 1)]
    before = (original / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    after = (out / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(after) == 4 and [after[0], *after[2:]] == [before[0], *before[2:]]
    assert after[1] == (rerun / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0]
    stitched = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["stitched"]
    assert stitched["original_run"] == "measured" and stitched["rerun"] == "rerun"
    assert (stitched["original_commit"], stitched["rerun_commit"]) == ("c0", "c1")
    assert stitched["replaced"] == [["dev-001", 1]]
    assert analyze.setup_differences(original, out, ignore=()) == []
    assert respell.changed_episodes(out) == []
    with pytest.raises(ValueError, match="itself stitched"):  # 언제나 측정한 실행에서 시작한다
        respell.stitch(out, rerun, tmp_path / "again")


# 다시 잰 실행이 원래와 다른 조건이거나, 바꿀 에피소드가 없거나 infra 오류거나 지금 코드로 풀어 쓴 것이 아니면
# 끼워 넣지 않는다.
@pytest.mark.parametrize(
    ("rerun_episode", "rerun_manifest", "reason"),
    [
        (None, {"agent_provider": {"model": "qwen3.5:4b", "ollama_version": "0.34.2"}}, "agent_provider"),
        (None, {"user_provider": {"model": "qwen2.5:7b-instruct", "num_gpu": 20}}, "user_provider"),
        (None, {"trials": 3}, "trials"),
        (None, {"task_sha256": {"dev-001": "a"}}, "task_sha256"),
        (None, {"config": {"voice": "V4"}}, "config.voice"),
        (None, {"voice_channel": {"tts": "melotts/KR", "device": "cpu"}}, "voice_channel"),
        (None, {"versions": {"python": "3.12.0"}}, "versions"),
        (None, {"only_episodes": None}, "run --episode"),
        (episode("dev-001", 0, item(SAID)), {}, "dev-001#1 is not in"),
        (episode("dev-001", 1, status="infra_error"), {}, "infra error"),
        (episode("dev-001", 1, item(SAID, BEFORE)), {}, "not spelled by the current code"),
    ],
)
def test_stitch_refuses_a_rerun_that_does_not_match(tmp_path, rerun_episode, rerun_manifest, reason):
    original, rerun = measured_and_rerun(tmp_path, rerun_episode, **rerun_manifest)
    assert any(reason in problem for problem in respell.stitch_problems(original, rerun))
    with pytest.raises(ValueError):
        respell.stitch(original, rerun, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_run_without_changed_episodes_is_not_stitched(tmp_path):
    original = write_run(tmp_path / "measured", [episode("dev-001", 0, item(SAID))])
    rerun = write_run(tmp_path / "rerun", [episode("dev-001", 0, item(SAID))], only_episodes=[["dev-001", 0]])
    assert any("nothing to stitch" in p for p in respell.stitch_problems(original, rerun))


# 커밋된 음성 기록에서 바뀐 발화가 든 에피소드가 재측정 계획(docs/experiments.md 9단계 끝)의 목록과 같은지
# 본다. 계획의 --episode 값이 이 목록이다. 나머지 음성 기록에는 바뀐 에피소드가 없다.
_V2_DEV_5 = "20260921-145634-qwen2.5-7b-instruct-P0-R0-G0-F0-V2-v2"
_V2_DEV_9 = "20261004-131932-qwen3.5-4b-P0-R0-G0-F0-V2-v2-4b-stage9"
_V4_DEV = [("dev-002", 0), ("dev-002", 2), ("dev-002", 3), ("dev-018", 0)]
RESPELLED_EPISODES = {
    _V2_DEV_5: [("dev-018", 1)],
    _V2_DEV_9: [("dev-018", 1), ("dev-018", 2)],
    "20261004-154016-qwen3.5-4b-P0-R0-G0-F0-V4-v4-k0": _V4_DEV,
    "20261008-145519-v4r-a-dev": _V4_DEV,
    "20261009-101009-qwen3.5-4b-P0-R0-G0-F0-V2-test-v2-4b-stage9": [("test-025", 2)],
}


def test_the_committed_voice_records_change_in_the_planned_episodes_only():
    voice_runs = [
        run_dir
        for run_dir in sorted(REPORTS.iterdir())
        if (run_dir / "utterances.jsonl").exists()
        or (
            (run_dir / "manifest.json").exists()
            and analyze.load_manifest(run_dir)["config"].get("voice", "V0") != "V0"
        )
    ]
    assert len(voice_runs) >= 9
    assert {d.name: respell.changed_episodes(d) for d in voice_runs if respell.changed_episodes(d)} == (
        RESPELLED_EPISODES
    )
