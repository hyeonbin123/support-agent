"""말로 풀어 쓰기(voice/verbalize.py)를 고친 뒤의 재측정. 여기서는 모델을 부르지 않는다.

음성 실행은 고객 발화마다 말로 풀어 쓴 형태를 합성했고 그것을 기록했다(`spoken`). verbalize()가 바뀌어도,
기록된 발화를 지금 코드가 기록과 똑같이 풀어 쓰는 에피소드는 고친 코드로 돌렸어도 같은 길을 갔을 것이므로
그 기록을 그대로 둔다. 지금 다르게 풀어 쓰는 발화가 든 에피소드는 다시 돌려야 한다. `respelled`는 그런
에피소드를 나열하고, `stitch`는 실행의 기록을 복사하면서 그 에피소드들을, 그것들만 다시 돈 실행(원래
실행의 옵션으로 `run --episode TASK:TRIAL`)의 같은 에피소드로 바꿔 넣는다. 바꾸기 전에 다시 잰 실행이
원래 실행과 같은 설정인지 확인한다(docs/experiments.md 9단계 끝).

사용법:
    uv run python -m support_agent.analyze respelled reports/<음성 실행> [...]
    uv run python -m support_agent.analyze stitch reports/<음성 실행> reports/<다시 잰 실행> [--out <폴더>]
"""

# [파일 역할] 말로 풀어 쓰기(voice/verbalize.py)를 고친 뒤, 기록된 음성 실행 가운데 바뀐 발화가 든
# 에피소드만 다시 재고 원래 기록에 끼워 넣는 도구다. 모델을 부르지 않는다.
# [왜 필요한가] 합성기에 들어간 글(spoken)이 같으면 같은 소리 → 같은 받아쓰기라, 그 에피소드는 고친 코드로
# 돌려도 같은 길을 간다. 그래서 실행 전체를 다시 재면 바뀌지 않은 에피소드까지 GPU 비결정성의 잡음을 새로
# 얹게 되고, 고침의 효과와 다시 잰 잡음이 섞인다. 바뀐 에피소드만 다시 재면 그 둘이 갈린다(docs/experiments.md
# 9단계 끝의 재측정 계획).
# [설계 이유] 끼워 넣기 전에 다시 잰 실행이 원래 실행과 같은 설정인지(설정, 시도 수, 과제·seed·프롬프트 해시,
# 두 제공자의 모델 digest·Ollama 버전·옵션, 음성 채널, 패키지 버전) 확인하고, 하나라도 다르면 거절한다. 원래
# 기록의 줄은 바이트 그대로 옮긴다. 바꾼 에피소드와 두 실행의 이름·커밋은 manifest의 stitched에 남는다.
# [연결] analyze.py의 respelled, stitch 명령이 respelled_table, stitch를 부른다. 다시 재는 쪽은 run.py의
# --episode다.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from support_agent.analyze import load_episodes, load_manifest, setup_differences
from support_agent.voice.verbalize import verbalize

# 원래 실행과 다시 잰 실행이 같아야 하는 manifest 항목 가운데 setup_differences가 보지 않는 것.
_ALSO_COMPARED = ("tasks_file", "voice_channel", "caller_id", "versions")


# 기록된 음성 항목(said, spoken) 가운데 지금 코드가 다르게 풀어 쓰는 것의 순번.
def respelled_indexes(items: list[dict[str, Any]]) -> list[int]:
    return [i for i, item in enumerate(items) if verbalize(item["said"]) != item["spoken"]]


# 실행 폴더의 음성 기록을 (과제, 시도, 음성 항목 목록)으로 읽는다. 에피소드 기록(episodes.jsonl의 voice)과
# 11단계 기록(utterances.jsonl, 줄마다 발화 하나)을 같은 꼴로 맞춘다.
def _voice_records(run_dir: Path) -> list[tuple[str, int, list[dict[str, Any]]]]:
    if (run_dir / "episodes.jsonl").exists():
        return [(e["task_id"], e["trial"], e.get("voice") or []) for e in load_episodes(run_dir)]
    lines = (run_dir / "utterances.jsonl").read_text(encoding="utf-8").splitlines()
    return [(u["task_id"], u["trial"], [u]) for u in (json.loads(line) for line in lines if line.strip())]


# 지금 코드가 다르게 풀어 쓰는 발화가 하나라도 든 (과제, 시도).
def changed_episodes(run_dir: Path) -> list[tuple[str, int]]:
    """음성 실행의 에피소드 가운데 지금 verbalize()가 다르게 풀어 쓰는 발화가 든 것."""
    return sorted(
        {(task, trial) for task, trial, items in _voice_records(run_dir) if respelled_indexes(items)}
    )


# 실행마다 음성 발화 수, 지금 코드가 다르게 풀어 쓰는 발화 수, 그 발화가 든 에피소드. 시험용 기록도 읽으므로
# 발화 글은 내지 않고 위치만 낸다.
def respelled_table(run_dirs: list[Path]) -> str:
    lines = [
        "| 실행 | 음성 발화 | 지금 코드로 다르게 풀어 쓰는 발화 | 그 발화가 든 에피소드 |",
        "|---|---|---|---|",
    ]
    for run_dir in run_dirs:
        records = _voice_records(run_dir)
        total = sum(len(items) for _, _, items in records)
        changed = sum(len(respelled_indexes(items)) for _, _, items in records)
        episodes = ", ".join(f"{task}#{trial}" for task, trial in changed_episodes(run_dir)) or "없음"
        lines.append(f"| {run_dir.name} | {total} | {changed} | {episodes} |")
    return "\n".join(lines)


# 끼워 넣을 수 없는 이유 목록(빈 목록 = 끼워 넣어도 됨).
# - 다시 잰 실행은 --episode로 고른 에피소드만 돈 것이어야 한다.
# - 설정이 원래 실행과 같아야 한다: setup_differences의 모든 항목(무시하는 축 없음)에서 과제별 판정된 시도만
#   빼고, 과제 파일 이름·음성 채널·발신 번호 설정·패키지 버전까지.
# - 원래 실행에서 바뀐 에피소드마다 다시 잰 실행에 같은 (과제, 시도)가 있고, infra 오류가 아니고, 그 발화를
#   지금 코드로 풀어 쓴 것이어야 한다.
def stitch_problems(original: Path, rerun: Path) -> list[str]:
    """`rerun`이 `original`의 바뀐 에피소드를 대신할 수 없는 이유들. 대신해도 되면 빈 목록."""
    a, b = load_manifest(original), load_manifest(rerun)
    problems = []
    if a.get("stitched"):
        problems.append(f"{original.name} is itself stitched: start from the run that was measured")
    if not b.get("only_episodes"):
        problems.append(f"{rerun.name} did not run chosen episodes (run --episode)")
    # 다시 잰 실행은 에피소드 몇 개만 돌았으므로, 과제별 판정된 시도 수만은 달라도 된다.
    problems += [
        d for d in setup_differences(original, rerun, ignore=()) if not d.startswith("judged trials")
    ]
    problems += [f"{key} differs" for key in _ALSO_COMPARED if a.get(key) != b.get(key)]
    needed = changed_episodes(original)
    if not needed:
        problems.append(f"no episode of {original.name} is spelled differently now: nothing to stitch")
    rerun_episodes = {(e["task_id"], e["trial"]): e for e in load_episodes(rerun)}
    for task, trial in needed:
        episode = rerun_episodes.get((task, trial))
        if episode is None:
            problems.append(f"{task}#{trial} is not in {rerun.name}")
        elif episode["status"] == "infra_error":
            problems.append(f"{task}#{trial} is an infra error in {rerun.name}")
        elif respelled_indexes(episode.get("voice") or []):
            problems.append(f"{task}#{trial} of {rerun.name} is not spelled by the current code")
    return problems


# 원래 기록을 out_dir에 옮기면서 바뀐 에피소드의 줄만 다시 잰 실행의 줄로 바꾼다. 줄 순서는 원래 기록
# 그대로다. manifest는 원래 것에 run_id와 stitched(원래 실행·다시 잰 실행과 각 커밋, 바꾼 에피소드)를
# 더한다. summary.json은 쓰지 않는다(수치는 analyze가 episodes.jsonl에서 다시 계산한다).
def stitch(original: Path, rerun: Path, out_dir: Path) -> list[tuple[str, int]]:
    """`out_dir`를 쓴다: `original`의 기록에서 바뀐 에피소드만 `rerun`의 것으로 바꾼 기록."""
    problems = stitch_problems(original, rerun)
    if problems:
        raise ValueError("; ".join(problems))
    needed = set(changed_episodes(original))

    def lines_by_episode(run_dir: Path) -> list[tuple[tuple[str, int], str]]:
        lines = (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        return [
            ((e["task_id"], e["trial"]), line) for line in lines if line.strip() for e in [json.loads(line)]
        ]

    replacements = dict(lines_by_episode(rerun))
    manifest = load_manifest(original)
    manifest["run_id"] = out_dir.name
    manifest["stitched"] = {
        "original_run": original.name,
        "original_commit": manifest.get("commit"),
        "rerun": rerun.name,
        "rerun_commit": load_manifest(rerun).get("commit"),
        "replaced": [[task, trial] for task, trial in sorted(needed)],
        "why": "an utterance of these episodes is spelled differently by the current verbalize()",
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    body = [replacements[key] if key in needed else line for key, line in lines_by_episode(original)]
    (out_dir / "episodes.jsonl").write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")
    return sorted(needed)
