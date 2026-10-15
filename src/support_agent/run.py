"""로컬 모델로 과제 × 시도를 돌리고 에피소드마다 JSONL 기록 한 줄을 쓴다.

사용법:
    uv run python -m support_agent.run --tasks smoke --trials 1
    uv run python -m support_agent.run --tasks smoke --model qwen2.5:14b-instruct --policy P1 --label p1-14b
    uv run python -m support_agent.run --tasks smoke --policy P1 --model qwen3:4b-instruct-2507-q4_K_M
        --think off --num-ctx 12288 --user-num-ctx 12288 --label smoke-m2a   (한 줄 명령)
    12단계(다른 시뮬레이터): --user-model qwen3:4b-instruct-2507-q4_K_M --user-think off
        --user-num-ctx 8192 를 더한다
    전체 실행의 에피소드 몇 개만 다시(말로 풀어 쓰기를 고친 뒤): 그 실행의 옵션에
        --episode dev-018:1 --episode dev-018:2 를 더하고, 뒤에 `analyze stitch` (docs/experiments.md 9단계)

기록은 outputs/runs/<run_id>/에 쌓인다(git이 무시함). --official을 주면 reports/에 쓴다(커밋 대상). 그러려면
작업 트리가 깨끗해야 하고, 시험용 과제 파일은 --allow-test도 있어야 한다. GPU를 다른 프로젝트와 나눠 쓰므로,
다른 것이 GPU 메모리를 쥐고 있으면 실행이 시작을 거부한다(--force 참고).
"""

# [파일 역할] 평가 실행기 CLI. 과제 × 시도마다 에피소드를 돌려 episodes.jsonl에 한 줄씩 쓰고, 실행의 조건을
# manifest.json에, 집계를 summary.json에 남긴다.
# [설계 이유] 이 프로젝트의 측정 규칙(규칙을 먼저 커밋하고 재기, 시험용은 단계마다 한 번, 기록에서 수치를 다시
# 계산)을 실행기 자체가 지키게 한다.
# - --official은 작업 트리가 깨끗해야 하고 reports/에 쓴다(커밋 대상). 기록의 커밋 해시가 실제로 돈 코드를
#   가리키게 하려는 것이다.
# - 시험용 과제 파일은 --allow-test 없이는 돌지 않는다.
# - manifest에 커밋, 설정, 모델 digest, Ollama 버전, 프롬프트·도구·과제·seed DB 해시를 남겨, analyze
#   same-setup이 두 실행을 짝지어도 되는지 확인할 수 있게 한다(README 다시 재는 방법).
# - GPU를 다른 프로젝트와 나눠 쓰므로 Ollama가 아닌 것이 GPU 메모리를 3 GB 넘게 쓰고 있으면 시작하지 않는다.
# - 모델 없이 실패할 수 있는 일(과제 로드, 정답 재실행)은 모델을 올리기 전에 모두 해서, 몇 시간의 GPU 시간을
#   쓴 뒤에 실패하지 않게 한다.
# [흐름] 인자 확인 → RunConfig → 제공자 준비 → GPU 확인 → 도구·seed·정답 덤프 → 음성 채널 → 모델 적재 →
# run_id·폴더 → manifest·seed 덤프 → _run_all → summary(Ctrl+C나 오류에도 finally로 씀).

from __future__ import annotations

# subprocess: git(커밋 해시, 작업 트리 상태)과 nvidia-smi(GPU 메모리)를 부른다. importlib.metadata.version:
# 주요 패키지 버전을 manifest에 남긴다.
import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

# 실행에 필요한 내부 모듈 전부: DB 덤프·해시(db), 프롬프트와 도구 목록(agent), 설정(config), R2 미리보기
# 코드(confirm), 에피소드(episode), pass^k(judge), Ollama 제공자, 경로(paths), 기록(records), seed DB(seed),
# 과제(tasks), 도구 등록부(tools), 시뮬레이터(user_sim), 음성 패키지 버전(voice.speech).
from support_agent import db
from support_agent.agent import build_system_prompt, load_policy, visible_tools
from support_agent.config import RunConfig
from support_agent.confirm import CONFIRM_CODE
from support_agent.episode import gold_dump_of, run_episode
from support_agent.judge import pass_k_table
from support_agent.ollama import OllamaProvider
from support_agent.paths import OUTPUTS, REPORTS, ROOT
from support_agent.records import EpisodeResult
from support_agent.seed import build_seed_engine
from support_agent.tasks import load_tasks
from support_agent.tools import NAME_TOLERANCE, build_registry
from support_agent.user_sim import LLMUser, build_user_prompt
from support_agent.voice.speech import versions as speech_versions

# infra_error가 3번 잇달아 나면 모델 서버가 죽었다고 보고 멈춘다. 나머지 과제를 모두 infra_error로 태우지 않게
# 하려는 것이다. GPU 한도 3,000 MiB는 데스크톱 앱이 쓰는 1~2 GiB보다 크면 다른 작업(다른 프로젝트의
# 학습·측정)이 돈다고 보는 기준이다.
MAX_CONSECUTIVE_INFRA_ERRORS = 3  # 모델 서버가 죽었을 것이다: 과제 목록을 태우지 말고 멈춘다
OTHER_GPU_USE_LIMIT_MIB = 3000  # 데스크톱 앱이 1~2 GiB를 쓴다. 이보다 많으면 다른 작업이 도는 것이다


# 저장소 루트에서 git 명령을 돌려 출력을 돌려준다. git이 없거나 실패하면 빈 문자열(그러면 --official은
# 거부된다).
def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


# nvidia-smi로 첫 GPU의 사용 중인 메모리(MiB)를 읽는다. nvidia-smi가 없거나 20초 안에 답하지 않으면
# None(검사를 건너뜀).
def gpu_memory_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


# Ollama가 아닌 것이 쓰는 GPU 메모리 = 전체 사용량 − Ollama가 보고한 모델별 GPU 적재량(size_vram).
# Windows에서는 프로세스별 조회가 쓸모없어 이렇게 뺀다(docstring).
def gpu_used_by_others_mib(provider: OllamaProvider) -> int | None:
    """Ollama가 아닌 것이 쥐고 있는 GPU 메모리. Windows에서는 `nvidia-smi --query-compute-apps`가 쓸모없어서
    (모든 데스크톱 앱을 나열한다) 전체 사용량에서 Ollama가 보고한 양을 뺀다."""
    used = gpu_memory_used_mib()
    if used is None:
        return None
    ollama_mib = sum(int(m.get("size_vram") or 0) for m in provider.loaded_models()) // (1024 * 1024)
    return max(used - ollama_mib, 0)


# 서버가 지금 메모리에 들고 있는 모델 목록. manifest(적재 직후)와 summary(끝날 때)에 남겨, 실행 중 모델이
# 내려가거나 CPU로 옮겨졌는지 기록으로 확인한다. loaded_models가 없는 제공자는 빈 목록이다.
def ollama_ps(provider: OllamaProvider) -> list[dict[str, Any]]:
    """서버가 메모리에 들고 있는 것(name, size, size_vram). 그것을 알려 줄 수 없는 제공자면 []."""
    loaded = getattr(provider, "loaded_models", None)
    return loaded() if callable(loaded) else []


# 글의 sha256. manifest의 프롬프트·도구 해시에 쓴다. safe_name은 모델 이름·라벨을 폴더 이름에 쓸 수 있는
# 글자로 바꾼다(qwen2.5:7b-instruct → qwen2.5-7b-instruct).
def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


# 실행 결과를 summary.json용으로 집계한다. infra_error는 분모에서 빼고(counted), 모든 시도가 infra_error인
# 과제는 표에서 조용히 사라지지 않게 따로 적는다.
# 형식 오류 수에서 멈춤(stall), 근거 없는 완료 주장(unbacked_claim), 되풀이(repeat, 13단계 D1)는 뺀다. 셋은
# 형식이 맞는 답을 가드가 돌려보낸 것이라 따로 세고, analyze table과 같은 기준이다(docs/experiments.md 7단계
# 코드 검토 뒤에 고친 것). 시간 값은 에피소드당 시간, 에이전트 호출당 시간, 모델 적재 시간 등이다.
def summarise(results: list[EpisodeResult]) -> dict[str, Any]:
    counted = [r for r in results if r.status != "infra_error"]
    by_task: dict[str, list[bool]] = {}
    for r in counted:
        by_task.setdefault(r.task_id, []).append(bool(r.verdict and r.verdict.success))
    # 모든 시도가 infra_error인 과제는 흔적 없이 표에서 사라지면 안 된다.
    unmeasured = sorted({r.task_id for r in results} - set(by_task))
    agent_calls = [c for r in counted for c in r.llm_calls if c.who == "agent"]
    user_calls = [c for r in counted for c in r.llm_calls if c.who == "user"]

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 1) if values else 0.0

    return {
        "episodes": len(results),
        "infra_errors": sum(r.status == "infra_error" for r in results),
        "truncated": sum(r.status == "truncated" for r in results),
        "terminations": {
            t: sum(r.termination == t for r in results) for t in sorted({r.termination for r in results})
        },
        "successes": sum(sum(v) for v in by_task.values()),
        # pass^k는 과제의 유효한 시도를 모두 쓴다(과제마다 자기 n으로 C(c,k)/C(n,k)), k는 가장 작은 n까지.
        # 그래서 infra_error로 잃은 시도는 k의 범위를 줄일 뿐 성공 수를 줄이지 않는다.
        "pass_hat_k": pass_k_table(by_task) if by_task else {},
        "by_task": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_task.items())},
        "tasks_without_valid_trials": unmeasured,
        "successes_with_unexpected_writes": sum(
            bool(r.verdict and r.verdict.success and r.verdict.unexpected_writes) for r in counted
        ),
        "unexpected_writes": sum(r.verdict.unexpected_writes for r in counted if r.verdict),
        "policy_violations": sum(len(r.verdict.policy_violations) for r in counted if r.verdict),
        "policy_blocks": sum(len(r.verdict.policy_blocks) for r in counted if r.verdict),
        "auth_blocks": sum(r.verdict.auth_blocks for r in counted if r.verdict),
        "format_errors": sum(
            c.format_error not in (None, "stall", "unbacked_claim", "repeat") for c in agent_calls
        ),
        "held_claims": sum(c.format_error == "unbacked_claim" for c in agent_calls),
        # 13단계: 되풀이 가드(D1)가 붙잡은 답 수. 형식 오류로 세지 않는다.
        "held_repeats": sum(c.format_error == "repeat" for c in agent_calls),
        "confirmation_requests": sum(t.error_code == CONFIRM_CODE for r in counted for t in r.tool_calls),
        "dropped_calls": sum(c.dropped_calls for c in agent_calls),
        "seconds_per_episode": mean([r.wall_seconds for r in counted]),
        "agent_calls_per_episode": mean([float(sum(c.who == "agent" for c in r.llm_calls)) for r in counted]),
        "agent_call_ms": mean([c.wall_ms for c in agent_calls]),
        "agent_prompt_eval_ms": mean([c.prompt_eval_ms for c in agent_calls]),
        "agent_prompt_tokens_max": max((c.prompt_tokens for c in agent_calls), default=0),
        # 컨텍스트 한도 검사는 prompt_tokens를 읽는다. 그 값을 보고하지 않은 호출은 검사의 사각지대다.
        "agent_calls_without_prompt_tokens": sum(c.prompt_tokens == 0 for c in agent_calls),
        "user_call_ms": mean([c.wall_ms for c in user_calls]),
        "model_load_ms_total": round(sum(c.load_ms for c in agent_calls + user_calls), 1),
    }


# 음성 조건(V1·V2·V4)의 음성 채널을 만든다. 음성 패키지(MeloTTS, faster-whisper)는 선택 의존성 그룹 voice에만
# 있으므로 함수 안에서 가져온다. 그래서 텍스트 실행과 테스트는 그 패키지 없이 돈다.
# V2·V4는 인식 결과를 규칙 기반 정규화기에 거친다. V4는 듣는 방식이 V2와 같고 본인 확인 방법만 다르다.
# 합성·인식 결과는 outputs/voice-cache에 캐시한다.
def build_channel(config: RunConfig, tts_device: str, stt_device: str):
    """V1/V2/V4의 음성 채널(`voice` 의존성 그룹이 필요하다). 텍스트 조건이면 None."""
    if config.voice == "V0":
        return None
    from support_agent.voice.channel import SpeechChannel
    from support_agent.voice.speech import MeloSpeaker, WhisperListener

    normalizer = None
    if config.voice in ("V2", "V4"):  # V4는 V2와 똑같이 듣고 본인 확인 방법만 다르다
        from support_agent.voice.normalize import normalize_heard as normalizer
    print(f"loading the speech models (tts on {tts_device}, stt on {stt_device}) ...")
    return SpeechChannel(
        MeloSpeaker(device=tts_device),
        WhisperListener(device=stt_device),
        normalizer=normalizer,
        cache_path=OUTPUTS / "voice-cache" / "roundtrips.jsonl",
    )


# --episode 값('dev-018:1')을 (과제, 시도) 집합으로 바꾼다. 과제 파일에 없는 과제나 --trials 밖의 시도는
# 오류다(ValueError, main이 parser.error로 낸다).
def parse_episodes(values: list[str], task_ids: list[str], trials: int) -> set[tuple[str, int]]:
    """'dev-018:1' -> ('dev-018', 1): 전체 실행의 에피소드. 과제 파일과 --trials에 비춰 확인한다."""
    picked = set()
    for value in values:
        task_id, sep, trial = value.rpartition(":")
        if not sep or not trial.isdigit():
            raise ValueError(f"--episode wants TASK:TRIAL, not {value!r}")
        if task_id not in task_ids:
            raise ValueError(f"--episode {value}: no task {task_id!r} in the task file")
        if int(trial) >= trials:
            raise ValueError(f"--episode {value}: trial {trial} is not below --trials {trials}")
        picked.add((task_id, int(trial)))
    return picked


# 명령줄 진입점. 옵션 이름은 실험 축 이름과 같다(--policy P1, --reasoning R1 …). README '다시 재는 방법'에
# 단계별 옵션이 있다.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tasks", default="smoke", help="tasks/ 안의 과제 파일 이름(.yaml 없이) 또는 파일 경로"
    )
    parser.add_argument("--task-id", action="append", help="이 과제 id만 돌린다 (여러 번 줄 수 있음)")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--episode",
        action="append",
        metavar="TASK:TRIAL",
        help="전체 실행 가운데 이 에피소드만 돌린다 (여러 번 줄 수 있음): seed가 같고, manifest는 전체 "
        "실행(과제·프롬프트 해시, 시도 수)을 적고 고른 에피소드를 나열한다",
    )
    parser.add_argument("--model", default=RunConfig.model)
    parser.add_argument("--user-model", default=None, help="시뮬레이터 모델 (기본: qwen2.5:7b-instruct)")
    parser.add_argument("--policy", choices=["P0", "P1"], default="P0")
    parser.add_argument(
        "--reasoning",
        choices=["R0", "R1", "R2"],
        default="R0",
        help="R1: think 도구. R2: 고객이 미리보기에 동의한 뒤에만 쓰기가 실행된다",
    )
    parser.add_argument(
        "--guard", choices=["G0", "G1", "G2"], default="G0", help="G1: 하겠다는 약속만 하는 답을 붙잡는다"
    )
    parser.add_argument(
        "--rescue", choices=["F0", "F1"], default="F0", help="F1: 본문에 샌 도구 호출을 실행한다"
    )
    parser.add_argument(
        "--claims",
        choices=["C0", "C1", "C2"],
        default="C0",
        help="C1: 도구 결과가 보여 주지 않는 일을 끝냈다고 말하는 답을 붙잡는다. C2: 재시도를 표본 추출로",
    )
    parser.add_argument(
        "--errors",
        choices=["E0", "E1"],
        default="E0",
        help="E1(13단계): 규정이 막은 쓰기의 오류 글에 그 검사가 쓴 사실(수령·출고 시각, 접수 마감일, 이미 "
        "발급된 쿠폰, 도착 예정일)을 붙인다",
    )
    parser.add_argument(
        "--repeat",
        choices=["D0", "D1"],
        default="D0",
        help="D1(13단계): 바로 앞에 고객에게 전달한 답과 같은 답을 붙잡아 한 번 다시 시킨다",
    )
    parser.add_argument(
        "--user-on-cpu",
        action="store_true",
        help="시뮬레이터 모델을 CPU에서 돌린다 (에이전트 모델이 GPU 메모리를 남기지 않을 때)",
    )
    parser.add_argument(
        "--user-num-gpu",
        type=int,
        default=None,
        help="시뮬레이터 모델을 GPU에 올릴 층 수, 나머지는 CPU (0이면 --user-on-cpu와 같음)",
    )
    parser.add_argument(
        "--think",
        choices=["on", "off"],
        default=None,
        help="에이전트 모델에만 think를 보낸다 (기본: 보내지 않음. 사고 모드가 없는 모델은 거절한다)",
    )
    parser.add_argument(
        "--user-think",
        choices=["on", "off"],
        default=None,
        help="시뮬레이터 모델에 think를 보낸다 (12단계. 기본: 7B 시뮬레이터처럼 보내지 않음)",
    )
    parser.add_argument(
        "--voice",
        choices=["V0", "V1", "V2", "V4"],
        default="V0",
        help="V1: 고객의 말이 음성 합성과 인식을 거쳐 들린다. V2: 거기에 정규화기까지. V4: V2이고, "
        "find_customer 대신 verify_caller가 발신 번호의 고객과 이름을 맞춰 본다",
    )
    parser.add_argument("--tts-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--stt-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-ctx", type=int, default=RunConfig.num_ctx)
    parser.add_argument(
        "--user-num-ctx", type=int, default=None, help="시뮬레이터의 num_ctx (기본: --num-ctx 값)"
    )
    parser.add_argument("--label", default="", help="run id에 붙일 짧은 이름")
    parser.add_argument("--official", action="store_true", help="reports/에 쓴다 (작업 트리가 깨끗해야 함)")
    parser.add_argument("--allow-test", action="store_true", help="시험용 과제 파일을 돌릴 때 필요하다")
    parser.add_argument("--force", action="store_true", help="GPU가 비어 있는지 보는 검사를 건너뛴다")
    args = parser.parse_args()

    # 시험용 과제는 단계가 끝날 때 고른 구성만 한 번 잰다. 실수로 시험용을 돌려 보는 것을 막는다.
    if "test" in Path(args.tasks).stem and not args.allow_test:
        parser.error("test tasks are measured once per stage; pass --allow-test when the stage is done")
    # 작업 트리가 깨끗한지 볼 때 reports/는 뺀다. 앞선 공식 실행의 기록이 커밋을 기다리며 남아 있을 수 있고,
    # 그것은 코드가 아니기 때문이다.
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--", ".", ":(exclude)reports"))
    if args.official and (dirty or not commit):
        parser.error(
            "--official needs git and a clean working tree, so that the commit describes the code that ran"
        )

    # 시뮬레이터 자리 옵션 정리. --user-on-cpu는 --user-num-gpu 0과 같다(2단계 M1: 14B가 GPU를 차지해
    # 시뮬레이터를 CPU에서 돌림). --user-num-ctx를 안 주면 에이전트와 같은 컨텍스트를 쓴다.
    if args.user_on_cpu and args.user_num_gpu not in (None, 0):
        parser.error("--user-on-cpu means --user-num-gpu 0")
    user_num_gpu = 0 if args.user_on_cpu else args.user_num_gpu
    user_num_ctx = args.user_num_ctx or args.num_ctx
    user_model = args.user_model or RunConfig.user_model
    # 두 역할이 같은 모델이면 실행기 옵션이 같아야 한다. Ollama는 모델마다 실행기를 하나 두므로, 옵션이 다르면
    # 턴마다 모델을 다시 읽게 된다.
    if user_model == args.model and (user_num_ctx != args.num_ctx or user_num_gpu is not None):
        parser.error(
            "the agent and the simulator are the same model, so they need the same runner options: "
            "drop --user-num-ctx/--user-on-cpu/--user-num-gpu or use another simulator model"
        )
    # think는 on/off를 줬을 때만 True/False로 보내고, 안 주면 None(요청에 넣지 않음)이다. 사고 모드가 없는
    # 모델은 think 값을 거절한다.
    think = None if args.think is None else args.think == "on"
    user_think = None if args.user_think is None else args.user_think == "on"

    config = RunConfig(
        model=args.model,
        user_model=user_model,
        policy=args.policy,
        reasoning=args.reasoning,
        guard=args.guard,
        rescue=args.rescue,
        voice=args.voice,
        claims=args.claims,
        errors=args.errors,
        repeat=args.repeat,
        num_ctx=args.num_ctx,
        user_num_ctx=user_num_ctx,
    )
    # 과제 파일을 읽고, --task-id가 있으면 그 과제만 남긴다. 모르는 id는 오류다.
    tasks = load_tasks(args.tasks)
    if args.task_id:
        unknown = sorted(set(args.task_id) - {t.id for t in tasks})
        if unknown:
            parser.error(f"unknown task ids: {unknown}")
        tasks = [t for t in tasks if t.id in set(args.task_id)]
    # --episode는 전체 실행(과제 파일 × --trials) 가운데 고른 에피소드만 다시 돈다. 말로 풀어 쓰기를 고친 뒤
    # 글이 바뀐 발화가 든 에피소드만 다시 재는 데 쓴다(docs/experiments.md 9단계 끝). 시도 번호가 같으면
    # seed도 같고, manifest의 과제·프롬프트 해시와 시도 수는 전체 실행의 것을 적어 원래 기록과 짝지을 수 있게
    # 한다(analyze stitch).
    manifest_tasks, only = tasks, None
    if args.episode:
        if args.task_id:
            parser.error("--episode picks the tasks itself; drop --task-id")
        try:
            only = parse_episodes(args.episode, [t.id for t in tasks], args.trials)
        except ValueError as error:
            parser.error(str(error))
        tasks = [t for t in tasks if t.id in {task_id for task_id, _ in only}]
    if not tasks:
        parser.error("no tasks selected")

    # think는 실행기 옵션이 아니라 요청 필드라 모델을 다시 읽게 하지 않는다. 에이전트 쪽에만 보낸다.
    provider = OllamaProvider(
        config.model, num_ctx=config.num_ctx, **({} if think is None else {"think": think})
    )
    # 두 역할이 같은 모델이고 다른 점이 하나도 없으면 제공자 객체를 하나만 쓴다: 실행기 옵션이 같아진다.
    same = config.user_model == config.model and think is None and user_think is None
    user_provider = (
        provider
        if same
        else OllamaProvider(
            config.user_model,
            num_ctx=config.user_num_ctx,
            num_gpu=user_num_gpu,
            **({} if user_think is None else {"think": user_think}),
        )
    )

    # 다른 작업이 GPU를 쓰고 있으면 시작하지 않는다. 같이 돌리면 두 쪽의 속도·메모리 수치가 모두 흔들린다.
    # --force는 이 검사를 건너뛴다.
    others = gpu_used_by_others_mib(provider)
    if others is not None and others > OTHER_GPU_USE_LIMIT_MIB and not args.force:
        sys.exit(
            f"{others} MiB of GPU memory is in use by something other than Ollama (limit "
            f"{OTHER_GPU_USE_LIMIT_MIB}). Another job is probably running; not starting (--force overrides)."
        )

    # V4면 find_customer 대신 verify_caller가 든 등록부를 만든다. seed DB는 프로세스에서 한 번 만들어 읽기
    # 전용으로 쓰고, 에피소드마다 복사한다.
    registry = build_registry(caller_id=config.caller_id)
    seed_engine = build_seed_engine()
    seed_dump = db.dump_db(seed_engine)
    policy_text = load_policy()
    # 모델 없이도 실패할 수 있는 일은 몇 시간의 GPU 시간을 쓰기 전에 여기서 모두 실패하게 한다.
    gold_dumps = {task.id: gold_dump_of(task, seed_engine, registry) for task in tasks}
    channel = build_channel(config, args.tts_device, args.stt_device)
    # 측정 전에 모델을 올려 둔다. 첫 에피소드의 시간에 모델 적재 시간이 섞이지 않게 하려는 것이다.
    print(f"loading {config.model} ... {provider.preload() / 1000:.1f} s")
    if user_provider is not provider:
        print(f"loading {config.user_model} ... {user_provider.preload() / 1000:.1f} s")
        # 서버는 둘째 모델을 올리려고 첫 모델을 내릴 수 있다. 그 뒤에 첫 모델이 둘째 옆에 들어갈 수 있어도
        # 그렇다(2026-10-04에 봄: qwen2.5 7B가 qwen3 4B를 내렸는데, 4B를 다시 올리자 7B 옆에 들어갔다). 그래서
        # 여기서 한 번 다시 올려, 첫 에피소드가 그 재적재 시간을 떠안지 않게 한다.
        held = [m.get("name") for m in ollama_ps(provider)]
        if held and config.model not in held:
            seconds = provider.preload() / 1000
            print(f"{config.model} was evicted by that load; loaded it again ... {seconds:.1f} s")
    if config.user_model != config.model:
        print("warning: two models take turns; if both do not fit in GPU memory every turn reloads one")
    # run_id = 시작 시각(UTC) + 모델 + 축 코드 + 라벨. 음성 축은 채널이 있을 때만, C·E·D 축은
    # 기본값(C0·E0·D0)이 아닐 때만 넣는다(그래서 지금까지의 실행 이름 꼴이 그대로다). 이 실행기는 평가
    # 하니스라 벽시계를 써도 된다(도구·seed·판정 코드만 시각을 읽지 않는다).
    started = datetime.now(UTC)
    run_id = "-".join(
        p
        for p in [
            started.strftime("%Y%m%d-%H%M%S"),
            safe_name(config.model),
            config.policy,
            config.reasoning,
            config.guard,
            config.rescue,
            config.voice if channel else "",
            config.claims if config.claims != "C0" else "",
            config.errors if config.errors != "E0" else "",
            config.repeat if config.repeat != "D0" else "",
            safe_name(args.label),
        ]
        if p
    )
    # 공식 측정은 reports/, 그 밖은 outputs/runs/. 같은 이름의 폴더가 이미 있으면 덮어쓰지 않고 오류를 낸다.
    run_dir = (REPORTS if args.official else OUTPUTS / "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # manifest: 이 실행을 다른 실행과 짝지어도 되는지 판단할 모든 조건. 프롬프트 해시는 첫 과제의 시각으로
    # 만든 에이전트 시스템 프롬프트와 시뮬레이터 프롬프트, 규정 문서, 모델에게 보인 도구 목록의 해시다. V4면
    # 발신 번호 확인의 이름 허용 오차도 남긴다.
    tools_json = json.dumps(visible_tools(registry, config), ensure_ascii=False, sort_keys=True)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started.isoformat(timespec="seconds"),
        "official": args.official,
        "commit": commit,
        "dirty": dirty,
        "config": config.to_dict(),
        "trials": args.trials,
        "tasks_file": str(args.tasks),
        "task_sha256": {t.id: t.sha256() for t in manifest_tasks},
        "seed_hash": db.state_hash(seed_dump),
        "prompt_sha256": {
            "policy": text_sha256(policy_text),
            "agent_system": text_sha256(
                build_system_prompt(policy_text, manifest_tasks[0].now, caller_id=config.caller_id)
            ),
            "user_sim": text_sha256(build_user_prompt(manifest_tasks[0])),
            "tools": text_sha256(tools_json),
        },
        "agent_provider": provider.describe(),
        "user_provider": user_provider.describe(),
        "voice_channel": channel.describe() if channel else None,
        "caller_id": (
            {"tool": "verify_caller", "name_tolerance": NAME_TOLERANCE} if config.caller_id else None
        ),
        "gpu_mib_used_by_others_at_start": others,
        "ollama_ps_after_preload": ollama_ps(provider),
        "versions": {"python": sys.version.split()[0]}
        | {name: version(name) for name in ("sqlalchemy", "pydantic", "httpx", "pyyaml")}
        | (speech_versions("faster-whisper", "ctranslate2", "melotts", "torch") if channel else {}),
    }
    if only:
        manifest["only_episodes"] = [[task_id, trial] for task_id, trial in sorted(only)]
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    # seed 덤프를 해시별로 한 번만 저장한다. seed 생성기가 나중에 바뀌어도 옛 기록을 그때의 seed로 다시 판정할
    # 수 있게 하려는 것이다.
    seed_file = (REPORTS if args.official else OUTPUTS) / "seeds" / f"{manifest['seed_hash'][:16]}.json"
    if not seed_file.exists():
        seed_file.parent.mkdir(parents=True, exist_ok=True)
        seed_file.write_text(json.dumps(seed_dump, ensure_ascii=False), encoding="utf-8", newline="\n")

    picked = f" (only {len(only)} episodes)" if only else ""
    print(f"run {run_id}: {len(tasks)} tasks x {args.trials} trials{picked} -> {run_dir}")
    results: list[EpisodeResult] = []
    # 실행 도중 Ctrl+C나 오류로 멈춰도 finally에서 그때까지의 summary를 쓴다. 끝날 때의 ollama ps도 남겨
    # 모델이 쫓겨났거나 CPU로 옮겨졌는지 보이게 한다.
    try:
        _run_all(
            tasks,
            args.trials,
            results,
            run_dir,
            config,
            provider,
            user_provider,
            registry,
            seed_engine,
            policy_text,
            gold_dumps,
            run_id,
            channel,
            only,
        )
    finally:  # Ctrl-C나 비정상 종료 뒤에도: 그때까지 잰 것을 읽을 수 있게 남긴다
        summary = summarise(results)
        summary["ollama_ps_at_end"] = ollama_ps(provider)  # 모델이 내려갔거나 CPU로 옮겨졌으면 여기 보인다
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=1))


# 과제 × 시도를 차례로 돌린다. 에피소드마다 새 시뮬레이터(LLMUser)를 만들고, 결과를 한 줄씩 쓰고 바로 flush해,
# 실행이 중간에 멈춰도 그때까지의 기록이 남는다. 진행 상황(PASS/fail/infra, 종료 사유, 시간)을 출력한다.
def _run_all(
    tasks,
    trials,
    results,
    run_dir,
    config,
    provider,
    user_provider,
    registry,
    seed_engine,
    policy_text,
    gold_dumps,
    run_id,
    channel=None,
    only=None,
) -> None:
    infra_streak = 0
    with (run_dir / "episodes.jsonl").open("a", encoding="utf-8", newline="\n") as out:
        for task in tasks:
            gold_dump = gold_dumps[task.id]
            for trial in range(trials):
                # --episode면 고른 (과제, 시도)만 돈다. 건너뛴 시도도 번호는 그대로라 seed가 전체 실행과 같다.
                if only is not None and (task.id, trial) not in only:
                    continue
                result = run_episode(
                    task,
                    trial,
                    config=config,
                    provider=provider,
                    user=LLMUser(user_provider, task, config, trial),
                    registry=registry,
                    seed_engine=seed_engine,
                    policy_text=policy_text,
                    gold_dump=gold_dump,
                    run_id=run_id,
                    channel=channel,
                )
                results.append(result)
                out.write(result.to_json_line() + "\n")
                out.flush()
                mark = "infra" if result.verdict is None else ("PASS" if result.verdict.success else "fail")
                print(f"  {task.id} #{trial}: {mark:5} {result.termination:22} {result.wall_seconds:6.1f} s")
                # infra_error 연속 횟수를 세고, 정상 에피소드가 나오면 0으로 되돌린다.
                infra_streak = infra_streak + 1 if result.status == "infra_error" else 0
                if infra_streak >= MAX_CONSECUTIVE_INFRA_ERRORS:
                    sys.exit(f"{infra_streak} infra errors in a row; stopping. Last: {result.error[:300]}")


if __name__ == "__main__":
    main()
