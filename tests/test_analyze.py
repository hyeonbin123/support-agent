# [테스트 묶음] analyze.py가 실행 기록(episodes.jsonl)만으로 pass^k, 95% 구간, 마크다운 표, 값만 놓친 에피소드
# 목록, 점검용 표본을 다시 계산하는지 확인한다.
# [왜 중요한가] 이 저장소의 수치는 모두 reports/의 기록에서 analyze로 다시 계산된다 (README "다시 재는 방법").
# 계산이 바뀌면 문서의 수치를 다시 낼 수 없으므로, 손으로 셀 수 있는 작은 실행으로 결과를 고정한다.
# [방법] test_tasks.py의 대본 에이전트 도우미로 스모크 과제 하나를 네 번 돌려 실행 폴더를 만든다. 모델은 쓰지
# 않는다.

import json

import pytest
from test_tasks import ALL_TASKS, CONTACTS, episode, say_values

from support_agent import analyze, run, user_sim
from support_agent.chat import ToolCall
from support_agent.config import RunConfig
from support_agent.guard_report import GUARD_AXES
from support_agent.paths import REPORTS

# 9단계 시험용 기록(reports/ 아래 폴더). 끼워 넣은 V2가 그 단계의 값이고, 원래 잰 V2는 "수정 전 코드로 낸
# 값"으로 남아 있다(docs/experiments.md 9단계 재측정).
STAGE9_TEST_V2 = "20261009-101009-qwen3.5-4b-P0-R0-G0-F0-V2-test-v2-4b-stage9-respelled"
STAGE9_TEST_V2_AS_MEASURED = "20261009-101009-qwen3.5-4b-P0-R0-G0-F0-V2-test-v2-4b-stage9"
STAGE9_TEST_V4 = "20261009-132408-qwen3.5-4b-P0-R0-G0-F0-V4-test-v4-k0"
# 1단계 첫 기준선(턴 리마인더 전, 시뮬레이터 temperature 0.7)과 리마인더를 넣은 둘째 기준선, 8단계 M2b
# 개발용과 13단계 개발용 팔 둘(C1·D1). 옛 manifest(harness_sha256 없음)의 짝짓기를 실제 기록으로 본다.
FIRST_BASELINE = "20260919-093328-qwen2.5-7b-instruct-P0-R0-baseline"
FIRST_BASELINE_COMMIT = "f4ec9a48da2715c5722a8b8ba0647fc0bc77663f"
SECOND_BASELINE = "20260919-102552-qwen2.5-7b-instruct-P0-R0-baseline2"
DEV_M2B = "20261003-200306-qwen3.5-4b-P1-R0-G0-F0-m2b"
STAGE13_DEV_C1 = "20261018-021911-qwen3.5-4b-P1-R0-G0-F0-C1-s13-c1"
STAGE13_DEV_D1 = "20261018-032917-qwen3.5-4b-P1-R0-G0-F0-D1-s13-d1"


# smoke-action-01 과제의 가짜 실행 폴더를 만든다. 네 시도: 정답(성공), DB는 맞지만 환불 금액을 말하지
# 않음(실패), 아무것도 하지 않고 거절(실패), 정답(성공). 시도 번호를 0~3으로 매겨 episodes.jsonl에 쓴다.
def make_run(tmp_path):
    task = next(t for t in ALL_TASKS if t.id == "smoke-action-01")
    name, contact = CONTACTS[task.customer_id]
    good = [ToolCall("find_customer", {"name": name, "contact": contact})]
    good += [ToolCall(a.tool, a.args) for a in task.gold_actions] + [say_values(task)]
    quiet = good[:-1] + ["취소가 완료되었습니다."]  # DB는 맞지만 환불 금액을 말하지 않는다
    results = [episode(task, good), episode(task, quiet), episode(task, ["안 됩니다."]), episode(task, good)]
    for trial, result in enumerate(results):
        result.trial = trial
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    lines = "".join(r.to_json_line() + "\n" for r in results)
    (run_dir / "episodes.jsonl").write_text(lines, encoding="utf-8", newline="\n")
    return run_dir


# 과제별 성공 목록과 pass^k를 확인한다. pass^k는 관측한 n번 중 k번을 뽑아 모두 성공할 확률 C(c,k)/C(n,k)의
# 과제 평균이다: 4번 중 2번 성공이면 pass^1 = 0.5, pass^2 = 1/6. 구간은 시도가 아니라 과제를 다시 뽑는
# 부트스트랩이라, 과제가 하나면 구간이 점 하나가 된다.
def test_pass_k_and_the_interval_come_from_the_records(tmp_path):
    by_task = analyze.successes_by_task(analyze.load_episodes(make_run(tmp_path)))
    assert by_task == {"smoke-action-01": [True, False, False, True]}
    assert analyze.pass_k(by_task, 1) == 0.5 and analyze.pass_k(by_task, 2) == 1 / 6
    assert analyze.bootstrap_interval(by_task, 1) == (0.5, 0.5)  # 과제가 하나라 다시 뽑을 것이 없다


# table()이 실행마다 한 줄(과제 수, 에피소드 수, infra, 중단, pass^1과 구간)과 과제 유형별·종료 사유별 줄을
# 내는지 확인한다. 문서의 표가 이 출력에서 나온다.
def test_the_table_is_markdown_with_one_row_per_run(tmp_path):
    text = analyze.table([make_run(tmp_path)])
    assert "| run-a | 1 | 4 | 0 | 0 | 50.0% [50.0%, 50.0%] |" in text
    assert "| run-a | action | 1 | 50.0%" in text and "| run-a | user_stop | 4 |" in text


# misses()는 DB는 정답과 같은데 전달 값만 놓친 에피소드만 보여 줘야 한다 (여기서는 '취소가 완료되었습니다.'만
# 말한 시도 하나). 실패 분석에서 '처리는 했는데 금액을 안 말함'을 따로 보려는 도구다.
def test_misses_lists_only_episodes_that_failed_on_the_value_alone(tmp_path):
    text = analyze.misses(make_run(tmp_path))
    assert text.count("## smoke-action-01") == 1 and "취소가 완료되었습니다." in text


# 시뮬레이터 점검용 표본(sample)은 고정 seed로 뽑아 몇 번을 불러도 같아야 한다. 점검을 다시 해도 같은
# 에피소드를 읽게 하려는 것이다. 표본에는 도구 호출이 '(도구) 이름 인자' 줄로 보인다.
def test_the_sample_is_the_same_every_time(tmp_path):
    run_dir = make_run(tmp_path)
    assert analyze.sample(run_dir, 2) == analyze.sample(run_dir, 2)
    assert "(도구) find_customer" in analyze.sample(run_dir, 4)


# 과제별 성공 여부 목록으로 episodes.jsonl만 있는 가짜 실행 폴더를 만든다. 판정에 필요한 최소 필드(상태, 종료
# 이유, db_match, 빈 values)만 넣는다(tests/test_stage2.py의 compare 테스트와 같은 꼴).
def write_run(tmp_path, name, trials_by_task):
    run_dir = tmp_path / name
    run_dir.mkdir()
    lines = [
        json.dumps(
            {
                "task_id": task,
                "trial": i,
                "status": "completed",
                "termination": "user_stop",
                "verdict": {"db_match": ok, "values": {}},
            }
        )
        for task, oks in trials_by_task.items()
        for i, ok in enumerate(oks)
    ]
    (run_dir / "episodes.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return run_dir


# 과제 단위 부호 뒤집기 순열 검정(양측)의 정확 p를 손으로 셀 수 있는 예로 고정한다. 과제별 차이(후보 − 기준)의
# 부호를 모두 뒤집어 보는 2^n 배치 가운데 |합|이 관측값 이상인 비율이다. 0인 차이는 뒤집어도 같으므로 세지
# 않는다(docs/experiments.md 공통 규칙의 2026-10-29 메모).
def test_the_sign_flip_p_is_exact_for_small_examples():
    # 0이 아닌 차이 셋이 모두 크기 0.25: 어떻게 뒤집어도 |합|은 0.25 또는 0.75라 관측값 0.25 이상 → 8/8
    assert analyze.sign_flip_p([0.25, 0.25, -0.25, 0.0]) == (1.0, 3, True)
    # 모두 양수: 16개 배치 가운데 |합|이 관측값 1.25 이상인 것은 부호가 모두 같은 둘뿐 → 2/16
    assert analyze.sign_flip_p([0.5, 0.25, 0.25, 0.25]) == (0.125, 4, True)
    # 차이가 전부 0이거나 합이 0이면 p = 1
    assert analyze.sign_flip_p([0.0, 0.0]) == (1.0, 0, True)
    assert analyze.sign_flip_p([0.5, -0.5]) == (1.0, 2, True)


# 0이 아닌 차이가 20개를 넘으면 고정 seed(20261024) 몬테카를로 20,000회다: 같은 입력은 같은 p를 내고, 전수
# 값과 가깝다. 21개(+0.25 × 14, −0.25 × 7)의 전수 값은 2 × P(이항(21, 1/2) ≥ 14) = 396,880 / 2^21 ≈ 0.189다.
def test_the_sign_flip_p_falls_back_to_a_seeded_monte_carlo():
    diffs = [0.25] * 14 + [-0.25] * 7
    p, nonzero, exact = analyze.sign_flip_p(diffs)
    assert (nonzero, exact) == (21, False)
    assert analyze.sign_flip_p(diffs) == (p, 21, False)  # 고정 seed라 다시 불러도 같다
    exact_p = analyze.sign_flip_p(diffs, exact_max=21)
    assert exact_p == (pytest.approx(396_880 / 2**21), 21, True)
    assert abs(p - exact_p[0]) < 0.01


# analyze compare는 표 아래 빈 줄 뒤에 후보마다 순열 p 줄을 더 낸다. 표 자체(열과 순서)는 그대로라
# docs/experiments.md에 붙인 표와 어긋나지 않는다. 과제별 pass^1 차이 [+0.5, +0.25, 0]: 부호 배치 4개 가운데
# |합|이 관측값 0.75 이상인 것은 둘 → 0.5. pass^4 차이는 모두 0 → 1. pass^1 구간의 아래 끝은 기준 에피소드
# 수(12)를 곱한 '에피소드 몫'으로도 적는다('경계' 꼬리표 규칙이 에피소드 2개 몫을 문턱으로 쓴다).
def test_compare_adds_the_permutation_p_below_the_table(tmp_path):
    base_by_task = {"a": [True, False, False, False], "b": [False] * 4, "c": [True] * 4}
    other_by_task = {"a": [True, True, True, False], "b": [True, False, False, False], "c": [True] * 4}
    base = write_run(tmp_path, "base", base_by_task)
    other = write_run(tmp_path, "other", other_by_task)
    text = analyze.compare(base, [other])
    table, _, extra = text.partition("\n\n")
    assert table.startswith("| 실행 | pass^1 | 기준과의 차이 [95% 구간] | pass^4 | 차이 [95% 구간] |")
    assert table.count("\n") == 3 and table.endswith("|")
    _, low, _ = analyze.paired_difference(base_by_task, other_by_task, 1)
    assert extra == (
        "순열 p (과제 단위 부호 뒤집기, 양측): other pass^1 0.500 (정확, 0이 아닌 차이 2개), "
        f"pass^4 1.000 (정확, 0이 아닌 차이 0개); pass^1 구간 아래 끝 = 에피소드 {low * 12:.1f}개 몫"
    )


# 9단계 시험용 pass^1 V4 − V2의 순열 p를 기록에서 다시 낸다(시험용 기록은 수치만 읽는다): 끼워 넣은 V2로
# 0.058, 수정 전 코드로 낸 V2로 0.092. 둘 다 0이 아닌 차이가 20개라 전수다. 등록 판정(백분위 부트스트랩,
# +8.8%p [+1.2, +16.9])은 그대로이고 이 값은 compare가 표 아래에 더 내는 보조 지표다.
@pytest.mark.parametrize(("v2", "expected"), [(STAGE9_TEST_V2, 0.058), (STAGE9_TEST_V2_AS_MEASURED, 0.092)])
def test_stage9_test_pass1_sign_flip_p_is_reproduced_from_the_records(v2, expected):
    base = analyze.successes_by_task(analyze.load_episodes(REPORTS / v2))
    other = analyze.successes_by_task(analyze.load_episodes(REPORTS / STAGE9_TEST_V4))
    p, nonzero, exact = analyze.paired_sign_flip_p(base, other, 1)
    assert (round(p, 3), nonzero, exact) == (expected, 20, True)


# harness_sha256이 생기기 전의 manifest(필드 없음)는 2026-10-29 코드의 값(analyze.py에 고정)과 같은 설정으로
# 짝지어지고, 값이 다른 새 manifest는 어느 키가 다른지와 함께 차이로 나온다. 옛 기록 전부와의 짝짓기가 새 필드
# 때문에 막히면 안 되고, 하니스 글을 고친 뒤의 실행은 옛 기록과 걸려야 한다.
def test_old_manifests_without_harness_hashes_still_pair(tmp_path):
    def run_dir(name, harness):
        folder = tmp_path / name
        folder.mkdir()
        provider = {"provider": "ollama", "model": "qwen3.5:4b", "truncate": False}
        manifest = {
            "config": RunConfig().to_dict(),
            "trials": 1,
            "task_sha256": {"dev-001": "x"},
            "seed_hash": "s",
            "prompt_sha256": {"policy": "p"},
            "agent_provider": provider,
            "user_provider": provider,
        }
        if harness is not None:
            manifest["harness_sha256"] = harness
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (folder / "episodes.jsonl").write_text(
            json.dumps({"status": "completed", "task_id": "dev-001", "trial": 0}) + "\n", encoding="utf-8"
        )
        return folder

    frozen = dict(analyze.OLD_RECORDS_HARNESS_SHA256)
    old, new = run_dir("old", None), run_dir("new", frozen)
    changed = run_dir("changed", frozen | {"stall_notice": "0" * 64})
    assert analyze.setup_differences(old, new) == [] and analyze.setup_differences(new, old) == []
    assert analyze.setup_differences(new, changed) == ["harness_sha256 differs (stall_notice)"]
    assert analyze.setup_differences(old, changed) == ["harness_sha256 differs (stall_notice)"]
    # 글이 달랐던 첫 기준선의 커밋이면 옛 manifest라도 turn_reminder가 다르게 채워진다.
    first = run_dir("first", None)
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    (first / "manifest.json").write_text(
        json.dumps(manifest | {"commit": FIRST_BASELINE_COMMIT}), encoding="utf-8"
    )
    assert analyze.setup_differences(first, new) == ["harness_sha256 differs (turn_reminder)"]
    assert analyze.setup_differences(first, old) == ["harness_sha256 differs (turn_reminder)"]


# 첫 기준선(2026-09-19 f4ec9a4)은 시뮬레이터 턴 리마인더를 넣기 전이라 하니스 글이 지금과 다른 하나뿐인 옛
# 기록이다(docs/experiments.md 공통 규칙의 2026-10-29 메모, git 이력으로 확인). 그 기록에 채우는
# turn_reminder는 리마인더 없이 상담원의 말을 그대로 보내는 형식을 지금 방법으로 해시한 값과 같아야 하고, 실제
# 기록끼리 짝지으면 다른 차이(시뮬레이터 temperature·프롬프트)와 함께 나온다. 다른 옛 기록끼리의 짝짓기(13단계
# 팔과 8단계 M2b 개발용)는 새 필드 때문에 막히지 않는다. 기록은 manifest만 읽는다.
def test_the_first_baseline_pairs_with_a_different_turn_reminder(monkeypatch):
    first = analyze.old_records_harness(FIRST_BASELINE_COMMIT)
    assert {k: v for k, v in first.items() if k != "turn_reminder"} == {
        k: v for k, v in analyze.OLD_RECORDS_HARNESS_SHA256.items() if k != "turn_reminder"
    }
    monkeypatch.setattr(user_sim, "reminded_message", lambda agent_text, known: agent_text)
    assert first["turn_reminder"] == run.harness_sha256()["turn_reminder"]
    monkeypatch.undo()
    assert analyze.load_manifest(REPORTS / FIRST_BASELINE)["commit"] == FIRST_BASELINE_COMMIT
    assert analyze.setup_differences(REPORTS / FIRST_BASELINE, REPORTS / SECOND_BASELINE) == [
        "config.user_temperature: 0.7 != 0.3",
        "prompt_sha256 differs (user_sim)",
        "harness_sha256 differs (turn_reminder)",
    ]
    for arm in (STAGE13_DEV_C1, STAGE13_DEV_D1):
        assert analyze.setup_differences(REPORTS / DEV_M2B, REPORTS / arm, ignore=tuple(GUARD_AXES)) == []


# 옛 manifest에 채우는 고정값은 2026-10-29 코드의 하니스 글 해시다. 에이전트를 돌린 기록의 커밋 전부에서 그때
# 있던 글이 지금 글과 글자까지 같다는 확인이 근거다(docs/experiments.md 공통 규칙의 2026-10-29 메모. 다른
# 하나인 첫 기준선은 위 테스트). 하니스 글을 일부러 바꾸면 이 테스트가 먼저 실패한다. 그때는 고정값을 바꾸지
# 말고(옛 기록이 본 글은 그대로다) 바꾼 키를 이 단언에서 빼고 날짜 메모를 적는다. 새 실행은 새 해시를 남겨 옛
# 기록과 same-setup에서 걸리는 것이 맞는 동작이다.
def test_the_frozen_hashes_for_old_records_are_those_of_the_current_texts():
    assert analyze.OLD_RECORDS_HARNESS_SHA256 == run.harness_sha256()
