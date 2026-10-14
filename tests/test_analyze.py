# [테스트 묶음] analyze.py가 실행 기록(episodes.jsonl)만으로 pass^k, 95% 구간, 마크다운 표, 값만 놓친 에피소드
# 목록, 점검용 표본을 다시 계산하는지 확인한다.
# [왜 중요한가] 이 저장소의 수치는 모두 reports/의 기록에서 analyze로 다시 계산된다 (README "다시 재는 방법").
# 계산이 바뀌면 문서의 수치를 다시 낼 수 없으므로, 손으로 셀 수 있는 작은 실행으로 결과를 고정한다.
# [방법] test_tasks.py의 대본 에이전트 도우미로 스모크 과제 하나를 네 번 돌려 실행 폴더를 만든다. 모델은 쓰지
# 않는다.

from test_tasks import ALL_TASKS, CONTACTS, episode, say_values

from support_agent import analyze
from support_agent.chat import ToolCall


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
