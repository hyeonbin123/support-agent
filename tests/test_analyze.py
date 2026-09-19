from test_tasks import ALL_TASKS, CONTACTS, episode, say_values

from support_agent import analyze
from support_agent.chat import ToolCall


def make_run(tmp_path):
    task = next(t for t in ALL_TASKS if t.id == "smoke-action-01")
    name, contact = CONTACTS[task.customer_id]
    good = [ToolCall("find_customer", {"name": name, "contact": contact})]
    good += [ToolCall(a.tool, a.args) for a in task.gold_actions] + [say_values(task)]
    quiet = good[:-1] + ["취소가 완료되었습니다."]  # the database matches, the refund amount is never said
    results = [episode(task, good), episode(task, quiet), episode(task, ["안 됩니다."]), episode(task, good)]
    for trial, result in enumerate(results):
        result.trial = trial
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    lines = "".join(r.to_json_line() + "\n" for r in results)
    (run_dir / "episodes.jsonl").write_text(lines, encoding="utf-8", newline="\n")
    return run_dir


def test_pass_k_and_the_interval_come_from_the_records(tmp_path):
    by_task = analyze.successes_by_task(analyze.load_episodes(make_run(tmp_path)))
    assert by_task == {"smoke-action-01": [True, False, False, True]}
    assert analyze.pass_k(by_task, 1) == 0.5 and analyze.pass_k(by_task, 2) == 1 / 6
    assert analyze.bootstrap_interval(by_task, 1) == (0.5, 0.5)  # one task: nothing to resample


def test_the_table_is_markdown_with_one_row_per_run(tmp_path):
    text = analyze.table([make_run(tmp_path)])
    assert "| run-a | 1 | 4 | 0 | 0 | 50.0% [50.0%, 50.0%] |" in text
    assert "| run-a | action | 1 | 50.0%" in text and "| run-a | user_stop | 4 |" in text


def test_misses_lists_only_episodes_that_failed_on_the_value_alone(tmp_path):
    text = analyze.misses(make_run(tmp_path))
    assert text.count("## smoke-action-01") == 1 and "취소가 완료되었습니다." in text


def test_the_sample_is_the_same_every_time(tmp_path):
    run_dir = make_run(tmp_path)
    assert analyze.sample(run_dir, 2) == analyze.sample(run_dir, 2)
    assert "(도구) find_customer" in analyze.sample(run_dir, 4)
