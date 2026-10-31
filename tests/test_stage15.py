"""15단계(읽기 도구 출력 축 X1: get_order에 배송 시각)를 모델 없이 시험한다: 도구 결과와 설명, 실행기 옵션과
manifest, 짝짓기(same-setup --ignore reads), 선택·판정 보고서, 등록한 규칙 문서."""

# [테스트 묶음] docs/experiments.md "## 15단계"가 정한 것을 코드가 그대로 하는지 고정한다.
# - X0(기본): get_order의 결과·설명과 도구 목록이 이 단계 전과 같다(도구 목록 해시가 8단계 M2b 기록과 같다는
#   것은 tests/test_stage10.py, 기록된 도구 호출의 재실행은 tests/test_concurrency.py가 본다).
# - X1: get_order 결과의 shipping 바로 뒤에 shipment(track_shipment와 같은 값·표기, 배송 행이 없으면 None),
#   설명의 둘째 문단만 바뀜, 택배사·송장 번호와 판단 값은 없음, 오류는 X0과 같음, 하니스 글 해시는 그대로.
# - 8단계 M2b 개발용 기록의 도구 호출을 X1로 다시 실행하면 get_order 성공 결과에 shipment만 더해지고, 나머지
#   결과 글과 최종 DB는 기록과 같다.
# - 실행기: --reads가 설정·실행 이름·manifest(축 값마다의 도구 목록 해시)에 남고, 모델에 가는 도구 목록에 X1
#   설명이 들어간다. 등록한 명령으로 새로 돌면 M2b 기록과 reads 말고 같은 설정이다(GPU 전 확인).
# - 짝짓기: same-setup --ignore reads는 도구 목록 해시를 X0 판으로 맞대, 다른 도구 글의 차이는 그대로 잡는다.
# - analyze guards·guards-verdict: X1 팔 이름, 선택 조건 5와 판정 4, 15단계 보고 표, 발동 판정(에이전트 호출).
# - 문서: 15단계 규칙의 설명 글·키 목록·기준 기록·기준 값·명령이 코드와 기록에 맞다.
# [방법] GPU·모델 없이 tests/factories.py의 작은 가게, 커밋된 reports/ 기록(개발용만 대화까지 읽고 시험용은
# manifest만), 임시 폴더의 합성 기록, 가짜 Ollama 제공자를 쓴다.

from __future__ import annotations

import json
import re
import typing
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from conftest import NOW
from factories import extend_tiny_rows
from sqlalchemy import delete
from sqlalchemy.orm import Session
from test_negative_control import episode as conversation
from test_negative_control import write_run
from test_stage8 import RecordingOllama
from test_stage13 import BASE_WINS, of_type, plus, synthetic_run

from support_agent import analyze, db, guard_report, run
from support_agent.agent import visible_tools
from support_agent.chat import ChatResponse
from support_agent.config import RunConfig
from support_agent.paths import REPORTS, ROOT
from support_agent.seed import build_seed_engine
from support_agent.tasks import load_tasks
from support_agent.toolkit import ConversationState, ToolContext, execute
from support_agent.tools import GET_ORDER_X1_DESCRIPTION, READS_VALUES, build_registry

EXPERIMENTS = (ROOT / "docs" / "experiments.md").read_text(encoding="utf-8")
DESIGN = (ROOT / "docs" / "design.md").read_text(encoding="utf-8")
STAGE15 = EXPERIMENTS[EXPERIMENTS.index("## 15단계") : EXPERIMENTS.index("## 보조 지표")]
M2B_DEV = REPORTS / "20261003-200306-qwen3.5-4b-P1-R0-G0-F0-m2b"
M2B_TEST = REPORTS / "20261003-214557-qwen3.5-4b-P1-R0-G0-F0-test-m2b"
X0, X1 = build_registry(), build_registry(reads="X1")
# 이 단계 전 get_order의 도구 설명(모델이 본 글). X0은 이 글 그대로여야 한다.
X0_DESCRIPTION = (
    "주문 한 건의 상품, 금액, 결제, 배송지, 반품·교환 접수, 보상 쿠폰 내역을 조회한다.\n\n"
    "출고·수령 시각은 track_shipment로 본다."
)
X1_SECOND = (
    "배송 상태와 출고·수령 시각, 도착 예정일도 함께 보여 준다. 택배사·송장 번호는 track_shipment로 본다."
)
# track_shipment 결과에서 배송 행의 사실이 아닌 키(주문 번호, 택배사, 송장 번호).
NOT_SHIPMENT_FACTS = ("order_id", "carrier", "tracking_no")
SHIPMENT_KEYS = ["status", "status_label", "shipped_at", "delivered_at", "promised_by"]


# 작은 가게(tests/factories.py: O-1 결제완료, O-2 배송중(출고 기록 없음), O-3 9월 8일 수령, O-4 9월 2일 수령,
# O-5 배송준비중, O-6 취소, O-9는 다른 고객 C-2의 주문)를 새로 만든다.
@pytest.fixture
def shop(tiny_engine):
    store = db.memory_engine(tiny_engine)
    with Session(store) as session:
        extend_tiny_rows(session)
        session.commit()
    yield store
    store.dispose()


# 고객 C-1(기본)로 본인 확인된 P1 문맥. None이면 본인 확인 전이다.
def ctx(customer_id: str | None = "C-1") -> ToolContext:
    return ToolContext(
        now=NOW, enforce_policy=True, state=ConversationState(verified_customer_id=customer_id)
    )


# 성공해야 하는 호출의 JSON 결과.
def shown(registry, store, name: str, **args: Any) -> dict:
    result = execute(registry, store, ctx(), name, args)
    assert result.ok, result.content
    return json.loads(result.content)


# X0 결과의 키 목록에 shipping 바로 뒤 shipment를 끼운 목록(X1 결과의 키 순서).
def with_shipment(keys: list[str]) -> list[str]:
    at = keys.index("shipping") + 1
    return [*keys[:at], "shipment", *keys[at:]]


# seed DB는 만드는 데 1초 넘게 걸려 이 파일에서 한 번만 만든다(에피소드는 그 복사본을 쓴다).
@cache
def seed_engine():
    return build_seed_engine()


# ==================================================================== 도구: X0과 X1


# 축 값은 X0·X1 둘이고 설정의 타입과 같다. 기본은 X0이고, 기본 등록부는 X0 등록부와 같다. 모르는 값은
# 거절한다.
def test_the_axis_has_two_values_and_x0_is_the_default():
    hint = typing.get_type_hints(RunConfig)["reads"]
    assert typing.get_args(hint) == READS_VALUES == ("X0", "X1")
    assert RunConfig().reads == "X0"
    assert build_registry() == build_registry(reads="X0")
    with pytest.raises(ValueError, match="X2"):
        build_registry(reads="X2")


# X1 등록부는 도구 이름·순서·쓰기 표시·인자가 X0과 같고 get_order의 설명과 핸들러만 다르다. X0의 설명은 이
# 단계 전의 글 그대로다. X1 설명은 첫 문단이 같고 둘째 문단만 바뀐다.
def test_x1_changes_only_the_get_order_description():
    assert list(X1) == list(X0)
    for name in X0:
        if name != "get_order":
            assert X1[name] == X0[name], name
    assert X0["get_order"].description == X0_DESCRIPTION
    assert X1["get_order"].description == GET_ORDER_X1_DESCRIPTION
    first, second = GET_ORDER_X1_DESCRIPTION.split("\n\n")
    assert (first, second) == (X0_DESCRIPTION.split("\n\n")[0], X1_SECOND)
    assert X1["get_order"].schema()["parameters"] == X0["get_order"].schema()["parameters"]
    assert (X1["get_order"].write, X1["get_order"].args_model) == (False, X0["get_order"].args_model)
    # 모델에 가는 도구 목록에서 다른 것은 get_order의 description 하나다.
    differing = [
        (a["name"], key)
        for a, b in zip(visible_tools(X0, RunConfig()), visible_tools(X1, RunConfig()), strict=True)
        for key in a
        if a[key] != b[key]
    ]
    assert differing == [("get_order", "description")]


# X1 도구 목록도 16k 컨텍스트 안에 넉넉하다(tests/test_tools.py와 같은 문턱).
def test_the_x1_schemas_stay_small():
    text = json.dumps([spec.schema() for spec in X1.values()], ensure_ascii=False)
    assert len(text) < 9000


# X1의 get_order는 X0 결과의 shipping 바로 뒤에 shipment를 끼운다. 다른 키와 값은 X0과 같고, shipment는 같은
# 주문의 track_shipment 결과에서 주문 번호·택배사·송장 번호를 뺀 것과 값·표기·순서까지 같다.
@pytest.mark.parametrize("order_id", ["O-1", "O-2", "O-3", "O-4", "O-5", "O-6"])
def test_x1_inserts_the_shipment_facts_after_the_shipping_block(shop, order_id):
    before = shown(X0, shop, "get_order", order_id=order_id)
    after = shown(X1, shop, "get_order", order_id=order_id)
    assert list(after) == with_shipment(list(before))
    shipment = after.pop("shipment")
    assert after == before
    tracked = shown(X0, shop, "track_shipment", order_id=order_id)
    assert shipment == {k: v for k, v in tracked.items() if k not in NOT_SHIPMENT_FACTS}
    assert list(shipment) == SHIPMENT_KEYS


# 날짜 표기: 9월 2일 수령한 O-4는 출고·수령 시각(KST 'YYYY-MM-DD HH:MM')과 도착 예정일(날짜)을, 결제완료 O-1은
# 아직 출고·수령 전이라 None을 준다.
def test_x1_shows_kst_times_and_nulls_before_shipping(shop):
    assert shown(X1, shop, "get_order", order_id="O-4")["shipment"] == {
        "status": "delivered",
        "status_label": "배송완료",
        "shipped_at": "2026-08-29 09:00",
        "delivered_at": "2026-09-02 14:20",
        "promised_by": "2026-08-30",
    }
    pending = shown(X1, shop, "get_order", order_id="O-1")["shipment"]
    assert (pending["shipped_at"], pending["delivered_at"], pending["promised_by"]) == (
        None,
        None,
        "2026-09-13",
    )


# 배송 행이 없는 주문: X1의 get_order는 shipment None으로 성공하고(track_shipment는 shipment_not_found),
# 나머지 결과는 X0과 같다.
def test_x1_gives_none_for_an_order_without_a_shipment_row(shop):
    with Session(shop) as session:
        session.execute(delete(db.Shipment).where(db.Shipment.order_id == "O-5"))
        session.commit()
    after = shown(X1, shop, "get_order", order_id="O-5")
    assert after.pop("shipment") is None
    assert after == shown(X0, shop, "get_order", order_id="O-5")
    assert execute(X1, shop, ctx(), "track_shipment", {"order_id": "O-5"}).error_code == "shipment_not_found"


# 사실만 준다: 택배사·송장 번호(track_shipment에만 있음)와 계산된 판단(반품 기한 2026-09-09, 'deadline',
# 'returnable')이 X1 결과에 없다(docs/experiments.md 공통 규칙 "도구와 규정").
def test_x1_gives_facts_not_judgements_and_keeps_the_carrier_in_track_shipment(shop):
    text = execute(X1, shop, ctx(), "get_order", {"order_id": "O-4"}).content
    tracked = shown(X0, shop, "track_shipment", order_id="O-4")
    assert tracked["carrier"] not in text and tracked["tracking_no"] not in text
    assert "2026-09-09" not in text and "deadline" not in text and "returnable" not in text


# 오류는 X0과 글자까지 같다(본인 확인 전, 없는 주문, 다른 고객의 주문).
@pytest.mark.parametrize(
    ("customer", "order_id", "code"),
    [
        (None, "O-3", "identity_not_verified"),
        ("C-1", "O-404", "order_not_found"),
        ("C-1", "O-9", "not_order_owner"),
    ],
)
def test_x1_fails_like_x0(shop, customer, order_id, code):
    x0, x1 = (execute(r, shop, ctx(customer), "get_order", {"order_id": order_id}) for r in (X0, X1))
    assert x0 == x1 and x0.error_code == code


# X1은 새 검사를 더하지 않으므로 도구 오류 글(하니스 글 해시의 tool_errors)을 비롯한 하니스 글이 그대로다.
def test_x1_leaves_the_harness_texts_alone():
    assert run.harness_sha256() == analyze.OLD_RECORDS_HARNESS_SHA256


# 8단계 M2b 개발용 기록의 도구 호출을 기록된 순서와 인자 그대로 X1로 다시 실행한다. get_order 성공 결과에는
# shipping 뒤 shipment만 더해지고(그 시점의 track_shipment와 같은 값) 나머지는 기록의 글과 같다. 다른 호출의
# 결과 글과 에피소드가 끝난 DB의 해시도 기록과 같다(X1은 읽기 결과만 바꾼다).
def test_x1_adds_only_the_shipment_to_the_recorded_get_order_results():
    manifest = analyze.load_manifest(M2B_DEV)
    assert manifest["config"]["policy"] == "P1" and manifest["config"].get("voice", "V0") == "V0"
    tasks = {task.id: task for task in load_tasks(manifest["tasks_file"])}
    orders = with_times = 0
    for episode in analyze.load_episodes(M2B_DEV):
        if episode["status"] == "infra_error":
            continue
        store = db.memory_engine(seed_engine())
        context = ToolContext(now=tasks[episode["task_id"]].now, enforce_policy=True)
        where = (episode["task_id"], episode["trial"])
        for call in episode["tool_calls"]:
            result = execute(X1, store, context, call["name"], call["raw_arguments"])
            if call["name"] != "get_order" or not call["ok"]:
                assert (result.ok, result.content) == (call["ok"], call["content"]), (*where, call["name"])
                continue
            recorded, now = json.loads(call["content"]), json.loads(result.content)
            assert list(now) == with_shipment(list(recorded)), where
            shipment = now.pop("shipment")
            assert now == recorded, where
            state = ConversationState(**context.state.to_dict())
            look = ToolContext(now=context.now, enforce_policy=True, state=state)
            tracked = execute(X0, store, look, "track_shipment", {"order_id": recorded["order_id"]})
            facts = json.loads(tracked.content) if tracked.ok else None
            expected = (
                None if facts is None else {k: v for k, v in facts.items() if k not in NOT_SHIPMENT_FACTS}
            )
            assert shipment == expected, where
            orders += 1
            with_times += bool(shipment and (shipment["shipped_at"] or shipment["delivered_at"]))
        assert db.state_hash(db.dump_db(store)) == episode["final_hash"], where
        store.dispose()
    assert orders == 66 and with_times > 0  # 기록의 get_order 성공 66번(15단계 기준 행)


# ==================================================================== 실행기와 manifest


class ToolsSeen(RecordingOllama):
    """에이전트(도구와 함께 불림)에게 간 도구 목록을 남기는 가짜 Ollama. 고객은 대화마다 한 번 묻고 끝낸다."""

    seen: list[list[dict]] = []

    def chat(self, messages, tools=(), *, temperature=0.0, seed=None, max_tokens=1024):
        if tools:
            ToolsSeen.seen.append([dict(t) for t in tools])
            return ChatResponse("네, 확인해 보겠습니다.")
        asked = any(m.role == "assistant" for m in messages)
        return ChatResponse("###STOP###" if asked else "주문 배송 상황을 알려 주세요.")


# 가짜 Ollama로 실행기를 끝까지 돌리고 실행 폴더와 manifest를 돌려준다. 실제 모델·GPU 검사·출력 폴더를 바꾼다.
def start(tmp_path, monkeypatch, *argv: str) -> tuple[Path, dict]:
    ToolsSeen.seen = []
    monkeypatch.setattr(run, "OllamaProvider", ToolsSeen)
    monkeypatch.setattr(run, "gpu_used_by_others_mib", lambda provider: None)
    monkeypatch.setattr(run, "OUTPUTS", tmp_path)
    monkeypatch.setattr("sys.argv", ["run", *argv])
    run.main()
    (run_dir,) = (tmp_path / "runs").iterdir()
    return run_dir, json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


SMOKE_ONE = ("--tasks", "smoke", "--task-id", "smoke-lookup-01", "--label", "unit")


# 실행기가 manifest에 남기는 도구 목록 해시와 같은 셈.
def tools_sha256(registry, config: RunConfig) -> str:
    return run.text_sha256(json.dumps(visible_tools(registry, config), ensure_ascii=False, sort_keys=True))


# 에이전트에게 간 도구 목록마다의 get_order 설명.
def get_order_descriptions() -> list[str]:
    return [t["description"] for listed in ToolsSeen.seen for t in listed if t["name"] == "get_order"]


# --reads X1이 설정과 실행 이름(-X1-)에 남고, manifest에 축 값마다의 도구 목록 해시가 남는다. 이 실행의 도구
# 해시(prompt_sha256.tools)는 X1 판이고, 에이전트에게 간 도구 목록에 X1 설명이 들어 있다.
def test_the_runner_records_x1(tmp_path, monkeypatch):
    _, manifest = start(tmp_path, monkeypatch, *SMOKE_ONE, "--reads", "X1")
    assert manifest["config"]["reads"] == "X1"
    assert manifest["run_id"].endswith("-P0-R0-G0-F0-X1-unit")
    by_reads = manifest["tools_sha256_by_reads"]
    assert set(by_reads) == set(READS_VALUES)
    assert by_reads["X1"] == manifest["prompt_sha256"]["tools"] != by_reads["X0"]
    assert by_reads["X0"] == tools_sha256(X0, RunConfig())
    descriptions = get_order_descriptions()
    assert descriptions and set(descriptions) == {GET_ORDER_X1_DESCRIPTION}


# 기본(--reads 없음)은 X0: 실행 이름 꼴이 그대로이고, 도구 해시는 X0 판이며 모델은 X0 설명을 본다.
def test_the_default_runner_stays_x0(tmp_path, monkeypatch):
    _, manifest = start(tmp_path, monkeypatch, *SMOKE_ONE)
    assert manifest["config"]["reads"] == "X0"
    assert manifest["run_id"].endswith("-P0-R0-G0-F0-unit")
    assert manifest["prompt_sha256"]["tools"] == manifest["tools_sha256_by_reads"]["X0"]
    descriptions = get_order_descriptions()
    assert descriptions and set(descriptions) == {X0_DESCRIPTION}


def registered_commands() -> list[list[str]]:
    """문서 15단계 '명령'의 실행기 줄들(스모크, 개발용, 시험용 순)의 인자."""
    lines = re.findall(r"^uv run --no-sync python -m support_agent\.run (.*)$", STAGE15, re.MULTILINE)
    return [line.split() for line in lines]


# 문서의 실행기 명령 셋: 13단계와 같은 M2b 구성(P1, qwen3.5:4b think off, num_ctx 12288, 시뮬레이터 12288)에
# --reads X1만 더하고 가드 축은 켜지 않는다. 스모크는 outputs/, 개발용·시험용은 공식 기록(시험용은
# --allow-test).
def test_the_registered_commands_add_only_reads_to_the_m2b_setup():
    smoke, dev, test = registered_commands()
    setup = (
        "--policy P1 --model qwen3.5:4b --think off --num-ctx 12288 --user-num-ctx 12288 --reads X1".split()
    )
    heads = (["--tasks", "smoke", "--trials", "1"], ["--tasks", "dev", "--trials", "4"])
    heads += (["--tasks", "test", "--trials", "4"],)
    for command, head in zip((smoke, dev, test), heads, strict=True):
        assert command[:4] == head and command[4 : 4 + len(setup)] == setup
        assert not {"--claims", "--guard", "--errors", "--repeat", "--force"} & set(command)
    assert "--official" not in smoke and "--official" in dev and "--official" in test
    assert "--allow-test" in test and "--allow-test" not in dev
    labels = [command[command.index("--label") + 1] for command in (smoke, dev, test)]
    assert labels == ["smoke-s15", "s15-x1", "s15-test"]


# GPU 전 확인: 문서의 개발용 명령(--official만 빼고, 빠르게 dev-001만)을 가짜 모델로 돌린 manifest가 M2b
# 개발용 기록과 reads 하나만 다르다. 설정(기록에 없는 나중 필드는 기본값), 규정·시스템·시뮬레이터 프롬프트
# 해시, 하니스 글 해시가 같고, 도구 목록은 X0 판 해시가 기록의 것과 같다. same-setup --ignore reads에 남는
# 차이는 가짜 제공자와 일부 과제만 돈 것 때문인 것뿐이다.
def test_a_new_x1_run_pairs_with_the_m2b_record(tmp_path, monkeypatch):
    _, dev, _ = registered_commands()
    argv = [arg for arg in dev if arg != "--official"] + ["--task-id", "dev-001"]
    run_dir, manifest = start(tmp_path, monkeypatch, *argv)
    record = analyze.load_manifest(M2B_DEV)
    defaults = RunConfig().to_dict()
    config = manifest["config"]
    assert {k: v for k, v in config.items() if k != "reads"} == {
        k: record["config"].get(k, defaults[k]) for k in config if k != "reads"
    }
    assert (config["reads"], record["config"].get("reads", "X0")) == ("X1", "X0")
    assert {k: v for k, v in manifest["prompt_sha256"].items() if k != "tools"} == {
        k: v for k, v in record["prompt_sha256"].items() if k != "tools"
    }
    assert manifest["tools_sha256_by_reads"]["X0"] == record["prompt_sha256"]["tools"]
    assert manifest["harness_sha256"] == analyze.old_records_harness(record.get("commit"))
    expected = ("task_sha256 differs", "judged trials of ", "agent_provider differs", "user_provider differs")
    left = [
        p
        for p in analyze.setup_differences(M2B_DEV, run_dir, ignore=("reads",))
        if not p.startswith(expected)
    ]
    assert left == []


# ==================================================================== 짝짓기: same-setup --ignore reads


# 합성 manifest 한 쌍: 기준(X0, 옛 기록처럼 축 값별 해시 없음, 도구 해시 t0)과 후보(reads, 도구 해시, 축 값별
# 해시).
def pair(folder: Path, *, reads: str, tools: str, by_reads: dict | None) -> tuple[Path, Path]:
    episodes = [conversation("dev-001", 0, [])]
    base = write_run(folder, "base", episodes, prompts={"policy": "p", "tools": "t0"})
    cand = write_run(folder, "cand", episodes, axes={"reads": reads}, prompts={"policy": "p", "tools": tools})
    if by_reads is not None:
        manifest = json.loads((cand / "manifest.json").read_text(encoding="utf-8"))
        manifest["tools_sha256_by_reads"] = by_reads
        (cand / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return base, cand


# reads만 다르고 X1 실행의 X0 판 도구 해시가 기준의 도구 해시와 같으면 --ignore reads로 짝지어진다. 무시하지
# 않으면 설정과 도구 해시가 다르다고 낸다.
def test_ignoring_reads_compares_the_x0_tool_hashes(tmp_path):
    base, cand = pair(tmp_path, reads="X1", tools="t1", by_reads={"X0": "t0", "X1": "t1"})
    assert analyze.setup_differences(base, cand, ignore=("reads",)) == []
    assert analyze.setup_differences(base, cand) == [
        "config.reads: 'X0' != 'X1'",
        "prompt_sha256 differs (tools)",
    ]


# 다른 도구 글도 달라졌으면(X1 실행의 X0 판 해시가 기준과 다름) --ignore reads로도 짝짓지 않는다. 도구 해시를
# 통째로 빼면 이 차이까지 가려진다.
def test_ignoring_reads_still_catches_other_tool_changes(tmp_path):
    base, cand = pair(tmp_path, reads="X1", tools="t1", by_reads={"X0": "other", "X1": "t1"})
    assert analyze.setup_differences(base, cand, ignore=("reads",)) == ["prompt_sha256 differs (tools)"]


# 축 값별 해시가 manifest의 도구 해시와 맞지 않거나(X1인데 X1 판이 다름), X1인데 축 값별 해시가 없으면
# 실행기가 만든 manifest가 아니므로 짝짓지 않는다.
@pytest.mark.parametrize("by_reads", [{"X0": "t0", "X1": "elsewhere"}, None], ids=["mismatch", "missing"])
def test_inconsistent_x1_manifests_do_not_pair(tmp_path, by_reads):
    base, cand = pair(tmp_path, reads="X1", tools="t1", by_reads=by_reads)
    problems = analyze.setup_differences(base, cand, ignore=("reads",))
    assert any("tools_sha256_by_reads" in p for p in problems), problems


# X0 실행끼리는 지금처럼 도구 해시를 그대로 맞댄다: 축 값별 해시가 없는 옛 기록과 새 X0 실행은 reads를
# 무시하든 안 하든, 13단계처럼 가드 축과 함께 무시해도 짝지어진다.
def test_x0_runs_pair_as_before(tmp_path):
    base, cand = pair(tmp_path, reads="X0", tools="t0", by_reads={"X0": "t0", "X1": "t1"})
    assert analyze.setup_differences(base, cand) == []
    assert analyze.setup_differences(base, cand, ignore=("reads",)) == []
    assert analyze.setup_differences(base, cand, ignore=tuple(guard_report.GUARD_AXES)) == []


# ==================================================================== analyze guards·guards-verdict


# 축 이름: reads가 가드 축과 함께 짝짓기에서 무시되고, 팔 이름은 X1, 같은 성공률일 때의 순서는 13단계 넷 뒤에
# X1, 판정 값과 같은 종류의 사실을 주는 팔은 E1·X1이다.
def test_x1_is_a_named_arm():
    assert guard_report.GUARD_AXES["reads"] == "X0"
    assert guard_report.TIE_ORDER == ("C1", "G1", "D1", "E1", "X1")
    assert guard_report.FACT_ARMS == ("E1", "X1")
    assert guard_report.arm_name(["X1"]) == "X1" and guard_report.arm_name(["X1", "C1"]) == "C1+X1"


# 거절 유형이 오르고 다른 유형이 그대로인 X1 팔은 다섯 조건을 모두 만족해 고른다. 보고서에 15단계 보고 표가
# 붙는다.
def test_guards_select_x1_when_every_condition_holds(tmp_path):
    base = synthetic_run(tmp_path, "base", {}, BASE_WINS)
    x1 = synthetic_run(tmp_path, "x1", {"reads": "X1"}, plus([(t, 2) for t in of_type("refusal")[:3]]))
    report, chosen = guard_report.select(base, [x1])
    assert chosen == "X1"
    assert "| x1 | X1 |" in report and "거절이 아닌 유형 pass^1 ≥ 기준 예" in report
    assert "배송 시각을 본 거절 에피소드" in report


# 조건 5(거절이 아닌 세 유형이 기준보다 낮지 않다)는 X1에도 걸린다: 거절 유형이 넷 올라도 다른 유형이 하나
# 내려가면 고르지 않는다.
def test_guards_reject_x1_when_other_types_drop(tmp_path):
    base = synthetic_run(tmp_path, "base", {}, BASE_WINS)
    lost = (of_type("lookup")[0], 0)
    x1 = synthetic_run(
        tmp_path, "x1", {"reads": "X1"}, plus([(t, 2) for t in of_type("refusal")[:4]], [lost])
    )
    report, chosen = guard_report.select(base, [x1])
    assert chosen is None and "거절이 아닌 유형 pass^1 ≥ 기준 아니오" in report


# 13단계 팔만 비교하면 15단계 보고 표는 붙지 않는다(13단계 보고서는 그대로다).
def test_stage13_reports_have_no_reads_table(tmp_path):
    base = synthetic_run(tmp_path, "base", {}, BASE_WINS)
    e1 = synthetic_run(tmp_path, "e1", {"errors": "E1"}, plus([(t, 2) for t in of_type("refusal")[:3]]))
    report, chosen = guard_report.select(base, [e1])
    assert chosen == "E1" and "배송 시각을 본 거절 에피소드" not in report
    assert "배송 시각을 본 거절 에피소드" not in guard_report.verdict(base, e1)[0]


# 시험용 판정: X1이면 판정 4가 X1 이름으로 나오고, 올랐고 다른 유형이 내려가지 않으면 "개선", 거절이 아닌
# 유형이 내려가면 "개선을 확인하지 못함 (X1: …)"이다. 보고 표가 붙는다.
def test_the_x1_verdict_applies_the_other_types_condition(tmp_path):
    base = synthetic_run(tmp_path, "base", {}, BASE_WINS)
    refusal_all = [(t, k) for t in of_type("refusal") for k in (2, 3)]
    better = synthetic_run(tmp_path, "better", {"reads": "X1"}, plus(refusal_all))
    report, result = guard_report.verdict(base, better)
    assert result == "개선" and "4. (X1) 거절이 아닌 유형 pass^1" in report
    assert "배송 시각을 본 거절 에피소드" in report
    dropped = synthetic_run(
        tmp_path, "dropped", {"reads": "X1"}, plus(refusal_all, [(of_type("action")[0], 0)])
    )
    assert guard_report.verdict(base, dropped)[1] == "개선을 확인하지 못함 (X1: 거절이 아닌 유형이 내려감)"


# 배송 시각을 본 에피소드: 성공한 track_shipment, 또는 shipment가 있는 get_order 결과(X1). X0의 get_order
# 결과와 실패한 track_shipment는 아니다.
def test_saw_shipment_times_reads_track_shipment_and_x1_get_order(shop):
    x0 = execute(X0, shop, ctx(), "get_order", {"order_id": "O-4"})
    x1 = execute(X1, shop, ctx(), "get_order", {"order_id": "O-4"})
    tracked = execute(X0, shop, ctx(), "track_shipment", {"order_id": "O-4"})
    failed = execute(X0, shop, ctx(None), "track_shipment", {"order_id": "O-4"})

    def saw(*calls: tuple[str, Any]) -> bool:
        tool_calls = [{"name": name, "ok": r.ok, "content": r.content, "ms": 2.0} for name, r in calls]
        return guard_report.saw_shipment_times({"tool_calls": tool_calls})

    assert not saw(("get_order", x0)) and not saw(("track_shipment", failed))
    assert saw(("get_order", x0), ("track_shipment", tracked)) and saw(("get_order", x1))


# 발동 판정: X1은 도구 목록(모든 에이전트 호출의 입력)의 설명을 바꾸므로 에이전트 호출이 한 번이라도 있으면
# 작동이다. 그래서 X1 팔의 음성 대조에서는 에이전트 호출이 있는 짝이 모두 '장치 작동'이고 미발동 짝이 없다.
def test_x1_fires_whenever_the_agent_was_called(tmp_path):
    turns = [("주문 확인해 주세요", "배송중입니다.")]
    assert guard_report.FIRED["reads"](conversation("dev-001", 0, turns))
    assert not guard_report.FIRED["reads"](conversation("dev-001", 1, []))
    episodes = [conversation(t, k, turns) for t in ("dev-001", "dev-002") for k in (0, 1)]
    base = write_run(tmp_path, "base", episodes)
    arm = write_run(tmp_path, "arm", episodes, axes={"reads": "X1"})
    nc = guard_report.negative_control(base, arm)
    assert (nc["axes"], nc["paired"], nc["fired"], nc["clean"]) == (["reads"], 4, 4, 0)


# ==================================================================== 문서 (15단계 규칙)


# 15단계 절은 14단계 뒤, 보조 지표 앞에 있다. 설명 두 판(X0 둘째 문단, X1 둘째 문단)과 shipment 키 목록, 기준
# 두 기록, 선택·판정 문턱이 코드와 같다. docs/design.md 도구 표 아래에도 X1 줄이 있다.
def test_the_stage15_rule_matches_the_code():
    assert EXPERIMENTS.index("## 14단계") < EXPERIMENTS.index("## 15단계") < EXPERIMENTS.index("## 보조 지표")
    assert X1_SECOND in STAGE15 and X0_DESCRIPTION.split("\n\n")[1] in STAGE15
    keys = f"`shipment{{{', '.join(SHIPMENT_KEYS)}}}`"
    assert keys in STAGE15 and keys in DESIGN and X1_SECOND in DESIGN
    assert M2B_DEV.name in STAGE15 and M2B_TEST.name in STAGE15
    for text in (
        "+3.1%p (96 에피소드 중 3)",
        "기준 + 2 (기준 3 → 5)",
        "합 ≤ 2",
        "기준 + 3 (13 → 16)",
        "거절이 아닌 세 유형",
        "`--ignore reads`",
        "`tools_sha256_by_reads`",
    ):
        assert text in STAGE15, text
    assert guard_report.MIN_GAIN == 3 / 96 and guard_report.DEV_WRITE_SLACK == 2
    assert guard_report.TEST_WRITE_SLACK == 3 and guard_report.MAX_FORMAT_AND_CLAIM_ENDINGS == 2


# 기준 값: 문서의 기준 행은 analyze guards가 M2b 개발용 기록에서 내는 15단계 보고 표의 행 그대로다.
def test_the_registered_baseline_row_is_the_record():
    stats = {M2B_DEV.name: guard_report.run_stats(M2B_DEV)}
    lines = guard_report.reads_table([M2B_DEV.name], {M2B_DEV.name: "기준"}, stats)
    (row,) = [line for line in lines if line.startswith(f"| {M2B_DEV.name} ")]
    assert row == f"| {M2B_DEV.name} | 기준 | 80 (66) | 25 (25) | 8/24 | 4431 | 5.1 |"
    assert row in STAGE15


# 왜 하는가의 수: M2b 개발용 거절 24 에피소드의 (배송 시각을 봄, 성공) 교차표와, 보지 않고 실패한 15개 가운데
# get_order는 성공했고 전달 값이 배송 시각인 과제(dev-015·016·017·019)의 에피소드 수(X1이면 날짜가 보였을
# 실패)를 기록에서 다시 센다.
def test_the_registered_refusal_counts_are_the_record():
    types = analyze.load_task_types()
    refusal = [
        e
        for e in analyze.load_episodes(M2B_DEV)
        if e["status"] != "infra_error" and types[e["task_id"]] == "refusal"
    ]
    table: dict[tuple[bool, bool], int] = {}
    for e in refusal:
        key = (guard_report.saw_shipment_times(e), analyze.is_success(e))
        table[key] = table.get(key, 0) + 1
    assert table == {(False, False): 15, (True, True): 7, (True, False): 1, (False, True): 1}
    reachable = [
        e
        for e in refusal
        if not guard_report.saw_shipment_times(e)
        and not analyze.is_success(e)
        and e["task_id"] in ("dev-015", "dev-016", "dev-017", "dev-019")
        and any(t["ok"] and t["name"] == "get_order" for t in e["tool_calls"])
    ]
    assert len(reachable) == 7
    assert "(안 봄, 실패) 15, (봄, 성공) 7, (봄, 실패) 1, (안 봄, 성공) 1" in STAGE15
    assert "이 7개" in STAGE15


# 시험용 기준 기록은 manifest만 읽는다: 시험용 40과제, 도구 해시가 지금 코드의 X0 판과 같다(그래서 시험용 X1
# 실행도 --ignore reads로 짝지어진다).
def test_the_test_baseline_tools_hash_is_the_x0_hash():
    record = analyze.load_manifest(M2B_TEST)
    assert len(record["task_sha256"]) == 40
    config = RunConfig(model="qwen3.5:4b", policy="P1", num_ctx=12288, user_num_ctx=12288)
    assert record["prompt_sha256"]["tools"] == tools_sha256(X0, config)


# tests/test_concurrency.py는 기록을 그 기록의 reads 값으로 만든 등록부로 다시 실행한다(X1 기록이 reports/에
# 더해져도 같은 검사가 돈다).
def test_the_replay_test_builds_the_registry_of_the_record():
    source = (Path(__file__).parent / "test_concurrency.py").read_text(encoding="utf-8")
    assert 'build_registry(caller_id=caller_id, reads=config.get("reads", "X0"))' in source
