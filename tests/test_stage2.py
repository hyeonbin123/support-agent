"""2단계 후보: 정체 가드(G1)와 글로 새어 나온 도구 호출 살리기(F1)."""

# [파일 역할] 2단계(개선 실험) 후보 가운데 에이전트 루프에 들어간 것들을 검증하는 테스트 묶음이다: 멈춤 가드
# G1·G2, 본문에 샌 도구 호출 구제 F1, 그리고 비교 표를 만드는 analyze의 짝지은 부트스트랩. 끝에는 서비스에만
# 켜는 언어 가드(L1) 테스트도 있다.
# [왜 중요한가] 2단계는 기준(P0·R0·G0·F0)에서 한 축씩만 바꿔 개발용 pass^1 차이로 후보를
# 고른다(docs/experiments.md 2단계 "후보", "선택 규칙"). 후보 코드가 의도와 다르게 동작하거나
# 기본값(G0·F0·L0)에서도 동작이 바뀌면, 후보의 효과와 기준이 섞여 비교가 무의미해진다.
# [방법] 장난감 등록부(toy_tools)와 작은 DB 복사본 위에서 agent_turn 한 턴을 대본 제공자로 돌리고, 메시지
# 기록·LLM 호출 기록·DB 상태를 확인한다.

from __future__ import annotations

# NOW는 conftest의 고정 시각, TOY_REGISTRY는 도메인 규칙이 없는 장난감 도구 등록부다.
import pytest
from conftest import NOW
from toy_tools import TOY_REGISTRY

# agent: 검증 대상인 한 턴 실행(agent_turn)과 판별 함수(is_stall, leaked_tool_call, in_another_language), 다시
# 시킬 때 넣는 안내문. analyze: 개발용 비교 표의 짝지은 차이 계산.
from support_agent import analyze, db
from support_agent.agent import (
    LANGUAGE_NOTICE,
    STALL_NOTICE,
    agent_turn,
    in_another_language,
    is_stall,
    leaked_tool_call,
    new_state,
)
from support_agent.chat import ScriptedProvider, ToolCall
from support_agent.config import RunConfig
from support_agent.toolkit import ToolContext, execute


# 고객이 'O-1 주문을 취소해 주세요.'라고 한 턴을 대본대로 돌린다. config로 guard·rescue·language 같은
# RunConfig 축을 바꾼다.
# 에피소드 실행기처럼 DB 복사본과 ToolContext를 클로저로 감싼 run_tool을 넘긴다(루프는 DB를 직접 보지 않는
# 구조). 요청 기록을 보려고 마지막 제공자를 turn.provider에 붙여 둔다.
def turn(tiny_engine, script, **config):
    engine = db.memory_engine(tiny_engine)
    ctx = ToolContext(now=NOW)
    state = new_state("system prompt")
    provider = ScriptedProvider(script)
    result = agent_turn(
        state,
        "O-1 주문을 취소해 주세요.",
        provider=provider,
        registry=TOY_REGISTRY,
        run_tool=lambda name, args: execute(TOY_REGISTRY, engine, ctx, name, args),
        config=RunConfig(**config),
    )
    turn.provider = provider
    return result, state, engine


# 도구 호출 없이 '하겠다'는 말만 하고 고객에게 묻는 것이 없는 응답을 멈춤으로 잡는지 본다. 1단계 기준선 실패
# 분류에서 멈춤이 31건으로 많았고, G1의 판별 정규식은 개발용 기준선 에피소드만 보고 정했다(docs/experiments.md
# 2단계).
@pytest.mark.parametrize(
    "text",
    [
        "확인해 보겠습니다. 잠시만 기다려 주세요.",
        "네, 주문 내역을 조회해 보겠습니다.",
        "취소 처리해 드리겠습니다.",
        "상담원에게 연결하겠습니다.",
    ],
)
def test_a_reply_that_only_promises_is_a_stall(text):
    assert is_stall(text)


# 고객에게 묻는 말(물음표, '알려 주세요'), 결과 보고, 거절 안내는 멈춤이 아닌지 본다. 이것을 멈춤으로 잡으면
# 정상적인 본인 확인 질문까지 막아 오히려 성공률을 깎는다.
@pytest.mark.parametrize(
    "text",
    [
        "확인해 보겠습니다. 성함과 가입하신 전화번호를 알려 주세요.",
        "O-1 주문을 단순 변심으로 취소할까요?",
        "취소되었습니다. 38,900원이 환불됩니다.",
        "수령 후 7일이 지나 반품이 어렵습니다.",
    ],
)
def test_asking_or_reporting_is_not_a_stall(text):
    assert not is_stall(text)


# G1: 멈춤 응답은 고객에게 보내지 않고(delivered=False), 하니스 안내문(STALL_NOTICE, harness=True)을 넣어 다시
# 시킨 뒤 다음 호출의 도구 실행이 실제로 DB를 바꾸는지 본다. 멈춤은 stalls로 세고 형식 오류로는 세지 않으며,
# LLM 호출 기록에는 'stall'로 남는다.
def test_g1_holds_a_stall_back_and_the_retry_acts(tiny_engine):
    script = [
        "확인해 보겠습니다. 잠시만 기다려 주세요.",
        ToolCall("toy_cancel_order", {"order_id": "O-1"}),
        "취소했습니다.",
    ]
    result, state, engine = turn(tiny_engine, script, guard="G1")
    assert result.reply == "취소했습니다." and state.stalls == 1 and state.format_errors == 0
    held, notice = state.messages[3], state.messages[4]
    assert (held.delivered, notice.harness, notice.content) == (False, True, STALL_NOTICE)
    assert [log.format_error for log in state.llm_log] == ["stall", None, None]
    assert db.dump_db(engine)["orders"][0]["status"] == "cancelled"


# 기본값(G0)에서는 같은 멈춤 응답이 그대로 전달되고 stalls가 0인지 본다. 기준 구성이 가드에 영향받지 않아야
# 짝지은 비교가 성립한다.
def test_g0_delivers_the_same_reply(tiny_engine):
    result, state, _ = turn(tiny_engine, ["확인해 보겠습니다. 잠시만 기다려 주세요."])
    assert result.reply.startswith("확인해 보겠습니다") and state.stalls == 0


# G1이 한 턴에 두 번 돌려보낸 뒤에도 멈춤이면 포기하고 그 응답을 전달하는지(턴을 끝내지 않음, stop=None) 본다.
# 규칙 'G1은 한 턴에 2번까지, 넘으면 그대로 전달'(docs/experiments.md 2단계 후보 표)을 지킨다.
def test_g1_gives_up_after_its_retries_and_delivers_the_reply(tiny_engine):
    result, state, _ = turn(tiny_engine, ["잠시만 기다려 주세요."] * 3, guard="G1")
    assert result.reply == "잠시만 기다려 주세요." and result.stop is None and state.stalls == 2


# 본문에 샌 도구 호출을 뽑는 leaked_tool_call을 본다. qwen2.5가 실제로 남긴 모양(앞의 잡글과 </tool_call>
# 태그)에서 name과 arguments를 꺼내고, arguments가 없거나 JSON이 아닌 중괄호(주문 {O-1})는 호출로 보지 않는다.
# 잘못 뽑으면 엉뚱한 쓰기가 실행되므로 보수적으로 판별해야 한다.
def test_leaked_tool_call_parsing():
    call = leaked_tool_call(
        'portun\n{"name": "toy_get_order", "arguments": {"order_id": "O-1"}}\n</tool_call>'
    )
    assert call == ToolCall("toy_get_order", {"order_id": "O-1"})
    assert leaked_tool_call('{"name": "toy_get_order"}') is None
    assert leaked_tool_call("주문 {O-1}을 확인했습니다.") is None


# F1: 샌 호출을 실제로 실행해 DB가 바뀌되, 형식 오류 수에는 그대로 세고(format_errors 1) 기록에는
# 'rescued_tool_call'로 남는지 본다. 규칙에 '형식 오류로는 그대로 센다'고 정했기 때문에, 구제가 형식 오류
# 지표를 숨기지 않게 한다. 샌 JSON 글은 고객에게 전달되지 않아야 한다.
def test_f1_runs_the_leaked_call_and_still_counts_the_format_error(tiny_engine):
    leaked = 'portun\n{"name": "toy_cancel_order", "arguments": {"order_id": "O-1"}}\n</tool_call>'
    result, state, engine = turn(tiny_engine, [leaked, "취소했습니다."], rescue="F1")
    assert result.reply == "취소했습니다." and state.format_errors == 1
    assert state.llm_log[0].format_error == "rescued_tool_call"
    assert [log.name for log in state.tool_log] == ["toy_cancel_order"]
    assert db.dump_db(engine)["orders"][0]["status"] == "cancelled"
    assert all(m.delivered for m in state.messages if m.role == "assistant")  # 고객에게 새어 나간 것이 없다


# 기본값(F0)에서는 샌 호출을 실행하지 않고 형식 오류(leaked_tool_call)로 돌려보내, 도구 기록이 비고 주문이
# 그대로(paid)인지 본다.
def test_f0_sends_the_leaked_call_back(tiny_engine):
    leaked = '{"name": "toy_cancel_order", "arguments": {"order_id": "O-1"}}'
    _, state, engine = turn(tiny_engine, [leaked, "취소하려면 도구가 필요합니다."])
    assert state.llm_log[0].format_error == "leaked_tool_call" and state.tool_log == []
    assert db.dump_db(engine)["orders"][0]["status"] == "paid"


# analyze.paired_difference: 두 실행의 pass^k 차이를 과제 단위로 짝지은 채 부트스트랩하는지 본다. 차이의
# 점추정이 과제별 차이의 평균((0.5+0.25+0)/3)과 같고 구간이 그것을 감싸는지, 같은 실행끼리는 (0, 0, 0)인지,
# 과제 집합이 다르면 ValueError인지 확인한다. 같은 과제를 두 설정으로 돌렸으므로 과제를 짝지어 다시 뽑아야
# 구간이 과제 난이도 차이에 덜 흔들린다(analyze.py docstring).
def test_paired_difference_resamples_tasks_and_keeps_pairs():
    base = {"a": [True, False, False, False], "b": [False] * 4, "c": [True] * 4}
    better = {"a": [True, True, True, False], "b": [True, False, False, False], "c": [True] * 4}
    diff, low, high = analyze.paired_difference(base, better, 1)
    assert diff == pytest.approx((0.5 + 0.25 + 0.0) / 3) and low <= diff <= high and low >= 0.0
    same = analyze.paired_difference(base, base, 1)
    assert same == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="same tasks"):
        analyze.paired_difference(base, {"a": [True] * 4}, 1)


# G2: 멈춤으로 돌려보낸 뒤 그 턴의 남은 호출만 temperature 0.7로 뽑고, 첫 호출은 0.0 그대로인지 본다. G1에서
# temperature 0인 모델이 안내문을 받고도 같은 문장을 되풀이해 호출 한도만 썼기 때문에 추가한
# 후보다(docs/experiments.md 2단계 "추가 후보 G2"). 같은 대본을 G1로 돌리면 모두 0.0이어야 한다.
def test_g2_samples_the_retry_but_not_the_first_attempt(tiny_engine):
    script = ["확인해 보겠습니다.", ToolCall("toy_cancel_order", {"order_id": "O-1"}), "취소했습니다."]
    _, state, _ = turn(tiny_engine, script, guard="G2")
    temperatures = [request["temperature"] for request in turn.provider.requests]
    assert temperatures == [0.0, 0.7, 0.7] and state.stalls == 1
    _, _, _ = turn(tiny_engine, script, guard="G1")
    assert [request["temperature"] for request in turn.provider.requests] == [0.0, 0.0, 0.0]


# analyze.compare: 인프라 오류로 시도 하나를 잃어 판정된 시도가 3개뿐인 과제가 있으면, pass^1은 두 과제를 모두
# 쓰고 pass^4는 양쪽 모두 4회가 판정된 과제만으로 짝지어 칸에 '(1과제)'라고 밝히는지 본다. infra_error는
# 분모에서 빼는 규칙(docs/design.md) 때문에 생기는 과제 수 차이를 표가 숨기지 않게 한다.
def test_compare_pairs_pass_k_over_the_tasks_that_have_k_trials_in_both_runs(tmp_path):
    import json

    # 과제별 성공 여부 목록으로 episodes.jsonl만 있는 가짜 실행 폴더를 만든다. 판정에 필요한 최소 필드(상태,
    # 종료 이유, db_match, 빈 values)만 넣는다.
    def write(name, trials_by_task):
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
        (run_dir / "episodes.jsonl").write_text(chr(10).join(lines), encoding="utf-8")
        return run_dir

    base = write("base", {"a": [True] * 4, "b": [False] * 4})
    other = write("other", {"a": [True] * 4, "b": [True, True, True]})  # b의 시도 하나는 infra 오류였다
    table = analyze.compare(base, [other])
    assert "| other | 100.0% | +50.0%p" in table  # pass^1은 두 과제를 모두 쓴다
    assert "100.0% (1과제)" in table  # pass^4는 두 실행 모두에서 판정된 시도가 넷인 과제만 쓴다


# 아래는 언어 가드(L1) 테스트다. qwen2.5가 한국어 대화 중 중국어로 넘어가는 일이 있어, 서비스는 한자·가나가 두
# 글자 이상인 답을 돌려보내고 한국어로 다시 답하게 한다. 측정하지 않은 가드이고 평가의 기본값은 끈
# 상태(L0)다(docs/service.md "정한 것과 이유").
# -------------------------------------------------------------------- 언어 가드 (서비스 전용)

# 실제로 나올 법한 중국어 답의 예. 고객 id와 주문 번호가 섞여 있어도 한자가 여러 글자다.
CHINESE = "顾客ID为C-9003，您的订单O-90003的状态是已发货。"


# L1: 중국어 답은 전달하지 않고 LANGUAGE_NOTICE를 하니스 안내문으로 넣어 다시 시키며, 형식 오류 1로 세고
# 기록에 'wrong_language'로 남는지 본다. 다시 받은 한국어 답이 고객에게 간다.
def test_a_reply_in_chinese_is_held_back_and_asked_again_in_korean(tiny_engine):
    result, state, _ = turn(tiny_engine, [CHINESE, "주문 O-1은 결제 완료 상태입니다."], language="L1")
    assert result.reply == "주문 O-1은 결제 완료 상태입니다." and state.format_errors == 1
    held, notice = state.messages[3], state.messages[4]
    assert (held.content, held.delivered, notice.harness, notice.content) == (
        CHINESE,
        False,
        True,
        LANGUAGE_NOTICE,
    )
    assert [log.format_error for log in state.llm_log] == ["wrong_language", None]


# L1에서 재시도를 다 써도 중국어면 답 없이 agent_format_error로 끝나고, 중국어 답은 한 번도 전달되지 않는지
# 본다. 고객이 읽을 수 없는 답이 새지 않게 하는 것이 이 가드의 목적이다.
def test_when_it_keeps_answering_in_chinese_nothing_is_delivered(tiny_engine):
    result, state, _ = turn(tiny_engine, [CHINESE] * 3, language="L1")
    assert (result.reply, result.stop) == (None, "agent_format_error")
    assert not [m for m in state.messages if m.role == "assistant" and m.delivered and m.content == CHINESE]


# 평가 기본값(RunConfig().language == 'L0')에서는 중국어 답도 그대로 전달되고 형식 오류로 세지 않는지 본다.
# 측정 기록과 비교 가능성을 지키려고 평가에서는 이 가드를 켜지 않는다.
def test_the_evaluation_default_delivers_the_reply_as_it_is(tiny_engine):
    result, state, _ = turn(tiny_engine, [CHINESE])
    assert result.reply == CHINESE and state.format_errors == 0 and RunConfig().language == "L0"


# 이름 옆 한자 한 글자(金)는 통과시키고, 일본어 가나 문장과 중국어 문장은 다른 언어로 보는지 본다. 한 글자까지
# 막으면 정상적인 한국어 답을 돌려보내게 된다.
def test_one_stray_character_is_not_another_language():
    assert not in_another_language("김하준(金) 고객님, 주문 O-1은 결제 완료 상태입니다.")
    assert in_another_language("ご注文は発送済みです") and in_another_language(CHINESE)
