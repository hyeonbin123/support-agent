"""9단계(음성 V4): 발신 번호가 고객 한 명을 고르고, 들은 이름은 그 고객의 이름이어야 한다.

처음 규칙은 자모 하나(k=1)를 허용했다가 이름 충돌 관문에서 멈췄고, 2026-10-04 규칙 변경으로 측정 전에
이름 완전 일치(k=0)를 등록했다(docs/experiments.md 9단계).

여기의 모든 것은 모델 없이 돈다: 작은 가게 위의 도구, seed 데이터 위의 대본 에피소드, 가짜를 끼운 실행기,
손으로 만든 기록 위의 분석 명령.
"""

# [파일 역할] 9단계(음성 V4: 발신 번호로 하는 본인 확인)를 모델 없이 검증하는 테스트 묶음이다. 구역은 여섯
# 개다: 이름 허용 오차(자모 거리), verify_caller 도구, 에이전트가 보는 프롬프트·도구 목록, 에피소드 안의 V4,
# 실행기, 분석 명령(본인 확인 집계, 짝짓기, 이름 충돌 관문, caller-id 표).
# [왜 중요한가] V4는 받아 적은 연락처 대신 '전화가 온 번호의 고객 한 명 + 이름'으로 본인 확인을 바꾸는 설계
# 변경이다(docs/experiments.md 9단계). 처음 규칙(자모 1개 허용, k=1)은 재기 전에 정한 충돌 관문에서 가상 고객
# 109명 안의 다른 이름 6쌍을 받아들여 멈췄고, 2026-10-04 이름을 정확히 맞추는 규칙(k=0)으로 다시 적었다. 이
# 테스트들은 그 관문 결과가 seed에서 다시 나오는지, V0~V2의 프롬프트·도구가 바이트까지 그대로인지, V4가
# V4에서만 쓰이는지를 지킨다.
# [방법] 작은 가게 DB 위의 도구 호출, seed DB 위의 대본 에피소드, 가짜 Ollama·가짜 음성 모델을 쓴 실행기,
# 손으로 만든 기록 위의 분석 명령.

from __future__ import annotations

# hashlib: 프롬프트·도구 스키마의 sha256을 기록된 manifest 해시와 비교한다.
import hashlib
import json

# 다른 테스트 파일의 도우미를 그대로 가져다 쓴다: 가짜 Ollama(test_run_voice), 모든 과제·고객 연락처·전달 값
# 문장(test_tasks), 가짜 합성기·인식기(test_voice). factories는 작은 가게 행을 더한다.
import pytest
from conftest import NOW
from factories import extend_tiny_rows
from sqlalchemy.orm import Session
from test_run_voice import FakeOllama
from test_tasks import ALL_TASKS, CONTACTS, say_values
from test_voice import FakeListener, FakeSpeaker

# tools의 NAME_TOLERANCE(등록한 허용 오차)와 name_distance(자모 거리), build_registry(caller_id=True)로 만드는
# V4 등록부가 검증 대상이다. metrics.identified는 본인 확인 성공을 세는 함수다.
from support_agent import analyze, db, run
from support_agent.agent import build_system_prompt, load_policy, visible_tools
from support_agent.chat import ScriptedProvider, ToolCall
from support_agent.config import RunConfig
from support_agent.episode import run_episode
from support_agent.seed import build_seed_engine
from support_agent.toolkit import ConversationState, ToolContext, execute
from support_agent.tools import NAME_TOLERANCE, build_registry, name_distance
from support_agent.user_sim import ScriptedUser
from support_agent.voice import metrics
from support_agent.voice.channel import SpeechChannel

# TEXT_REGISTRY: 글·V0~V2·서비스·MCP가 쓰는 기본 등록부(find_customer 포함). V4_REGISTRY: find_customer 대신
# verify_caller가 든 등록부. V4: 음성 V4 설정.
TEXT_REGISTRY = build_registry()
V4_REGISTRY = build_registry(caller_id=True)
V4 = RunConfig(voice="V4")
# 지금까지 개발용 기록(5단계 V1/V2, 8단계 P0·7B)의 manifest에 남은 에이전트 시스템 프롬프트와 도구 스키마의
# sha256. V0~V2는 이 값을 그대로 유지해야 한다. V4를 넣으면서 V0~V2의 프롬프트·도구가 한 바이트라도 바뀌면
# 이전 기록과 짝지을 수 없게 되므로 이 값으로 고정한다(docs/experiments.md 9단계 "V4가 바꾸는 것").
RECORDED_AGENT_SYSTEM = "7238ce2ba88a44d37e194701bff80b2224156510e84f61dee6fc2520968dafc7"
RECORDED_TOOLS = "565d3ecc74d8b6cf826ce2c9c74026f3c42542a23628722017367c360e2f937a"


# 글의 sha256 16진 문자열. manifest가 해시를 남기는 방식과 같다.
def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 모델에 보이는 도구 스키마 목록의 해시. 키를 정렬한 JSON으로 만들어 manifest의 tools 해시와 같은 방식으로
# 계산한다.
def tools_hash(registry, config) -> str:
    return sha(json.dumps(visible_tools(registry, config), ensure_ascii=False, sort_keys=True))


# 이름 허용 오차: 이름을 자모 단위로 비교한 편집 거리.
# ------------------------------------------------------------------------------- 이름 허용 오차


# 등록한 허용 오차가 0(이름 완전 일치)인지 본다. k=1(규칙 커밋 e7c28c5)은 충돌 관문을 통과하지 못했고
# 2026-10-04 규칙 변경으로 k=0이 등록됐다. 누군가 값을 바꾸면 측정 규칙과 코드가 어긋난다.
def test_the_registered_tolerance_is_exact_names():
    assert NAME_TOLERANCE == 0


# 개발용 V1·V2에서 인식기가 실제로 적은 이름(왼쪽)과 등록된 이름의 자모 거리가 기대값인지 본다. 공백과 끝의
# '님'·'고객님'은 find_customer처럼 먼저 떼고 비교하고, 영문 이름은 자모가 하나도 맞지 않아 거리가 크다. 이
# 거리 분포가 처음 k=1을 고른 근거였다(docs/experiments.md 9단계 "재기 전에 정한 것" 1).
@pytest.mark.parametrize(
    ("said", "registered", "distance"),
    [  # 왼쪽 이름은 개발용 V1/V2 실행에서 인식기가 적은 그대로다
        ("배성훈", "배성훈", 0),
        ("배성분", "배성훈", 1),  # ㅎ 대신 ㅂ
        ("최민서", "채민서", 1),  # ㅐ 대신 ㅚ
        ("허윤설", "허윤서", 1),  # 받침 ㄹ이 붙음
        ("하도병", "하도경", 1),
        ("안재기", "안재희", 2),  # ㅎㅢ 대신 ㄱㅣ
        ("송민리", "송민규", 2),
        ("빌태윤희", "길태윤", 3),
        ("배성훈 고객님", "배성훈", 0),  # find_customer처럼 공백과 존칭을 먼저 뗀다
        ("배 성훈님", "배성훈", 0),
        ("Bae Sunghoon", "배성훈", 11),  # "BaeSunghoon"과 자모 여덟 개를 비교
    ],
)
def test_names_are_compared_jamo_by_jamo(said, registered, distance):
    assert name_distance(said, registered) == distance


# verify_caller: 발신 번호가 가리키는 고객 한 명의 이름과만 비교한다(1:N 검색 없음).
# ------------------------------------------------------------------------------- verify_caller


# test_tools.py와 같은 작은 가게. C-1 김하준과 C-2 이서연의 전화번호가 다르다.
@pytest.fixture
def shop(tiny_engine):
    engine = db.memory_engine(tiny_engine)
    with Session(engine) as session:
        extend_tiny_rows(session)  # C-1 김하준 01000000001, C-2 이서연 01000000002
        session.commit()
    yield engine
    engine.dispose()


# V4 등록부로 verify_caller(name)를 부른다. 발신 번호는 ctx.caller_phone에 하니스가 넣는다.
def verify(engine, ctx: ToolContext, name: str):
    return execute(V4_REGISTRY, engine, ctx, "verify_caller", {"name": name})


# C-1의 번호로 건 전화에서 C-1의 이름을 말하면 find_customer와 같은 키(customer_id, name, grade,
# grade_label)로 통과하고, 대화 상태에 확인된 고객이 남아 다음 도구(get_order)가 열리는지 본다.
def test_the_caller_number_and_the_name_verify_the_customer(shop):
    ctx = ToolContext(now=NOW, caller_phone="01000000001")
    result = verify(shop, ctx, "김하준")
    assert result.ok, result.content
    assert json.loads(result.content) == {
        "customer_id": "C-1",
        "name": "김하준",
        "grade": "normal",
        "grade_label": "일반",
    }
    assert ctx.state.verified_customer_id == "C-1"
    order = execute(V4_REGISTRY, shop, ctx, "get_order", {"order_id": "O-1"})
    assert order.ok  # find_customer와 똑같이 본인 확인됨


# 자모 하나만 다른 이름(김하순)은 k=0에서 name_mismatch로 거절되고 확인 상태가 남지 않으며, 띄어쓰기·호칭만
# 다른 이름은 통과하는지 본다. 거절된 뒤에 다시 부를 수 있어야 에이전트가 이름을 다시 물어 확인을 되찾을 수
# 있다.
def test_one_jamo_off_is_refused_and_the_normalised_name_passes(shop):
    ctx = ToolContext(now=NOW, caller_phone="01000000001")
    result = verify(shop, ctx, "김하순")  # ㅈ 대신 ㅅ: k=1이면 통과, k=0이면 거절
    assert not result.ok and result.error_code == "name_mismatch"
    assert ctx.state.verified_customer_id is None
    # find_customer처럼 공백과 존칭을 먼저 뗀다
    for said in ("김하준 고객님", "김 하준님", "김하준"):
        result = verify(shop, ctx, said)
        assert result.ok and json.loads(result.content)["name"] == "김하준", said


# 그 밖의 모든 경우가 거절되고 아무도 확인되지 않는지 본다: 자모 1·2개 차이, 같은 번호에서 다른 고객의 이름,
# 다른 번호의 고객 이름, 빈 이름(name_mismatch), 가입하지 않은 번호(caller_not_registered), 발신 번호
# 없음(no_caller_number).
@pytest.mark.parametrize(
    ("caller", "name", "code"),
    [
        ("01000000001", "김하순", "name_mismatch"),  # 자모 하나 다름
        ("01000000001", "김하정", "name_mismatch"),  # 자모 둘 다름
        ("01000000001", "이서연", "name_mismatch"),  # 이 번호로 말한 다른 고객의 이름
        ("01000000002", "김하준", "name_mismatch"),  # 다른 번호 고객의 이름
        ("01000000001", "", "name_mismatch"),
        ("01099999999", "김하준", "caller_not_registered"),
        (None, "김하준", "no_caller_number"),
    ],
)
def test_anything_else_is_refused_and_verifies_nobody(shop, caller, name, code):
    ctx = ToolContext(now=NOW, caller_phone=caller)
    result = verify(shop, ctx, name)
    assert not result.ok and result.error_code == code, result.content
    assert ctx.state.verified_customer_id is None


# 번호 하나에 고객이 둘이면 이름이 맞아도 caller_shared로 거절하는지 본다. 번호가 고객 한 명을 가리킨다는 1:1
# 전제가 깨지면 확인하지 않는다(seed의 전화번호 109개는 모두 다르다).
def test_a_number_shared_by_two_customers_is_not_one_to_one(shop):
    with Session(shop) as session:
        session.get(db.Customer, "C-2").phone = "01000000001"
        session.commit()
    ctx = ToolContext(now=NOW, caller_phone="01000000001")
    result = verify(shop, ctx, "김하준")
    assert not result.ok and result.error_code == "caller_shared"


# 이미 다른 고객이 확인된 대화에서는 already_verified로 막고, 같은 고객을 두 번 확인하는 것은 허용하는지 본다.
# 대화당 고객 1명 규칙은 V4에서도 그대로다.
def test_one_customer_per_conversation_still_holds(shop):
    ctx = ToolContext(
        now=NOW, caller_phone="01000000001", state=ConversationState(verified_customer_id="C-2")
    )
    result = verify(shop, ctx, "김하준")
    assert not result.ok and result.error_code == "already_verified"
    again = ToolContext(now=NOW, caller_phone="01000000001")
    assert verify(shop, again, "김하준").ok and verify(shop, again, "김하준").ok  # 같은 고객 두 번은 괜찮다


# 에이전트가 보는 것: 도구 목록과 시스템 프롬프트. V4만 바뀌고 V0~V2는 그대로여야 한다.
# ------------------------------------------------------------------------------- 에이전트가 보는 것


# V4 등록부는 find_customer 하나를 verify_caller로 같은 자리(맨 앞)에 바꿔 넣고 나머지 도구는 순서까지 같은지,
# 기본 등록부(서비스, MCP, V0~V2)에는 verify_caller가 없는지 본다.
def test_v4_swaps_find_customer_for_verify_caller_and_nothing_else():
    assert "verify_caller" not in TEXT_REGISTRY and "find_customer" in TEXT_REGISTRY  # 서비스, MCP, V0~V2
    assert "verify_caller" in V4_REGISTRY and "find_customer" not in V4_REGISTRY
    others = [name for name in TEXT_REGISTRY if name != "find_customer"]
    assert [name for name in V4_REGISTRY if name != "verify_caller"] == others
    assert list(V4_REGISTRY)[0] == "verify_caller"  # find_customer가 있던 자리


# V4에서 find_customer를 부르면 unknown_tool인지 본다. 받아 적은 연락처로 찾는 경로를 남기면 V2의 실패 경로가
# 섞이고, 우연히 다른 고객과 맞으면 발신자가 아닌 고객이 확인되어 1:1이 깨지기 때문에 뺐다(docs/experiments.md
# 9단계 "재기 전에 정한 것" 2).
def test_find_customer_is_an_unknown_tool_under_v4(shop):
    ctx = ToolContext(now=NOW, caller_phone="01000000001")
    result = execute(V4_REGISTRY, shop, ctx, "find_customer", {"name": "김하준", "contact": "010-0000-0001"})
    assert not result.ok and result.error_code == "unknown_tool"


# V0·V1·V2 설정의 시스템 프롬프트와 도구 스키마 해시가 기록된 값과 같은지 본다. 이전 측정 기록(5단계 V1·V2,
# 8단계 P0·7B)과 짝지은 비교가 성립하려면 V4 코드를 넣어도 이 조건들은 한 바이트도 바뀌면 안 된다.
def test_v0_to_v2_prompts_and_tools_are_byte_for_byte_the_recorded_ones():
    dev_now = next(t for t in ALL_TASKS if t.id == "dev-001").now
    for voice in ("V0", "V1", "V2"):
        config = RunConfig(voice=voice)
        assert sha(build_system_prompt(load_policy(), dev_now, caller_id=config.caller_id)) == (
            RECORDED_AGENT_SYSTEM
        )
        assert tools_hash(TEXT_REGISTRY, config) == RECORDED_TOOLS
    assert sha(build_system_prompt(load_policy(), dev_now)) == RECORDED_AGENT_SYSTEM


# V4 시스템 프롬프트는 글 프롬프트에 규칙 한 줄('- '로 시작, verify_caller와 '발신 번호'를 말함)만 더한 것이고
# 그 줄을 빼면 글 프롬프트와 같은지, caller_id는 V4에서만 켜지는지, V4 도구 해시는 기록된 것과 다른지
# 본다(prompts/caller_id.md).
def test_the_v4_prompt_adds_one_rule_line_and_nothing_else():
    dev_now = next(t for t in ALL_TASKS if t.id == "dev-001").now
    text = build_system_prompt(load_policy(), dev_now)
    voice = build_system_prompt(load_policy(), dev_now, caller_id=True)
    added = [line for line in voice.splitlines() if line not in text.splitlines()]
    assert len(added) == 1 and added[0].startswith("- ") and "verify_caller" in added[0]
    assert "발신 번호" in added[0] and voice.replace(added[0] + "\n", "", 1) == text
    assert V4.caller_id and not RunConfig(voice="V2").caller_id
    assert tools_hash(V4_REGISTRY, V4) != RECORDED_TOOLS


# V4 프롬프트를 에이전트 프롬프트로 넣고 모든 과제를 V4 등록부로 검증해도 문제가 없는지 본다. 더한 한 줄이
# 어떤 과제의 전달 값을 미리 알려 주면 조회 없이 통과할 수 있기 때문이다.
@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda task: task.id)
def test_no_required_value_is_given_away_by_the_v4_prompt(task):
    from support_agent.judge import validate_task

    prompt = build_system_prompt(load_policy(), task.now, caller_id=True)
    assert validate_task(task, build_seed_engine(), V4_REGISTRY, prompt_texts=[prompt]) == []


# 에피소드 안에서: 하니스가 과제 고객의 가입 번호를 발신 번호로 넣는다.
# ------------------------------------------------------------------------------- 에피소드 안에서


# seed DB 위에서 대본 에이전트로 과제 하나를 V4(기본) 설정과 등록부로 돌린다. 고객은 한 마디 뒤 STOP한다.
def v4_episode(task, script, *, config=V4, registry=V4_REGISTRY):
    return run_episode(
        task,
        0,
        config=config,
        provider=ScriptedProvider(script),
        user=ScriptedUser(["문의드립니다."]),
        registry=registry,
        seed_engine=build_seed_engine(),
        policy_text=load_policy(),
    )


# 모든 과제에서 verify_caller(과제 고객의 이름) → 정답 쓰기 → 전달 값 말하기를 따르면 V4·P1에서 성공하는지
# 본다. 하니스가 넣는 발신 번호가 언제나 과제의 고객을 가리킨다는 것, 즉 모든 과제가 V4에서도 풀 수 있다는
# 확인이다.
@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda task: task.id)
def test_the_gold_path_with_verify_caller_passes_under_v4(task):
    name, _ = CONTACTS[task.customer_id]
    script = [ToolCall("verify_caller", {"name": name})]
    script += [ToolCall(a.tool, a.args) for a in task.gold_actions] + [say_values(task)]
    result = v4_episode(task, script, config=RunConfig(voice="V4", policy="P1"))
    assert result.verdict is not None and result.verdict.success, (result.termination, result.verdict)
    assert result.tool_calls[0].ok


# dev-002(배성훈)에서 인식기가 적은 '배성분'은 거절되고 다시 부른 '배성훈'은 과제 고객(C-9102)으로 확인되는지,
# 잘못 들은 이름 한 번으로 끝난 에피소드는 본인 확인 성공으로 세지 않는지 본다.
def test_the_caller_is_the_task_customer_and_a_misheard_name_is_refused():
    task = next(t for t in ALL_TASKS if t.id == "dev-002")  # 이메일 주소를 댄 배성훈
    script = [
        ToolCall("verify_caller", {"name": "배성분"}),  # V1/V2 실행에서 인식기가 적은 그대로
        ToolCall("verify_caller", {"name": "배성훈"}),
        "확인되었습니다.",
    ]
    misheard, heard = v4_episode(task, script).tool_calls
    assert not misheard.ok and misheard.error_code == "name_mismatch"
    assert heard.ok and json.loads(heard.content)["customer_id"] == task.customer_id == "C-9102"
    refused = v4_episode(task, [ToolCall("verify_caller", {"name": "배성분"}), "네."])
    assert not metrics.identified(json.loads(refused.to_json_line()))


# V4 설정에 글 등록부를 주거나, V2 설정에 V4 등록부를 주면 에피소드 실행이 ValueError로 거절하는지 본다.
# 설정과 등록부가 어긋난 실행이 조용히 기록되는 것을 막는다.
def test_v4_needs_its_registry_and_the_other_conditions_refuse_it():
    task = next(t for t in ALL_TASKS if t.id == "dev-001")
    with pytest.raises(ValueError, match="caller_id"):
        v4_episode(task, ["네."], registry=TEXT_REGISTRY)
    with pytest.raises(ValueError, match="caller_id"):
        v4_episode(task, ["네."], config=RunConfig(voice="V2"))


# V2 에피소드는 발신 번호 없이 find_customer로 확인되는지 본다. V4 하니스가 다른 조건에 새지 않는다.
def test_a_text_episode_has_no_caller_number():
    task = next(t for t in ALL_TASKS if t.id == "dev-001")
    name, contact = CONTACTS[task.customer_id]
    result = v4_episode(
        task,
        [ToolCall("find_customer", {"name": name, "contact": contact}), "네."],
        config=RunConfig(voice="V2"),
        registry=TEXT_REGISTRY,
    )
    assert result.tool_calls[0].ok


# 실행기: V4는 V2와 같은 채널을 쓰고 manifest에 허용 오차를 남긴다.
# ------------------------------------------------------------------------------- 실행기


# 가짜 Ollama와 가짜 채널로 V2와 V4 실행을 하나씩 돌려, V4 실행 이름에 V4가 붙고, manifest에 caller_id(도구
# 이름, 허용 오차 0)가 V4에만 남으며, 두 실행의 프롬프트 해시 가운데 agent_system과 tools만 다른지 본다. 이 두
# 해시만 다를 때 V2와 V4를 짝짓는다는 9단계 짝짓기 규칙의 전제다.
def test_a_v4_run_uses_the_v2_normaliser_the_caller_registry_and_records_k(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OllamaProvider", FakeOllama)
    monkeypatch.setattr(run, "gpu_used_by_others_mib", lambda provider: None)
    monkeypatch.setattr(run, "OUTPUTS", tmp_path)
    built = []

    # 실행기의 채널 생성을 바꿔치기해, 어떤 음성 조건으로 불렸는지 기록하고 가짜 음성 모델 채널을 준다.
    def build_channel(config, tts_device, stt_device):
        built.append(config.voice)
        return SpeechChannel(FakeSpeaker(), FakeListener(), cache_path=tmp_path / "cache.jsonl")

    monkeypatch.setattr(run, "build_channel", build_channel)
    for voice in ("V2", "V4"):
        argv = ["run", "--tasks", "smoke", "--task-id", "smoke-lookup-01", "--voice", voice]
        monkeypatch.setattr("sys.argv", [*argv, "--label", voice.lower()])
        run.main()
    assert built == ["V2", "V4"]
    v2, v4 = sorted((tmp_path / "runs").iterdir(), key=lambda p: p.name[-2:])
    assert v4.name.endswith("-P0-R0-G0-F0-V4-v4")
    m2, m4 = (json.loads((d / "manifest.json").read_text(encoding="utf-8")) for d in (v2, v4))
    assert m4["config"]["voice"] == "V4" and m4["caller_id"] == {"tool": "verify_caller", "name_tolerance": 0}
    assert m2["caller_id"] is None
    assert m2["prompt_sha256"]["agent_system"] == sha(
        build_system_prompt(load_policy(), next(t for t in ALL_TASKS if t.id == "smoke-lookup-01").now)
    )
    differs = {k for k in m2["prompt_sha256"] if m2["prompt_sha256"][k] != m4["prompt_sha256"][k]}
    assert differs == {"agent_system", "tools"}


# 실제 build_channel이 V2와 V4 모두에 V2 정규화(normalize_heard)를 붙이는지 본다. MeloTTS·Whisper 클래스는
# 가짜로 바꿔 모델을 읽지 않는다. 고객의 말이 지나는 채널은 V2와 같아야 차이가 본인 확인 방식에서만 나온다.
def test_the_v4_channel_is_the_v2_channel(tmp_path, monkeypatch):
    import support_agent.voice.speech as speech

    class Speaker(FakeSpeaker):
        def __init__(self, device="cuda"):
            super().__init__()

    class Listener(FakeListener):
        def __init__(self, device="cuda"):
            pass

    monkeypatch.setattr(speech, "MeloSpeaker", Speaker)
    monkeypatch.setattr(speech, "WhisperListener", Listener)
    monkeypatch.setattr(run, "OUTPUTS", tmp_path)
    seen = {
        v: run.build_channel(RunConfig(voice=v), "cpu", "cpu").describe()["normalizer"] for v in ("V2", "V4")
    }
    assert seen == {"V2": "normalize_heard", "V4": "normalize_heard"}


# 분석 명령: 본인 확인 집계, 짝짓기 검사, 이름 충돌 관문, caller-id 표.
# ------------------------------------------------------------------------------- analysis


# 1차 지표(본인 확인 성공률)가 verify_caller 성공도 find_customer 성공처럼 세는지 본다.
def test_identification_counts_verify_caller_too():
    assert metrics.identified({"tool_calls": [{"name": "verify_caller", "ok": True}]})
    assert not metrics.identified({"tool_calls": [{"name": "verify_caller", "ok": False}]})
    assert metrics.identified({"tool_calls": [{"name": "find_customer", "ok": True}]})


# 짝짓기 검사에 필요한 최소 manifest와 에피소드 하나만 든 실행 폴더를 만든다.
def write_manifest(root, name, prompts, voice):
    run_dir = root / name
    run_dir.mkdir()
    manifest = {
        "config": {"voice": voice},
        "trials": 1,
        "task_sha256": {"dev-001": "a"},
        "seed_hash": "s",
        "prompt_sha256": prompts,
        "agent_provider": {"model": "m"},
        "user_provider": {"model": "m"},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    line = {"task_id": "dev-001", "trial": 0, "status": "completed"}
    (run_dir / "episodes.jsonl").write_text(json.dumps(line), encoding="utf-8")
    return run_dir


# same-setup이 프롬프트 해시 차이를 이름과 함께 알리고, --ignore-prompt로 지정한 해시(tools, agent_system)만
# 넘기며 다른 해시(policy) 차이는 여전히 잡는지, CLI가 'same setup'을 출력하는지 본다.
def test_same_setup_can_let_named_prompt_hashes_differ(tmp_path, monkeypatch, capsys):
    base = {"policy": "p", "agent_system": "a", "user_sim": "u", "tools": "t"}
    v2 = write_manifest(tmp_path, "v2", base, "V2")
    v4 = write_manifest(tmp_path, "v4", {**base, "agent_system": "a4", "tools": "t4"}, "V4")
    other = write_manifest(tmp_path, "v4p", {**base, "agent_system": "a4", "policy": "p4"}, "V4")
    assert analyze.setup_differences(v2, v4) == ["prompt_sha256 differs (agent_system, tools)"]
    assert analyze.setup_differences(v2, v4, ignore_prompts=("tools", "agent_system")) == []
    assert analyze.setup_differences(v2, other, ignore_prompts=("tools", "agent_system")) == [
        "prompt_sha256 differs (policy)"
    ]
    argv = ["analyze", "same-setup", str(v2), str(v4), "--ignore", "voice"]
    monkeypatch.setattr("sys.argv", [*argv, "--ignore-prompt", "tools", "agent_system"])
    analyze.main()
    assert "same setup" in capsys.readouterr().out


# 충돌 검사가 k 안의 다른 이름을 순서쌍으로 세고(양쪽 방향), 글자까지 같은 동명이인은 따로 세는지 본다.
# 동명이인은 k=0도 받아들이고 확인되는 고객이 번호의 주인이라 허용 오차 탓이 아니기 때문이다. k를 2로 넓히면
# 거리 2인 쌍(김하준·김하정)이 새로 잡힌다.
def test_the_collision_check_counts_other_names_within_k_and_sets_same_names_apart():
    names = {"C-1": "김하준", "C-2": "김하순", "C-3": "박서연", "C-4": "김하준", "C-5": "김하정"}
    collisions, same = analyze.name_collisions(names, k=1)
    # 김하정은 김하준과 자모 둘, 김하순과 자모 셋 차이라 k=1에서 충돌이 아니다
    assert [(a, b) for a, b, *_ in collisions] == [
        ("C-1", "C-2"),
        ("C-2", "C-1"),
        ("C-2", "C-4"),
        ("C-4", "C-2"),
    ]
    assert same == [("C-1", "C-4"), ("C-4", "C-1")]
    assert analyze.name_collisions({"C-1": "김하준", "C-3": "박서연"}, k=1) == ([], [])
    collisions_k2, _ = analyze.name_collisions(names, k=2)
    assert ("C-1", "C-5") in [(a, b) for a, b, *_ in collisions_k2]


# caller-id 표용 최소 에피소드 기록.
def episode_with(calls, task_id="dev-001"):
    return {"task_id": task_id, "status": "completed", "tool_calls": calls, "voice": []}


# 도구 호출 기록 하나(이름, 성공 여부, 인자, 출력, 오류 코드).
def call(name, ok, args=None, content="", code=None):
    return {"name": name, "ok": ok, "args": args, "content": content, "error_code": code}


# caller-id 표의 열을 손으로 만든 기록으로 확인한다: 에피소드 수(infra_error는 뺌), 본인 확인 성공, 확인
# 도구를 부르지 않음, 도구별 성공/호출, 말한 이름과 등록 이름이 달랐는데 통과한 확인(허용 오차로 통과), 없는
# 도구 호출, 발신 번호 오류. k=0에서 '허용 오차로 통과'가 0이 아니면 버그이므로 그 경우를 일부러 넣어 따로
# 세는지 본다.
def test_the_caller_id_table_tells_tolerance_passes_apart(tmp_path):
    # 손으로 만든 기록: k=0에서 "허용 오차" 칸은 0이어야 하고, 거기 세어진 통과는 버그다
    found = json.dumps({"customer_id": "C-9101", "name": "고은채"}, ensure_ascii=False)
    v4 = [
        episode_with([call("verify_caller", True, {"name": "고은채"}, found)]),
        episode_with(
            [
                call("verify_caller", False, None, code="name_mismatch"),
                call("verify_caller", True, {"name": "고은재"}, found),
            ]
        ),
        episode_with([call("find_customer", False, None, code="unknown_tool")]),
        episode_with([]),
        {**episode_with([call("verify_caller", False, code="no_caller_number")]), "status": "truncated"},
        {**episode_with([]), "status": "infra_error"},  # 빠진다
    ]
    v2 = [episode_with([call("find_customer", True, {"name": "고은채", "contact": "x"}, found)])] * 4
    table = analyze.caller_id_table({"v2": v2, "v4": v4})
    assert "| v2 | 4 | 4 (100.0%) | 0 | 4 / 4 | 0 / 0 | 0 | 0 | 0 |" in table
    assert "| v4 | 5 | 2 (40.0%) | 1 | 0 / 1 | 2 / 4 | 1 | 1 | 1 |" in table


# seed 고객 109명에 k=1 관문을 다시 돌리면 기록된 결과(동명이인 순서쌍 6, 충돌 순서쌍 12 = 이름 쌍 6, 쌍의
# 고객 번호까지)가 그대로 나오고 FAIL인지, k=0에서는 0건인지 본다. docs/experiments.md 9단계 "관문 결과" 표가
# 코드와 seed에서 다시 계산된다는 확인이다.
def test_the_recorded_gate_result_comes_back_from_the_seed():
    # docs/experiments.md 9단계: 첫 관문은 k=1에서 순서쌍 12개로 실패했고 단계가 멈췄다
    names = analyze.seed_names()
    collisions, same = analyze.name_collisions(names, k=1)
    assert len(names) == 109 and len(same) == 6 and len(collisions) == 12
    assert {frozenset((a, b)) for a, b, *_ in collisions} == {
        frozenset(pair)
        for pair in [
            ("C-0006", "C-0016"),
            ("C-0024", "C-0028"),
            ("C-0028", "C-0030"),
            ("C-9101", "C-9211"),
            ("C-9110", "C-9203"),
            ("C-9120", "C-9208"),
        ]
    }
    assert analyze.name_collisions(names, k=0)[0] == []
    report, passed = analyze.collision_report(1)
    assert not passed and report.endswith("gate: FAIL")


# 다시 등록한 k=0 관문이 PASS(0건)이고 동명이인 6쌍은 따로 적히는지, 그리고 analyze name-collisions의 기본 k가
# 등록한 값(0)이라 인자 없이 돌려도 같은 결과인지 본다. 관문이 실패하면 명령이 0이 아닌 값으로 끝난다.
def test_the_re_registered_gate_passes_at_exact_names(monkeypatch, capsys):
    # docs/experiments.md 9단계 2026-10-04 규칙 변경: k=0, 재기 전에 관문이 0건이어야 한다
    report, passed = analyze.collision_report(0)
    assert passed and report.endswith("gate: PASS (0 collisions)")
    assert "k = 0" in report and "same-name ordered pairs (set apart): 6 " in report
    monkeypatch.setattr("sys.argv", ["analyze", "name-collisions"])  # 기본 k가 등록한 값이다
    analyze.main()  # 관문이 실패하면 0이 아닌 값으로 끝난다
    out = capsys.readouterr().out
    assert "k = 0" in out and "gate: PASS (0 collisions)" in out
