"""컨텍스트 넘침을 잘라 답하지 않고 오류로 다루는지(Ollama truncate: false), 그리고 manifest의 Ollama 주소에서
인증 정보를 가리는지 시험한다. 실제 서버와 GPU 없이 httpx.MockTransport와 대본 제공자로 돈다."""

# [테스트 묶음] 2026-10-15에 넣은 두 가지를 고정한다.
# (1) ollama.py가 모든 /api/chat 요청에 truncate: false를 보내고, 서버가 넘침을 HTTP 400
#     (exceed_context_size_error)으로 거절하면 ContextOverflowError로 올린다. 에이전트 루프는 그것을
#     context_limit 종료로 바꾸고(평가에서는 실패, 서비스에서는 세션을 닫으며 새 상담을 권함), 시뮬레이터
#     쪽에서 넘치면 지금처럼 infra_error가 된다. 오류 본문 꼴은 같은 Ollama 0.35.1에서 다른
#     프로젝트(rag-doc-qa v11 스모크)가 받은 본문을 그대로 썼다.
# (2) manifest의 base_url에서 사용자 정보(아이디·비밀번호)를 가린다. 공식 실행의 manifest는 공개 저장소에
#     커밋되기 때문이다. 실제 요청은 가리지 않은 주소로 간다. 그리고 옛 manifest에 없는 truncate 필드가
#     짝짓기(analyze same-setup)를 막지 않는다.

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from test_service import engine, seeded_template  # noqa: F401 - fixture

from support_agent import analyze
from support_agent.agent import agent_turn, new_state
from support_agent.chat import (
    ChatProvider,
    ChatResponse,
    ContextOverflowError,
    Message,
    ProviderError,
    ScriptedProvider,
)
from support_agent.config import RunConfig
from support_agent.episode import run_episode
from support_agent.ollama import OllamaProvider, redact_url
from support_agent.seed import build_seed_engine
from support_agent.service.core import CLOSED_REPLY, ChatService
from support_agent.service.settings import Settings
from support_agent.tasks import load_tasks
from support_agent.tools import build_registry
from support_agent.user_sim import LLMUser, ScriptedUser

# 같은 PC의 Ollama 0.35.1이 truncate: false인 요청(1,634토큰, num_ctx 512)에 돌려준 본문 그대로다.
OVERFLOW_BODY = {
    "error": {
        "code": 400,
        "message": "request (1634 tokens) exceeds the available context size (512 tokens), try increasing it",
        "type": "exceed_context_size_error",
        "n_prompt_tokens": 1634,
        "n_ctx": 512,
    }
}


# 경로 하나(/api/chat)에 정해진 응답을 돌려주고 받은 요청을 모으는 가짜 Ollama 서버.
def ollama_with(answer: httpx.Response) -> tuple[OllamaProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answer

    return OllamaProvider("qwen3.5:4b", transport=httpx.MockTransport(handler)), seen


# 넘침 오류는 ProviderError의 하위 클래스인 ContextOverflowError로 오고, 서버가 알려 준 프롬프트 토큰 수와
# num_ctx를 담는다. 요청 본문에는 truncate: false가 들어 있다.
def test_an_overflowing_prompt_is_refused_not_truncated():
    provider, seen = ollama_with(httpx.Response(400, json=OVERFLOW_BODY))
    with pytest.raises(ContextOverflowError) as caught:
        provider.chat([Message("user", "긴 대화")])
    assert isinstance(caught.value, ProviderError)
    assert (caught.value.prompt_tokens, caught.value.num_ctx) == (1634, 512)
    assert json.loads(seen[0].content)["truncate"] is False


# error가 문자열로만 와도(본문 꼴이 다른 버전) 문구로 알아보고 토큰 수를 읽는다.
def test_the_overflow_message_alone_is_recognised():
    text = "request (20000 tokens) exceeds the available context size (16384 tokens), try increasing it"
    provider, _ = ollama_with(httpx.Response(400, json={"error": text}))
    with pytest.raises(ContextOverflowError) as caught:
        provider.chat([Message("user", "긴 대화")])
    assert (caught.value.prompt_tokens, caught.value.num_ctx) == (20000, 16384)


# 넘침이 아닌 400(예: 사고 모드가 없는 모델에 think를 보냄)과 500은 지금처럼 일반 ProviderError다.
@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(400, json={"error": '"qwen2.5:7b-instruct" does not support thinking'}),
        httpx.Response(500, text="boom"),
    ],
)
def test_other_errors_stay_plain_provider_errors(answer):
    provider, _ = ollama_with(answer)
    with pytest.raises(ProviderError) as caught:
        provider.chat([Message("user", "hi")])
    assert not isinstance(caught.value, ContextOverflowError)


# 정해진 순번의 호출에서만 넘침 오류를 내고, 나머지는 대본대로 답하는 제공자.
class OverflowAt(ChatProvider):
    def __init__(self, script: list, overflow_at: int):
        self._script = ScriptedProvider(script)
        self._at = overflow_at
        self.calls = 0

    def chat(self, messages, tools=(), *, temperature=0.0, seed=None, max_tokens=1024):
        self.calls += 1
        if self.calls == self._at:
            raise ContextOverflowError("ollama: context overflow", prompt_tokens=16500, num_ctx=16384)
        return self._script.chat(messages, tools, temperature=temperature, seed=seed, max_tokens=max_tokens)


# 에이전트 루프: 넘친 요청은 context_limit 종료가 된다. 형식 오류로 세지 않고(format_error None), 호출 기록의
# finish_reason에 context_overflow와 서버가 알려 준 프롬프트 토큰 수를 남긴다.
def test_the_agent_loop_ends_an_overflowing_turn_at_the_context_limit():
    state = new_state("system prompt")
    result = agent_turn(
        state,
        "안녕하세요",
        provider=OverflowAt([], overflow_at=1),
        registry=build_registry(),
        run_tool=lambda name, arguments: pytest.fail("no tool should run"),
        config=RunConfig(),
    )
    assert (result.reply, result.stop) == (None, "context_limit")
    (call,) = state.llm_log
    assert (call.format_error, call.finish_reason, call.prompt_tokens) == (None, "context_overflow", 16500)
    assert state.agent_calls == 1


# 평가 에피소드: 에이전트 쪽 넘침은 판정되는 실패(truncated, context_limit)이고 infra_error가 아니다.
def test_an_episode_whose_agent_overflows_is_a_failure_not_an_infra_error():
    task = load_tasks("smoke")[0]
    seed_engine = build_seed_engine()
    result = run_episode(
        task,
        0,
        config=RunConfig(),
        provider=OverflowAt(["어떤 주문을 도와드릴까요?"], overflow_at=2),
        user=ScriptedUser(["주문 문의요", "주문 번호는 모르겠어요"]),
        registry=build_registry(),
        seed_engine=seed_engine,
        policy_text="규정",
    )
    assert (result.status, result.termination) == ("truncated", "context_limit")
    assert result.verdict is not None and not result.verdict.success


# 시뮬레이터 쪽 넘침은 에이전트의 실패가 아니므로 지금처럼 infra_error로 기록하고, 오류 글에 넘침이 보인다.
def test_an_episode_whose_simulator_overflows_is_an_infra_error():
    task = load_tasks("smoke")[0]
    config = RunConfig()
    result = run_episode(
        task,
        0,
        config=config,
        provider=ScriptedProvider([]),
        user=LLMUser(OverflowAt([], overflow_at=1), task, config, 0),
        registry=build_registry(),
        seed_engine=build_seed_engine(),
        policy_text="규정",
    )
    assert result.status == "infra_error"
    assert "ContextOverflowError" in result.error and "context overflow" in result.error


# 서비스: 넘친 턴은 세션을 닫고 새 상담을 권하는 고정 안내(CLOSED_REPLY)로 끝난다. 이 턴의 LLM 호출 감사 행에
# context_overflow가 남는다.
def test_the_service_closes_an_overflowing_session_with_the_closing_reply(engine):  # noqa: F811
    service = ChatService(Settings(), engine, OverflowAt([], overflow_at=1))
    session_id = service.new_session()["session_id"]
    events: list[tuple[str, dict]] = []
    service.handle(session_id, "안녕하세요", lambda name, data: events.append((name, data)))

    assert ("reply", {"text": CLOSED_REPLY}) in events
    assert events[-1] == ("end", {"status": "closed"})
    audit = service.audit(session_id)
    llm = [e["payload"] for e in audit if e["kind"] == "llm_call"]
    assert [c["finish_reason"] for c in llm] == ["context_overflow"]
    assert not [e for e in audit if e["kind"] == "error"]


# 주소의 사용자 정보만 가리고 나머지(스킴, 호스트, 포트, 경로)는 그대로 둔다. 사용자 정보가 없으면 글자까지
# 같다(지난 기록의 manifest와 짝짓기가 바뀌지 않는다).
@pytest.mark.parametrize(
    ("url", "shown"),
    [
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("http://user:secret@gpu-box:11434", "http://***@gpu-box:11434"),
        ("https://token@ollama.example/base", "https://***@ollama.example/base"),
        ("http://user:pw@[::1]:11434", "http://***@[::1]:11434"),
    ],
)
def test_credentials_are_hidden_in_the_url(url, shown):
    assert redact_url(url) == shown


# manifest에 가는 describe()의 base_url은 가린 주소지만, 요청은 인증 정보를 담은 원래 주소로 간다.
def test_the_manifest_hides_credentials_but_requests_still_use_them():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "네"}, "done": True})
        return httpx.Response(200, json={"models": [], "version": "0.35.1"})

    provider = OllamaProvider(
        "qwen3.5:4b", base_url="http://user:secret@gpu-box:11434", transport=httpx.MockTransport(handler)
    )
    description = provider.describe()
    assert description["base_url"] == "http://***@gpu-box:11434"
    assert "secret" not in json.dumps(description)
    provider.chat([Message("user", "hi")])
    chat = next(r for r in seen if r.url.path == "/api/chat")
    assert chat.headers["authorization"].startswith("Basic ")  # httpx가 주소의 인증 정보를 헤더로 보낸다


# truncate 필드가 생기기 전의 manifest(필드 없음)는 지금 manifest(False)와 같은 설정으로 짝지어진다. 값이
# 다르면(True) 차이로 나온다.
def test_old_manifests_without_truncate_still_pair(tmp_path: Path):
    def run(name: str, provider_extra: dict) -> Path:
        folder = tmp_path / name
        folder.mkdir()
        provider = {"provider": "ollama", "model": "qwen3.5:4b", "num_ctx": 12288, **provider_extra}
        manifest = {
            "config": RunConfig().to_dict(),
            "trials": 1,
            "task_sha256": {"dev-001": "x"},
            "seed_hash": "s",
            "prompt_sha256": {"policy": "p"},
            "agent_provider": provider,
            "user_provider": provider,
        }
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (folder / "episodes.jsonl").write_text(
            json.dumps({"status": "completed", "task_id": "dev-001", "trial": 0}) + "\n", encoding="utf-8"
        )
        return folder

    old, new, other = run("old", {}), run("new", {"truncate": False}), run("other", {"truncate": True})
    assert analyze.setup_differences(old, new) == []
    assert analyze.setup_differences(new, old) == []
    assert analyze.setup_differences(new, other) == [
        "agent_provider differs (truncate)",
        "user_provider differs (truncate)",
    ]


# 대본 제공자가 넘침 오류 없이도 같은 응답을 주는지 확인하는 기본 경우: ChatResponse가 그대로 지나간다.
def test_overflow_provider_passes_normal_replies_through():
    provider = OverflowAt([ChatResponse(text="네")], overflow_at=99)
    assert provider.chat([Message("user", "hi")]).text == "네"
