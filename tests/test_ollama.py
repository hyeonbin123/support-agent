"""OllamaProvider against httpx.MockTransport. No real server, no GPU."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from support_agent.chat import Message, ProviderError, ToolCall
from support_agent.ollama import DEFAULT_BASE_URL, OllamaProvider

MODEL = "qwen2.5:7b-instruct"
TOOL = {
    "name": "get_order_detail",
    "description": "주문 번호로 주문 상세를 조회한다",
    "parameters": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
}


@pytest.fixture(autouse=True)
def _no_env_base_url(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)


def reply(message: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    body = {
        "model": MODEL,
        "message": {"role": "assistant", "content": "네"} if message is None else message,
        "done": True,
        "done_reason": "stop",
    }
    body.update(extra)
    return body


class Server:
    """Records requests and answers from a route table: path -> dict | httpx.Response | Exception."""

    def __init__(self, routes: dict[str, Any] | None = None):
        self.routes: dict[str, Any] = {"/api/chat": reply()}
        self.routes.update(routes or {})
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.routes.get(request.url.path)
        if answer is None:
            return httpx.Response(404, text="not found")
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=answer)

    def bodies(self, path: str = "/api/chat") -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]

    def count(self, path: str) -> int:
        return sum(1 for r in self.requests if r.url.path == path)


def make(server: Server, **kwargs: Any) -> OllamaProvider:
    return OllamaProvider(MODEL, transport=httpx.MockTransport(server), **kwargs)


# --- request body ---


def test_request_body_basics():
    server = Server()
    provider = make(server)
    provider.chat([Message("system", "규정"), Message("user", "안녕하세요")], temperature=0.7, seed=42)

    request = server.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{DEFAULT_BASE_URL}/api/chat"
    body = server.bodies()[0]
    assert body["model"] == MODEL
    assert body["stream"] is False
    assert body["keep_alive"] == "60m"
    assert body["options"] == {"num_ctx": 16384, "temperature": 0.7, "seed": 42, "num_predict": 1024}
    assert body["messages"] == [
        {"role": "system", "content": "규정"},
        {"role": "user", "content": "안녕하세요"},
    ]
    assert "tools" not in body
    assert "think" not in body


def test_seed_omitted_when_none_and_instance_options_used():
    server = Server()
    provider = make(server, num_ctx=8192, keep_alive="10m")
    provider.chat([Message("user", "hi")], max_tokens=256)

    body = server.bodies()[0]
    assert body["keep_alive"] == "10m"
    assert body["options"] == {"num_ctx": 8192, "temperature": 0.0, "num_predict": 256}


@pytest.mark.parametrize("think", [True, False])
def test_think_sent_at_top_level_when_set(think):
    server = Server()
    make(server, think=think).chat([Message("user", "hi")])

    body = server.bodies()[0]
    assert body["think"] is think
    assert "think" not in body["options"]


def test_tools_are_wrapped():
    server = Server()
    make(server).chat([Message("user", "hi")], tools=[TOOL])

    assert server.bodies()[0]["tools"] == [{"type": "function", "function": TOOL}]


def test_history_conversion():
    server = Server()
    history = [
        Message("user", "주문 조회", harness=True),
        Message(
            "assistant",
            "",
            tool_calls=(ToolCall("get_order_detail", {"order_id": "O-1001"}, id="call_1"),),
            delivered=False,
        ),
        Message("tool", '{"status": "결제완료"}', tool_name="get_order_detail", tool_call_id="call_1"),
        Message("assistant", "결제완료 상태입니다."),
    ]
    make(server).chat(history)

    assert server.bodies()[0]["messages"] == [
        {"role": "user", "content": "주문 조회"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_order_detail", "arguments": {"order_id": "O-1001"}}}],
        },
        {"role": "tool", "tool_name": "get_order_detail", "content": '{"status": "결제완료"}'},
        {"role": "assistant", "content": "결제완료 상태입니다."},
    ]


def test_request_body_is_utf8_json():
    server = Server()
    make(server).chat([Message("user", "환불해 주세요")])

    assert json.loads(server.requests[0].content.decode("utf-8"))["messages"][0]["content"] == "환불해 주세요"


# --- response parsing ---


def test_text_reply():
    response = make(Server()).chat([Message("user", "hi")])

    assert response.text == "네"
    assert response.tool_calls == ()
    assert response.thinking == ""
    assert response.finish_reason == "stop"


def test_tool_call_with_dict_arguments_and_id():
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_ab12cd34",
                "function": {"index": 0, "name": "get_order_detail", "arguments": {"order_id": "O-1"}},
            },
            {"function": {"index": 1, "name": "get_user", "arguments": {"user_id": "U-1"}}},
        ],
    }
    response = make(Server({"/api/chat": reply(message)})).chat([Message("user", "hi")], tools=[TOOL])

    assert response.text == ""
    assert response.tool_calls == (
        ToolCall("get_order_detail", {"order_id": "O-1"}, id="call_ab12cd34"),
        ToolCall("get_user", {"user_id": "U-1"}, id=""),
    )


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ('{"order_id": "O-1"}', {"order_id": "O-1"}),
        ("{not json", {"_raw": "{not json"}),
        ("[1, 2]", {"_raw": "[1, 2]"}),
        ([1, 2], {"_raw": [1, 2]}),
        (7, {"_raw": 7}),
        (None, {}),
    ],
)
def test_tool_call_argument_shapes(arguments, expected):
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "t", "arguments": arguments}}],
    }
    response = make(Server({"/api/chat": reply(message)})).chat([Message("user", "hi")])

    assert response.tool_calls == (ToolCall("t", expected),)


def test_tool_call_without_arguments_key():
    message = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t"}}]}
    response = make(Server({"/api/chat": reply(message)})).chat([Message("user", "hi")])

    assert response.tool_calls == (ToolCall("t", {}),)


@pytest.mark.parametrize("tool_calls", ["oops", ["oops"], [{"id": "x"}], [{"function": "t"}]])
def test_malformed_tool_calls_raise(tool_calls):
    message = {"role": "assistant", "content": "", "tool_calls": tool_calls}
    with pytest.raises(ProviderError, match="tool"):
        make(Server({"/api/chat": reply(message)})).chat([Message("user", "hi")])


def test_thinking_and_usage_mapping():
    body = reply(
        {"role": "assistant", "content": "답", "thinking": "생각 중"},
        done_reason="length",
        prompt_eval_count=912,
        eval_count=28,
        total_duration=1_834_000_000,
        load_duration=21_000_000,
        prompt_eval_duration=410_000_000,
        eval_duration=1_390_500_000,
    )
    response = make(Server({"/api/chat": body})).chat([Message("user", "hi")])

    assert response.text == "답"
    assert response.thinking == "생각 중"
    assert response.finish_reason == "length"
    usage = response.usage
    assert usage.prompt_tokens == 912
    assert usage.completion_tokens == 28
    assert usage.load_ms == pytest.approx(21.0)
    assert usage.prompt_eval_ms == pytest.approx(410.0)
    assert usage.eval_ms == pytest.approx(1390.5)
    assert usage.wall_ms > 0.0


def test_missing_usage_fields_default_to_zero():
    body = {"message": {"role": "assistant", "content": None}, "prompt_eval_count": None}
    response = make(Server({"/api/chat": body})).chat([Message("user", "hi")])

    assert response.text == ""
    assert response.finish_reason is None
    usage = response.usage
    assert (usage.prompt_tokens, usage.completion_tokens) == (0, 0)
    assert (usage.load_ms, usage.prompt_eval_ms, usage.eval_ms) == (0.0, 0.0, 0.0)


def test_lone_surrogates_are_cleaned():
    # An escaped "\ud83d" in the JSON body decodes to a lone surrogate in Python.
    message = {
        "role": "assistant",
        "content": "가\ud83d나",
        "thinking": "t\ud83d",
        "tool_calls": [
            {
                "function": {
                    "name": "t",
                    "arguments": {"a": "x\ud83d", "b": {"c": ["y\udc00", 1]}, "k\ud83d": True},
                }
            },
            {"function": {"name": "u", "arguments": "bad\ud83d"}},
        ],
    }
    raw = json.dumps({"message": message, "done": True}, ensure_ascii=True)
    assert "\\ud83d" in raw
    server = Server({"/api/chat": httpx.Response(200, content=raw.encode("ascii"))})
    response = make(server).chat([Message("user", "hi")])

    assert response.text == "가?나"
    assert response.thinking == "t?"
    assert response.tool_calls[0].arguments == {"a": "x?", "b": {"c": ["y?", 1]}, "k?": True}
    assert response.tool_calls[1].arguments == {"_raw": "bad?"}
    dumped = json.dumps(
        [response.text, response.thinking, [c.arguments for c in response.tool_calls]], ensure_ascii=False
    )
    dumped.encode("utf-8")  # must not raise


def test_valid_emoji_survives_cleaning():
    body = reply({"role": "assistant", "content": "감사합니다 😀"})
    assert make(Server({"/api/chat": body})).chat([Message("user", "hi")]).text == "감사합니다 😀"


# --- errors ---


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("timed out"),
        httpx.ConnectError("connection refused"),
        httpx.RemoteProtocolError("server disconnected"),
    ],
)
def test_transport_errors_become_provider_error(error):
    with pytest.raises(ProviderError, match=type(error).__name__):
        make(Server({"/api/chat": error})).chat([Message("user", "hi")])


def test_http_error_includes_truncated_body():
    text = '{"error": "\\"qwen2.5:7b-instruct\\" does not support thinking"}' + "x" * 1000
    server = Server({"/api/chat": httpx.Response(400, text=text)})
    with pytest.raises(ProviderError) as info:
        make(server, think=True).chat([Message("user", "hi")])

    message = str(info.value)
    assert "400" in message
    assert "does not support thinking" in message
    assert len(message) < 450
    assert server.count("/api/chat") == 1  # no retries


def test_invalid_json_raises():
    server = Server({"/api/chat": httpx.Response(200, text="<html>bad gateway</html>")})
    with pytest.raises(ProviderError, match="invalid JSON"):
        make(server).chat([Message("user", "hi")])


def test_non_object_body_raises():
    server = Server({"/api/chat": httpx.Response(200, json=[1, 2])})
    with pytest.raises(ProviderError, match="non-object"):
        make(server).chat([Message("user", "hi")])


@pytest.mark.parametrize("body", [{"done": True}, {"error": "model not found"}, {"message": "text"}])
def test_body_without_message_raises(body):
    with pytest.raises(ProviderError, match="no message"):
        make(Server({"/api/chat": body})).chat([Message("user", "hi")])


# --- describe / loaded_models / preload ---


def tags(*names: str) -> dict[str, Any]:
    return {"models": [{"name": n, "model": n, "digest": f"sha-{n}"} for n in names]}


def test_describe_with_server():
    server = Server({"/api/tags": tags("qwen2.5:14b-instruct", MODEL), "/api/version": {"version": "0.34.1"}})
    provider = make(server, num_ctx=8192, think=False)

    expected = {
        "provider": "ollama",
        "model": MODEL,
        "base_url": DEFAULT_BASE_URL,
        "num_ctx": 8192,
        "keep_alive": "60m",
        "think": False,
        "digest": f"sha-{MODEL}",
        "ollama_version": "0.34.1",
    }
    assert provider.describe() == expected
    assert provider.describe() == expected
    assert server.count("/api/tags") == 1  # cached
    assert server.count("/api/version") == 1
    assert server.count("/api/chat") == 0


def test_describe_matches_implicit_latest_tag():
    server = Server({"/api/tags": tags("llama3:latest"), "/api/version": {"version": "0.34.1"}})
    provider = OllamaProvider("llama3", transport=httpx.MockTransport(server))

    assert provider.describe()["digest"] == "sha-llama3:latest"


def test_describe_unknown_model_has_no_digest():
    server = Server({"/api/tags": tags("other:1b"), "/api/version": {"version": "0.34.1"}})
    description = make(server).describe()

    assert description["digest"] is None
    assert description["ollama_version"] == "0.34.1"


@pytest.mark.parametrize(
    "answer",
    [
        httpx.ConnectError("connection refused"),
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"models": "oops", "version": None}),
    ],
)
def test_describe_never_raises(answer):
    provider = make(Server({"/api/tags": answer, "/api/version": answer}))
    description = provider.describe()

    assert description["provider"] == "ollama"
    assert description["model"] == MODEL
    assert description["digest"] is None
    assert description["ollama_version"] is None


def test_loaded_models():
    ps = {
        "models": [
            {"name": MODEL, "model": MODEL, "size": 6_000_000_000, "size_vram": 5_500_000_000},
            {"model": "other:1b"},
        ]
    }
    provider = make(Server({"/api/ps": ps}))

    assert provider.loaded_models() == [
        {"name": MODEL, "size_vram": 5_500_000_000},
        {"name": "other:1b", "size_vram": 0},
    ]


@pytest.mark.parametrize(
    "answer", [httpx.ConnectError("refused"), httpx.Response(500, text="boom"), {"models": None}, {}]
)
def test_loaded_models_empty_on_failure(answer):
    assert make(Server({"/api/ps": answer})).loaded_models() == []


def test_preload():
    server = Server(
        {"/api/chat": {"model": MODEL, "done": True, "done_reason": "load", "load_duration": 2_500_000_000}}
    )
    provider = make(server, num_ctx=8192, think=False)

    assert provider.preload() == pytest.approx(2500.0)
    body = server.bodies()[0]
    assert body == {
        "model": MODEL,
        "messages": [],
        "stream": False,
        "keep_alive": "60m",
        "options": {"num_ctx": 8192},
    }


def test_preload_failure_raises():
    server = Server({"/api/chat": httpx.Response(404, json={"error": "model 'x' not found"})})
    with pytest.raises(ProviderError, match="404"):
        make(server).preload()


# --- base URL / lifecycle ---


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:1234/")
    server = Server()
    provider = make(server)
    provider.chat([Message("user", "hi")])

    assert provider.base_url == "http://ollama.test:1234"
    assert str(server.requests[0].url) == "http://ollama.test:1234/api/chat"
    assert provider.describe()["base_url"] == "http://ollama.test:1234"


def test_explicit_base_url_wins_over_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:1234")
    server = Server()
    make(server, base_url="http://explicit.test:9").chat([Message("user", "hi")])

    assert server.requests[0].url.host == "explicit.test"


def test_default_base_url(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert make(Server()).base_url == DEFAULT_BASE_URL


def test_close_and_context_manager():
    with make(Server()) as provider:
        provider.chat([Message("user", "hi")])
    with pytest.raises(RuntimeError):
        provider.chat([Message("user", "hi")])
