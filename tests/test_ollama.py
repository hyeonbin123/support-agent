"""OllamaProvider against httpx.MockTransport. No real server, no GPU."""

# [테스트 묶음] Ollama 연결(ollama.OllamaProvider)을 실제 서버 없이 httpx.MockTransport로 확인한다: 보내는
# 요청 본문, 응답 해석, 오류를 ProviderError로 바꾸는 것, manifest용 정보(describe), 적재된 모델 조회, 미리
# 올리기(preload), 주소 설정.
# [왜 중요한가] 측정의 재현성은 요청 본문에 달려 있다: seed, temperature, num_ctx, think, num_gpu가 의도대로
# 가야 하고, 실행기 옵션이 요청마다 다르면 Ollama가 모델을 다시 읽는다. 응답이 이상할 때(깨진 JSON, 잘못된
# 도구 호출, 짝 없는 서로게이트 문자) 조용히 넘어가지 않고 정해진 방식으로 처리돼야 기록이 깨지지 않는다.
# [방법] Server 클래스가 경로별 응답 표로 가짜 서버 노릇을 하며 받은 요청을 모두 기록한다. GPU와 모델 없이
# 돈다.

from __future__ import annotations

# httpx: MockTransport로 요청을 가로채고, 오류 예외(ReadTimeout 등)와 응답 객체를 직접 만든다.
import json
from typing import Any

import httpx
import pytest

from support_agent.chat import Message, ProviderError, ToolCall
from support_agent.ollama import DEFAULT_BASE_URL, OllamaProvider

# 시험용 모델 이름과 도구 정의 하나 (에이전트가 보내는 것과 같은 name·description·parameters 형태).
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


# 모든 테스트에서 OLLAMA_BASE_URL 환경 변수를 지운다. 개발자 PC의 설정이 기본 주소 테스트를 흔들지 않게 한다.
@pytest.fixture(autouse=True)
def _no_env_base_url(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)


# Ollama /api/chat의 정상 응답 본문을 만든다 (기본은 '네'라는 답, 끝난 이유 stop). extra로 사용량·시간 칸을
# 덧붙인다.
def reply(message: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    body = {
        "model": MODEL,
        "message": {"role": "assistant", "content": "네"} if message is None else message,
        "done": True,
        "done_reason": "stop",
    }
    body.update(extra)
    return body


# 가짜 Ollama 서버. 경로 → (dict면 200 JSON, Response면 그대로, 예외면 던짐) 표로 답하고, 받은 요청을 순서대로
# 모은다. 표에 없는 경로는 404다.
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


# 가짜 서버에 붙은 제공자를 만든다. 나머지 키워드는 제공자 설정(num_ctx, think, num_gpu 등)이다.
def make(server: Server, **kwargs: Any) -> OllamaProvider:
    return OllamaProvider(MODEL, transport=httpx.MockTransport(server), **kwargs)


# --- request body ---


# 기본 요청 본문: POST /api/chat, stream 끔, keep_alive 60분, options에 num_ctx
# 16384·temperature·seed·num_predict 1024. 도구를 주지 않으면 tools 칸이 없고, think를 정하지 않으면 think
# 칸도 보내지 않는다. keep_alive를 길게 주는 것은 모델 파일이 느린 디스크에 있어 다시 올리는 데 오래 걸리기
# 때문이다.
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


# seed가 없으면 options에서 빠지고, 인스턴스에 준 num_ctx·keep_alive와 호출의 max_tokens(num_predict)가 그대로
# 쓰인다.
def test_seed_omitted_when_none_and_instance_options_used():
    server = Server()
    provider = make(server, num_ctx=8192, keep_alive="10m")
    provider.chat([Message("user", "hi")], max_tokens=256)

    body = server.bodies()[0]
    assert body["keep_alive"] == "10m"
    assert body["options"] == {"num_ctx": 8192, "temperature": 0.0, "num_predict": 256}


# think는 options 안이 아니라 본문 맨 위 칸으로 보낸다(True·False 모두). 8단계 qwen3.5:4b의 비사고 모드가 이
# 값으로 정해진다.
@pytest.mark.parametrize("think", [True, False])
def test_think_sent_at_top_level_when_set(think):
    server = Server()
    make(server, think=think).chat([Message("user", "hi")])

    body = server.bodies()[0]
    assert body["think"] is think
    assert "think" not in body["options"]


# 도구 정의는 Ollama 형식 {"type": "function", "function": ...}으로 감싸 보낸다.
def test_tools_are_wrapped():
    server = Server()
    make(server).chat([Message("user", "hi")], tools=[TOOL])

    assert server.bodies()[0]["tools"] == [{"type": "function", "function": TOOL}]


# 대화 기록 변환: 하니스 표시(harness)와 전달 여부(delivered) 같은 내부 표시는 보내지 않고, 도구 호출은
# function 형태로, 도구 결과는 tool_name과 내용으로 보낸다. 호출 id는 Ollama에 보내지 않는다.
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


# 한국어가 든 요청 본문이 UTF-8 JSON으로 그대로 가는지 확인한다.
def test_request_body_is_utf8_json():
    server = Server()
    make(server).chat([Message("user", "환불해 주세요")])

    assert json.loads(server.requests[0].content.decode("utf-8"))["messages"][0]["content"] == "환불해 주세요"


# --- response parsing ---


# 글 답의 해석: 내용, 도구 호출 없음, 사고 글 없음, 끝난 이유.
def test_text_reply():
    response = make(Server()).chat([Message("user", "hi")])

    assert response.text == "네"
    assert response.tool_calls == ()
    assert response.thinking == ""
    assert response.finish_reason == "stop"


# 도구 호출이 여러 개 오면 모두 ToolCall로 바꾸고(실행은 루프가 첫 번째만), id가 있으면 남기고 없으면 빈
# 문자열이다.
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


# 도구 인자의 여러 모양: JSON 문자열은 풀어 쓰고, 풀 수 없거나 dict가 아니면 {"_raw": 원문}으로 남긴다. 그러면
# 도구 실행 단계가 인자 오류(invalid_arguments)로 처리해, 모델의 형식 실수가 조용히 사라지지 않고 기록된다.
# None은 빈 인자다.
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


# arguments 칸이 아예 없으면 빈 인자로 본다.
def test_tool_call_without_arguments_key():
    message = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t"}}]}
    response = make(Server({"/api/chat": reply(message)})).chat([Message("user", "hi")])

    assert response.tool_calls == (ToolCall("t", {}),)


# tool_calls 자체가 목록이 아니거나 function 칸이 없거나 이상하면 서버 응답이 잘못된 것이므로 ProviderError다
# (에피소드는 infra_error가 된다).
@pytest.mark.parametrize("tool_calls", ["oops", ["oops"], [{"id": "x"}], [{"function": "t"}]])
def test_malformed_tool_calls_raise(tool_calls):
    message = {"role": "assistant", "content": "", "tool_calls": tool_calls}
    with pytest.raises(ProviderError, match="tool"):
        make(Server({"/api/chat": reply(message)})).chat([Message("user", "hi")])


# 사고 글, 끝난 이유(length면 루프가 잘린 답으로 처리), 토큰 수, Ollama의 나노초 시간 칸을 ms로 바꾸는 것을
# 확인한다. 기록의 호출당 시간·토큰 수가 이 값이다.
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


# 사용량 칸이 없거나 None이면 0으로 채우고, 내용이 None이면 빈 글이다. 기록을 쓰다가 None 때문에 죽지 않게
# 한다.
def test_missing_usage_fields_default_to_zero():
    body = {"message": {"role": "assistant", "content": None}, "prompt_eval_count": None}
    response = make(Server({"/api/chat": body})).chat([Message("user", "hi")])

    assert response.text == ""
    assert response.finish_reason is None
    usage = response.usage
    assert (usage.prompt_tokens, usage.completion_tokens) == (0, 0)
    assert (usage.load_ms, usage.prompt_eval_ms, usage.eval_ms) == (0.0, 0.0, 0.0)


# 모델 출력에 짝 없는 서로게이트(\ud83d 같은 반쪽 이모지)가 섞이면 내용·사고 글·도구 인자(키와 중첩
# 값까지)에서 '?'로 바꾼다. 그대로 두면 기록을 UTF-8로 쓸 때 예외가 나 에피소드 기록이 깨진다.
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


# 정상적인 이모지는 바꾸지 않는다.
def test_valid_emoji_survives_cleaning():
    body = reply({"role": "assistant", "content": "감사합니다 😀"})
    assert make(Server({"/api/chat": body})).chat([Message("user", "hi")]).text == "감사합니다 😀"


# --- errors ---


# 시간 초과, 연결 거부, 연결 끊김 같은 전송 오류는 예외 종류 이름을 담은 ProviderError로 바뀐다. 에피소드
# 실행기는 이것을 infra_error로 기록하고 분모에서 뺀다.
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


# HTTP 오류(예: 사고 모드가 없는 모델에 think를 보내 400)는 상태 코드와 응답 앞부분(300자)을 담아
# ProviderError로 바뀐다. 오류 문구가 너무 길어지지 않고, 재시도 없이 한 번만 요청한다.
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


# JSON이 아닌 응답(프록시의 HTML 오류 페이지 등)은 ProviderError다.
def test_invalid_json_raises():
    server = Server({"/api/chat": httpx.Response(200, text="<html>bad gateway</html>")})
    with pytest.raises(ProviderError, match="invalid JSON"):
        make(server).chat([Message("user", "hi")])


# JSON이지만 객체가 아닌 응답도 ProviderError다.
def test_non_object_body_raises():
    server = Server({"/api/chat": httpx.Response(200, json=[1, 2])})
    with pytest.raises(ProviderError, match="non-object"):
        make(server).chat([Message("user", "hi")])


# message 칸이 없거나(오류 본문 포함) 객체가 아니면 'no message' ProviderError다.
@pytest.mark.parametrize("body", [{"done": True}, {"error": "model not found"}, {"message": "text"}])
def test_body_without_message_raises(body):
    with pytest.raises(ProviderError, match="no message"):
        make(Server({"/api/chat": body})).chat([Message("user", "hi")])


# --- describe / loaded_models / preload ---


# /api/tags 응답(설치된 모델 목록과 digest)을 만든다.
def tags(*names: str) -> dict[str, Any]:
    return {"models": [{"name": n, "model": n, "digest": f"sha-{n}"} for n in names]}


# describe()는 manifest의 제공자 정보다: 모델, 주소, num_ctx, keep_alive, think, num_gpu, 모델 digest, Ollama
# 버전. analyze same-setup이 두 실행을 짝지어도 되는지 이 값으로 확인한다. digest와 버전은 한 번만 조회해
# 둔다(다시 불러도 요청이 늘지 않음).
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
        "num_gpu": None,
        "digest": f"sha-{MODEL}",
        "ollama_version": "0.34.1",
    }
    assert provider.describe() == expected
    assert provider.describe() == expected
    assert server.count("/api/tags") == 1  # cached
    assert server.count("/api/version") == 1
    assert server.count("/api/chat") == 0


# 태그 없이 적은 모델 이름은 ':latest'로도 찾아 digest를 얻는다.
def test_describe_matches_implicit_latest_tag():
    server = Server({"/api/tags": tags("llama3:latest"), "/api/version": {"version": "0.34.1"}})
    provider = OllamaProvider("llama3", transport=httpx.MockTransport(server))

    assert provider.describe()["digest"] == "sha-llama3:latest"


# 설치되지 않은 모델이면 digest는 None이고 버전은 그대로 남는다.
def test_describe_unknown_model_has_no_digest():
    server = Server({"/api/tags": tags("other:1b"), "/api/version": {"version": "0.34.1"}})
    description = make(server).describe()

    assert description["digest"] is None
    assert description["ollama_version"] == "0.34.1"


# 서버에 닿지 못하거나 응답이 이상해도 describe()는 예외를 내지 않고 digest·버전을 None으로 둔다. manifest를
# 쓰다가 실행이 죽지 않게 한다.
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


# /api/ps에서 올라 있는 모델의 이름, 전체 크기, GPU에 올라간 크기(size_vram)를 읽는다. 두 값을 비교하면 모델이
# GPU에 얼마나 올라갔는지 알 수 있다 (8단계 스모크 관문). 이름이 없으면 model 칸을 쓴다.
def test_loaded_models():
    ps = {
        "models": [
            {"name": MODEL, "model": MODEL, "size": 6_000_000_000, "size_vram": 5_500_000_000},
            {"model": "other:1b"},
        ]
    }
    provider = make(Server({"/api/ps": ps}))

    # size next to size_vram tells how much of a model sits on the GPU (stage 8 smoke gate)
    assert provider.loaded_models() == [
        {"name": MODEL, "size": 6_000_000_000, "size_vram": 5_500_000_000},
        {"name": "other:1b", "size": 0, "size_vram": 0},
    ]


# /api/ps를 읽지 못하면 빈 목록을 돌려준다. 실행기의 GPU 확인이 Ollama 조회 실패 때문에 막히지 않게 한다.
@pytest.mark.parametrize(
    "answer", [httpx.ConnectError("refused"), httpx.Response(500, text="boom"), {"models": None}, {}]
)
def test_loaded_models_empty_on_failure(answer):
    assert make(Server({"/api/ps": answer})).loaded_models() == []


# preload는 빈 메시지로 /api/chat을 불러 모델만 올린다. 본문에는 chat()과 같은 num_ctx(실행기 옵션)가 들어가야
# 첫 에피소드에서 모델을 다시 읽지 않는다. 걸린 시간은 서버의 load_duration이 아니라 클라이언트가 잰 값이다.
def test_preload():
    server = Server(
        {"/api/chat": {"model": MODEL, "done": True, "done_reason": "load", "load_duration": 2_500_000_000}}
    )
    provider = make(server, num_ctx=8192, think=False)

    assert provider.preload() >= 0.0  # measured by the client, not taken from load_duration
    body = server.bodies()[0]
    assert body == {
        "model": MODEL,
        "messages": [],
        "stream": False,
        "keep_alive": "60m",
        "options": {"num_ctx": 8192},
    }


# 모델이 없어 올리지 못하면 ProviderError로 실행을 시작 전에 멈춘다.
def test_preload_failure_raises():
    server = Server({"/api/chat": httpx.Response(404, json={"error": "model 'x' not found"})})
    with pytest.raises(ProviderError, match="404"):
        make(server).preload()


# --- base URL / lifecycle ---


# OLLAMA_BASE_URL 환경 변수로 주소를 바꿀 수 있고, 끝의 '/'는 뗀다. describe()에도 바뀐 주소가 남는다.
def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:1234/")
    server = Server()
    provider = make(server)
    provider.chat([Message("user", "hi")])

    assert provider.base_url == "http://ollama.test:1234"
    assert str(server.requests[0].url) == "http://ollama.test:1234/api/chat"
    assert provider.describe()["base_url"] == "http://ollama.test:1234"


# 생성자에 준 주소가 환경 변수보다 우선한다.
def test_explicit_base_url_wins_over_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:1234")
    server = Server()
    make(server, base_url="http://explicit.test:9").chat([Message("user", "hi")])

    assert server.requests[0].url.host == "explicit.test"


# 아무것도 주지 않으면 기본 주소(127.0.0.1:11434)다.
def test_default_base_url(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert make(Server()).base_url == DEFAULT_BASE_URL


# with 문을 벗어나면 HTTP 연결이 닫혀, 그 뒤의 호출은 RuntimeError다.
def test_close_and_context_manager():
    with make(Server()) as provider:
        provider.chat([Message("user", "hi")])
    with pytest.raises(RuntimeError):
        provider.chat([Message("user", "hi")])


# num_gpu(GPU에 올릴 층 수)를 정하면 preload와 chat 모두에 같은 값이 가야 한다. 실행기 옵션이 요청마다 다르면
# Ollama가 모델을 다시 읽는다. 9단계는 시뮬레이터 층 일부를 CPU에 두려고 이 값을 썼다.
def test_num_gpu_is_sent_with_every_request_so_that_the_model_is_not_reloaded():
    server = Server()
    provider = make(server, num_gpu=0)
    provider.preload()
    provider.chat([Message("user", "hi")])
    assert [body["options"]["num_gpu"] for body in server.bodies()] == [0, 0]
    assert provider.describe()["num_gpu"] == 0
