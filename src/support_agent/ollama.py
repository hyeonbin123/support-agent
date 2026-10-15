"""Ollama 고유 API(/api/chat)용 ChatProvider. httpx로 직접 부른다."""

# [파일 역할] Ollama의 고유 API(/api/chat)를 httpx로 직접 부르는 ChatProvider 구현. 프로젝트의 중립 메시지
# 타입을 Ollama의 요청 JSON으로 바꾸고, 응답을 ChatResponse로 되돌린다. 모델 적재(preload), 적재된 모델
# 조회(/api/ps), manifest용 정보(모델 digest, 서버 버전)도 맡는다.
# [설계 이유] 측정과 서비스 모두 로컬 Ollama 모델로 했다(README 스택). SDK 대신 httpx로 직접 부르면 요청
# 본문(options, keep_alive, think)과 시간 값을 그대로 다룰 수 있고, 테스트에서는 transport를 바꿔 끼워 서버
# 없이 돌린다.
# [주의] 같은 모델을 다른 실행기 옵션(num_ctx, num_gpu)으로 부르면 Ollama가 모델을 다시 읽는다. 모델 파일이
# 느린 디스크에 있어 오래 걸리므로, 옵션은 인스턴스마다 고정하고 keep_alive를 길게 준다.
# [연결] run.py가 에이전트·시뮬레이터용으로 하나씩 만들고(같은 모델이면 하나를 공유), service/app.py가
# 서비스용으로 만든다.

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

# httpx: 동기 HTTP 클라이언트. 시간 제한을 연결과 읽기로 나눠 줄 수 있고, 테스트용 가짜 transport를 넣을 수
# 있다.
import httpx

from support_agent.chat import (
    ChatProvider,
    ChatResponse,
    ContextOverflowError,
    Message,
    ProviderError,
    ToolCall,
    Usage,
)

# Ollama 기본 주소. 환경 변수 OLLAMA_BASE_URL로 바꿀 수 있다(README 실행 절). 메타 조회(/api/tags 등)는 바로
# 답하거나 아예 답하지 않으므로 짧은 10초 제한을 쓴다. _UNSET은 '아직 조회 안 함'과 '조회 결과 None'을
# 구분하는 표시다.
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_META_TIMEOUT_S = 10.0  # /api/tags, /api/version, /api/ps는 바로 답하거나 아예 답하지 않는다
_UNSET: Any = object()


# 문자열 안의 짝 없는 서로게이트 문자를 대체 문자로 바꾼다. 그대로 두면 나중에 기록(episodes.jsonl)을 UTF-8로
# 쓸 때 오류가 난다. dict·list는 재귀로 처리한다.
def _clean(value: Any) -> Any:
    """모든 문자열의 짝 없는 서로게이트를 바꾼다. 그대로 두면 나중에 UTF-8로 쓸 수 없다."""
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, dict):
        return {_clean(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


# Ollama가 나노초로 보고하는 시간 값을 밀리초로 바꾼다.
def _ms(raw: dict[str, Any], key: str) -> float:
    return (raw.get(key, 0) or 0) / 1e6  # 서버는 나노초로 보고한다


# 중립 Message → Ollama 메시지 JSON. 도구 결과는 Ollama 고유 API의 방식대로 호출 id가 아니라 도구
# 이름(tool_name)으로 짝짓는다. 상담원의 도구 호출은 function 객체로 다시 써서, 다음 요청에서 모델이 자기가
# 부른 호출을 본다.
def _wire_message(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_name": m.tool_name or "", "content": m.content}
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.role == "assistant" and m.tool_calls:
        # arguments는 JSON 객체여야 한다. 서버는 여기에 문자열이 오면 거절한다
        out["tool_calls"] = [
            {"function": {"name": c.name, "arguments": dict(c.arguments)}} for c in m.tool_calls
        ]
    return out


# 중립 도구 정의({name, description, parameters}) → Ollama의 function 도구 형식. 인자가 없는 도구도 빈 object
# 스키마를 넣는다.
def _wire_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        },
    }


# 모델이 보낸 도구 인자를 dict로 바꾼다. Ollama는 보통 dict를 주지만 문자열 JSON이 올 수도 있어 둘 다 받는다.
# 해석할 수 없거나 dict가 아니면 {"_raw": 원문}으로 남겨 도구 실행 단계가 인자 오류로 처리하게 한다(조용히
# 버리지 않는다).
def _parse_arguments(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except ValueError:
            return {"_raw": _clean(args)}
        return _clean(parsed) if isinstance(parsed, dict) else {"_raw": _clean(args)}
    if isinstance(args, dict):
        return _clean(args)
    return {"_raw": _clean(args)}


# 응답의 tool_calls 목록을 ToolCall로 바꾼다. 목록 구조 자체가 깨졌으면(모델이 아니라 서버 응답 형식 문제)
# ProviderError로 올린다. id가 없으면 빈 문자열이다.
def _parse_tool_calls(items: Any) -> tuple[ToolCall, ...]:
    if not items:
        return ()
    if not isinstance(items, list):
        raise ProviderError(f"ollama: tool_calls is not a list: {str(items)[:300]}")
    calls = []
    for item in items:
        fn = item.get("function") if isinstance(item, dict) else None
        if not isinstance(fn, dict):
            raise ProviderError(f"ollama: malformed tool call: {str(item)[:300]}")
        calls.append(
            ToolCall(
                name=_clean(str(fn.get("name") or "")),
                arguments=_parse_arguments(fn.get("arguments")),
                id=_clean(str(item.get("id") or "")),
            )
        )
    return tuple(calls)


# 모델이 한 토큰을 끝없이 되풀이할 때 Ollama가 HTTP 500으로 끊으며 내는 오류 문구. 이것은 서버 장애가 아니라
# 모델 출력의 문제이므로 인프라 오류로 세지 않는다(docs/experiments.md 1단계 결과 1: 에이전트면 형식 오류
# 재시도, 시뮬레이터면 infra_error).
REPEAT_LIMIT_ERROR = "token repeat limit reached"

# 컨텍스트 넘침을 알아보는 표시. truncate: false인 요청의 프롬프트가 num_ctx보다 길면 Ollama 0.35.1은 HTTP
# 400과 함께 {"error": {"type": "exceed_context_size_error", "message": "request (N tokens) exceeds the
# available context size (M tokens), ...", "n_prompt_tokens": N, "n_ctx": M}}를 돌려준다(같은 PC에서
# rag-doc-qa 프로젝트의 v11 스모크가 확인한 본문). 본문 꼴이 버전마다 바뀔 수 있어 type과 문구 둘 다 본다.
_OVERFLOW_TYPE = "exceed_context_size_error"
_OVERFLOW_TEXT = "exceeds the available context size"
_OVERFLOW_COUNTS = re.compile(r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)")


# 컨텍스트 넘침 오류 응답이면 ContextOverflowError를, 아니면 None을 돌려준다. 서버가 알려 준 프롬프트
# 토큰 수와 num_ctx를 함께 담아, 에이전트 루프가 LLM 호출 기록에 남길 수 있게 한다.
def _overflow_error(method: str, path: str, resp: httpx.Response) -> ContextOverflowError | None:
    text = resp.text
    if resp.status_code != 400 or (_OVERFLOW_TYPE not in text and _OVERFLOW_TEXT not in text):
        return None
    prompt_tokens = num_ctx = 0
    try:
        error = resp.json().get("error")
    except (ValueError, AttributeError):
        error = None
    if isinstance(error, dict):
        prompt_tokens = int(error.get("n_prompt_tokens") or 0)
        num_ctx = int(error.get("n_ctx") or 0)
    if not prompt_tokens:
        found = _OVERFLOW_COUNTS.search(text)
        if found:
            prompt_tokens, num_ctx = int(found.group(1)), int(found.group(2))
    return ContextOverflowError(
        f"ollama: {method} {path} returned HTTP 400 (context overflow): {text[:300]}",
        prompt_tokens=prompt_tokens,
        num_ctx=num_ctx,
    )


# 주소에 든 인증 정보(http://사용자:비밀번호@호스트)를 가린다. manifest의 base_url은 공식 실행이면 공개
# 저장소에 커밋되므로(reports/), OLLAMA_BASE_URL에 인증 정보를 넣어 쓰는 경우에도 기록에는 남지 않게 한다.
# 인증 정보가 없는 주소(기본 http://127.0.0.1:11434)는 글자까지 그대로라 지난 기록과 짝짓기가 바뀌지 않는다.
# 실제 요청은 가리지 않은 주소로 간다(self.base_url).
def redact_url(url: str) -> str:
    """주소의 사용자 정보(아이디·비밀번호)를 `***`로 바꾼다. 사용자 정보가 없으면 그대로 돌려준다."""
    parts = urlsplit(url)
    if parts.username is None and parts.password is None:
        return url
    host = parts.hostname or ""
    if ":" in host:  # IPv6 주소는 대괄호로 다시 감싼다
        host = f"[{host}]"
    netloc = f"***@{host}" + (f":{parts.port}" if parts.port is not None else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


# Ollama 제공자. 인자: 모델 이름, 컨텍스트 크기, keep_alive(요청 뒤 모델을 메모리에 둘 시간, 기본 60분),
# think(사고 모드 켜기/끄기, None이면 보내지 않음), num_gpu(GPU에 올릴 층 수, 9단계에서 시뮬레이터 일부를
# CPU로 옮길 때 씀), base_url, 시간 제한, 테스트용 transport.
class OllamaProvider(ChatProvider):
    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        num_ctx: int = 16384,
        keep_alive: str | int = "60m",
        think: bool | None = None,
        num_gpu: int | None = None,
        base_url: str | None = None,
        timeout_s: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        # 인스턴스마다 고정한다. 요청 사이에 실행기 옵션이 달라지면 서버가 모델을 다시 읽는다.
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.think = think
        self.num_gpu = num_gpu  # 0이면 모든 층을 CPU에 둔다. None이면 서버가 정한다
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        # 첫 요청은 느린 디스크에서 모델을 읽어야 할 수 있어 읽기 시간 제한을 길게(기본 600초) 둔다. 연결
        # 제한은 10초 이하로 짧게 둔다.
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
            transport=transport,
        )
        self._digest: Any = _UNSET
        self._version: Any = _UNSET

    # 모델 실행기 옵션(num_gpu). preload()와 chat()에서 같아야 한다. 실행기 옵션이 하나라도 다르면 적재한
    # 모델을 다시 읽는다.
    def _runner_options(self) -> dict[str, Any]:
        return {} if self.num_gpu is None else {"num_gpu": self.num_gpu}

    # HTTP 연결을 닫는다. with 문으로 쓰면 자동으로 닫힌다.
    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OllamaProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # HTTP 요청 하나. 연결 실패, 2xx가 아닌 응답, JSON이 아닌 본문, 객체가 아닌 JSON을 모두 ProviderError로
    # 바꾼다. 에피소드 실행기는 이것을 infra_error로, 서비스는 대체 안내문으로 처리한다. 오류 문구에는 응답 앞
    # 300자만 넣는다.
    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
        try:
            resp = self._http.request(method, path, json=body, **kwargs)
        except httpx.HTTPError as e:
            raise ProviderError(f"ollama: {method} {path} failed: {type(e).__name__}: {e}") from e
        # 컨텍스트 넘침(HTTP 400)은 일반 서버 오류와 나눠 올린다. 에이전트 루프가 이것만 컨텍스트 한도 종료로
        # 바꾼다(agent.py).
        overflow = _overflow_error(method, path, resp)
        if overflow is not None:
            raise overflow
        if not resp.is_success:
            raise ProviderError(
                f"ollama: {method} {path} returned HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderError(f"ollama: {method} {path} returned invalid JSON: {resp.text[:300]}") from e
        if not isinstance(data, dict):
            raise ProviderError(f"ollama: {method} {path} returned a non-object body: {resp.text[:300]}")
        return data

    # 대화 전체와 도구 목록을 보내 상담원(또는 시뮬레이터) 한 턴을 받는다. stream=False로 한 번에 받는다.
    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        # num_predict는 생성 토큰 한도(max_tokens)다. 한도에 닿으면 done_reason이 length가 되어 agent.py가
        # 잘린 답(cut_off)으로 처리한다. seed는 derive_seed가 만든 호출별 고정 값이다.
        options: dict[str, Any] = {
            "num_ctx": self.num_ctx,
            "temperature": temperature,
            "num_predict": max_tokens,
            **self._runner_options(),
        }
        if seed is not None:
            options["seed"] = seed
        # truncate: false(2026-10-15부터): 프롬프트가 num_ctx보다 길면 서버가 앞부분(시스템 프롬프트 포함)을
        # 조용히 잘라 답하지 않고 HTTP 400으로 거절하게 한다. 그 오류는 _request가 ContextOverflowError로
        # 올린다. 프롬프트가 num_ctx 안에 들면 응답은 이 필드가 없을 때와 같다(잘림이 일어나지 않으므로).
        # 기록된 평가 실행에서는 같은 에피소드 안에서 프롬프트 길이가 줄어든 호출이 하나도 없어 잘림이 없었다
        # (docs/experiments.md 공통 규칙의 2026-10-15 메모, tests/test_recorded_numbers.py가 확인).
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_wire_message(m) for m in messages],
            "stream": False,
            "keep_alive": self.keep_alive,
            "truncate": False,
            "options": options,
        }
        if tools:
            body["tools"] = [_wire_tool(t) for t in tools]
        # think 필드는 값을 줬을 때만 보낸다. 사고 모드가 없는 모델은 think 값이 오면 어떤 값이든 400으로
        # 거절한다. 8단계 qwen3.5:4b는 think=False(비사고 모드)로 쟀다.
        if self.think is not None:  # 사고 모드가 없는 모델은 think 값이 무엇이든 400으로 답한다
            body["think"] = self.think

        # wall_ms는 클라이언트가 HTTP 호출 앞뒤로 잰 시간이다(서버의 시간 값과 따로 남김).
        start = time.perf_counter()
        try:
            raw = self._request("POST", "/api/chat", body)
        except ProviderError as error:
            if REPEAT_LIMIT_ERROR not in str(error):
                raise
            # 모델이 한 토큰을 되풀이하는 데 빠져 서버가 끊었다. 서버 장애가 아니라 모델 출력이 잘못된
            # 것이므로 비어 있고 잘린 답으로 보고한다: 빈 본문 + finish_reason repeat_limit. agent.py에서는
            # 빈 응답(형식 오류)이 되어 다시 시킨다.
            wall_ms = (time.perf_counter() - start) * 1000.0
            return ChatResponse(text="", usage=Usage(wall_ms=wall_ms), finish_reason="repeat_limit")
        wall_ms = (time.perf_counter() - start) * 1000.0

        # 응답에 message가 없으면 서버 쪽 오류(error 필드 등)로 보고 ProviderError로 올린다.
        msg = raw.get("message")
        if not isinstance(msg, dict):
            detail = raw.get("error") or json.dumps(raw, ensure_ascii=False)[:300]
            raise ProviderError(f"ollama: response has no message: {detail}")
        content = msg.get("content") or ""
        if not isinstance(content, str):
            raise ProviderError(f"ollama: message content is not text: {str(content)[:300]}")
        return ChatResponse(
            text=_clean(content),
            tool_calls=_parse_tool_calls(msg.get("tool_calls")),
            usage=Usage(
                # prompt_eval_count는 캐시된 토큰까지 포함한 전체 프롬프트 길이다(0단계에서 확인). 컨텍스트
                # 한도 검사가 이 값을 쓴다.
                prompt_tokens=raw.get("prompt_eval_count", 0) or 0,
                completion_tokens=raw.get("eval_count", 0) or 0,
                wall_ms=wall_ms,
                load_ms=_ms(raw, "load_duration"),
                prompt_eval_ms=_ms(raw, "prompt_eval_duration"),
                eval_ms=_ms(raw, "eval_duration"),
            ),
            finish_reason=raw.get("done_reason"),
            thinking=_clean(msg.get("thinking") or ""),
        )

    # 빈 메시지로 /api/chat을 불러 모델만 올려 둔다. 실행기가 측정 전에 부르고 걸린 시간(ms)을 출력한다.
    # chat()과 같은 num_ctx·실행기 옵션을 써야 첫 에피소드에서 다시 읽지 않는다.
    def preload(self) -> float:
        """이 제공자의 실행기 옵션으로 모델을 올린다. 적재에 걸린 시간을 ms로 돌려준다."""
        body = {
            "model": self.model,
            "messages": [],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"num_ctx": self.num_ctx, **self._runner_options()},
        }
        started = time.perf_counter()
        self._request("POST", "/api/chat", body)
        return (time.perf_counter() - started) * 1000

    # 지금 서버 메모리에 올라 있는 모델과 GPU에 올라간 크기(size_vram). run.py가 nvidia-smi의 전체 GPU
    # 사용량에서 Ollama 몫을 빼, 다른 작업이 GPU를 쓰고 있는지 판단할 때 쓴다. 서버에 물을 수 없으면 실행을
    # 막지 않도록 빈 목록을 돌려준다.
    def loaded_models(self) -> list[dict[str, Any]]:
        """서버가 지금 메모리에 들고 있는 모델들. 서버에 물을 수 없으면 []."""
        try:
            models = self._request("GET", "/api/ps", timeout=_META_TIMEOUT_S).get("models") or []
            return [
                {
                    "name": m.get("name") or m.get("model") or "",
                    "size": m.get("size", 0) or 0,  # size_vram과 함께 보면 GPU에 올라간 비율을 안다
                    "size_vram": m.get("size_vram", 0) or 0,
                }
                for m in models
                if isinstance(m, dict)
            ]
        except Exception:
            return []

    # 모델의 digest(내용 해시)를 /api/tags에서 찾는다. 태그 없이 적은 이름은 ':latest'로도 찾는다. 같은
    # 이름이라도 다시 받은 모델이면 digest가 달라지므로, manifest에 남겨 두 실행이 같은 모델인지 확인한다.
    def _fetch_digest(self) -> str | None:
        models = self._request("GET", "/api/tags", timeout=_META_TIMEOUT_S).get("models") or []
        wanted = {self.model} if ":" in self.model else {self.model, f"{self.model}:latest"}
        for m in models:
            if isinstance(m, dict) and (m.get("name") in wanted or m.get("model") in wanted):
                return m.get("digest")
        return None

    # Ollama 서버 버전. 8단계에서 0.34에서 0.35로 바뀌어 기준을 다시 쟀듯, 버전이 다른 기록끼리는 짝짓지
    # 않는다(README 8단계).
    def _fetch_version(self) -> str | None:
        return self._request("GET", "/api/version", timeout=_META_TIMEOUT_S).get("version")

    # manifest의 agent_provider·user_provider 값. analyze same-setup이 이 값을 비교한다. digest와 버전은 한
    # 번만 조회해 두되, 조회에 실패하면 저장하지 않아 다음 호출에서 다시 시도한다.
    # base_url은 인증 정보를 가린 주소다(redact_url). truncate는 chat()이 보내는 값(2026-10-15부터 False)이고,
    # 이 필드가 없는 옛 manifest는 analyze.setup_differences가 같은 값으로 보고 비교한다.
    def describe(self) -> dict[str, Any]:
        if self._digest is _UNSET:
            try:
                self._digest = self._fetch_digest()
            except Exception:
                pass
        if self._version is _UNSET:
            try:
                self._version = self._fetch_version()
            except Exception:
                pass
        return {
            "provider": self.name,
            "model": self.model,
            "base_url": redact_url(self.base_url),
            "num_ctx": self.num_ctx,
            "keep_alive": self.keep_alive,
            "truncate": False,
            "think": self.think,
            "num_gpu": self.num_gpu,
            "digest": None if self._digest is _UNSET else self._digest,
            "ollama_version": None if self._version is _UNSET else self._version,
        }
