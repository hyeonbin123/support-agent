# syntax=docker/dockerfile:1
# 위 줄은 파서 지시라 맨 첫 줄이어야 한다: Dockerfile 문법 1.x의 최신 안정판으로 읽어 아래의 캐시 마운트
# (`RUN --mount=type=cache`)를 쓸 수 있게 한다. 캐시 마운트는 BuildKit 빌더가 필요하다(지금 Docker와 Compose의
# 기본 빌더. 옛 빌더 DOCKER_BUILDKIT=0으로는 빌드되지 않는다).
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
# UV_LINK_MODE=copy: 캐시 마운트는 이미지와 다른 파일 시스템이라 uv가 캐시에서 하드 링크를 걸 수 없어 복사한다.
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
# 의존성 층과 프로젝트 층을 나눠 소스만 바뀌면 의존성 층은 다시 쓴다. uv의 내려받기 캐시(/root/.cache/uv)는
# 빌드하는 동안만 붙는 캐시 마운트에 둔다: 내려받은 휠이 .venv와 캐시로 이미지 레이어에 두 벌 남지 않고,
# 의존성 층을 다시 빌드할 때는 같은 캐시에서 설치한다(`--no-cache` 빌드는 이 캐시도 쓰지 않고 다시 내려받는다).
# 설치되는 판은 uv.lock 그대로라(--frozen) 캐시가 결과를 바꾸지 않는다.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s   CMD ["/app/.venv/bin/python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]
# --no-access-log: 고객 API는 세션 id(고객 쪽의 유일한 열쇠)를 경로에 넣으므로 요청 줄을 로그에 남기지 않는다
# (docs/service.md "세션 id가 곧 열쇠다"). 상담에서 일어난 일은 DB의 감사 로그에 남고, 서버의 시작·경고·오류
# 로그는 그대로 나온다.
CMD ["/app/.venv/bin/uvicorn", "support_agent.service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
