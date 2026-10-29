"""배포 설정(CI 워크플로, Dockerfile, compose.yaml)과 실행 문서가 지키기로 한 보안·운영 약속을 확인한다."""

# [테스트 묶음] 돌고 있는 서비스가 아니라 저장소에 있는 배포 설정의 글을 읽어 확인한다: GitHub Actions 작업의
# 토큰 권한이 읽기 전용이고 모든 액션이 커밋 SHA로 고정됐는지, Docker 이미지가 uv 내려받기 캐시를 이미지
# 레이어에 남기지 않고 uvicorn 접근 로그를 끄는지, compose의 앱 컨테이너가 권한을 줄여 뜨는지, 문서에 적힌
# 실제 서비스 실행 명령도 접근 로그를 끄는지.
# [왜 중요한가] 고객 쪽 API는 세션 id를 경로에 넣고(/api/sessions/{id}/…) 그 id가 고객의 유일한 열쇠다. 접근
# 로그에 경로가 남으면 로그를 읽을 수 있는 누구나 그 상담을 이어 갈 수 있다(docs/service.md "세션 id가 곧
# 열쇠다"). 태그로 부른 액션은 누가 태그를 다른 커밋으로 옮기면 그 코드가 CI에서 돈다. 이런 설정은 한 줄을
# 지워도 기능 테스트가 하나도 깨지지 않으므로 글자 그대로 고정해 둔다.
# [방법] YAML은 PyYAML로, Dockerfile과 문서는 줄 단위로 읽는다. Docker·GitHub·네트워크에는 닿지 않는다.
# 이미지를 실제로 빌드해 컨테이너를 띄워 보는 확인은 이 테스트의 몫이 아니다.

from __future__ import annotations

# json: Dockerfile의 CMD는 JSON 배열(exec 형식)이라 그대로 읽는다. re: `uses:` 줄과 커밋 SHA 모양을 본다.
# yaml: 워크플로와 compose.yaml을 읽는다(tests/test_service.py도 compose.yaml을 이렇게 읽는다).
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# 워크플로의 `uses: 소유자/저장소@참조  # 버전` 한 줄. 참조는 40자리 16진수 커밋 SHA여야 하고, 뒤 주석에
# 사람이 읽을 수 있는 버전(v7.0.1 같은)이 있어야 한다. SHA만 있으면 무엇을 고정했는지 알 수 없어 올릴 때
# 실수한다.
_USES = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<action>[^\s@]+)@(?P<ref>\S+)(?:\s+#\s*(?P<version>\S+))?\s*$")


# 모든 워크플로: 맨 위 permissions가 contents: read 하나뿐이고 작업(job) 단위로 다시 넓히지 않는다. 모든
# 액션은 커밋 SHA + 버전 주석으로 부르고, checkout은 토큰을 .git/config에 남기지 않는다(persist-credentials:
# false). 이 작업의 단계(uv sync, ruff, pytest)는 저장소에 쓰지도, 원격에 git으로 닿지도 않는다.
def test_ci_token_is_read_only_and_every_action_is_pinned_to_a_commit():
    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows  # 비어 있으면 아래 반복이 하나도 돌지 않고 통과해 버린다
    for path in workflows:
        text = path.read_text(encoding="utf-8")
        workflow = yaml.safe_load(text)
        assert workflow.get("permissions") == {"contents": "read"}, path.name
        for name, job in workflow["jobs"].items():
            assert "permissions" not in job, (path.name, name)
            for step in job.get("steps", []):  # 재사용 워크플로를 부르는 작업에는 steps가 없다
                if step.get("uses", "").startswith("actions/checkout@"):
                    assert step.get("with", {}).get("persist-credentials") is False, (path.name, name)
        uses = [line for line in text.splitlines() if re.match(r"^\s*(?:-\s*)?uses:", line)]
        assert uses, path.name
        for line in uses:
            found = _USES.match(line)
            assert found, line
            assert re.fullmatch(r"[0-9a-f]{40}", found["ref"]), line
            assert found["version"] and re.fullmatch(r"v\d+\.\d+\.\d+", found["version"]), line


# Dockerfile을 지시 단위로 나눈다: 빈 줄과 주석 줄은 빼고, 줄 끝 `\`로 이어진 줄은 한 지시로 붙인다. 첫 줄의
# 파서 지시(`# syntax=…`)도 주석 모양이라 여기서는 빠지므로 테스트가 따로 본다.
def _dockerfile_instructions(text: str) -> list[str]:
    instructions, current = [], ""
    for raw in text.splitlines():
        line = raw.strip()
        if not current and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            current += line[:-1].strip() + " "
            continue
        instructions.append(current + line)
        current = ""
    return instructions


# 이미지: 두 `uv sync`가 uv 캐시를 BuildKit 캐시 마운트에 두어 내려받은 휠이 이미지 레이어에 두 벌로 남지
# 않는다. 캐시 마운트는 이미지와 다른 파일 시스템이라 하드 링크를 못 쓰므로 UV_LINK_MODE=copy가 함께 있어야
# 한다. `RUN --mount`를 아는 Dockerfile 문법은 첫 줄의 syntax 지시가 고른다. 서버는 접근 로그를 끄고 뜬다.
def test_the_image_keeps_the_uv_cache_out_of_its_layers_and_logs_no_request_paths():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "# syntax=docker/dockerfile:1"
    instructions = _dockerfile_instructions(text)
    syncs = [line for line in instructions if line.startswith("RUN ") and "uv sync" in line]
    assert syncs
    for run in syncs:
        assert "--mount=type=cache,target=/root/.cache/uv" in run, run
    assert any(line.startswith("ENV ") and "UV_LINK_MODE=copy" in line for line in instructions)
    (cmd,) = [line for line in instructions if line.startswith("CMD ")]
    args = json.loads(cmd.removeprefix("CMD "))
    assert args[:2] == ["/app/.venv/bin/uvicorn", "support_agent.service.app:create_app"]
    assert "--no-access-log" in args


# compose의 앱 컨테이너: 모든 Linux capability를 빼고(비루트 uid 10001이 8000번에서 듣는 데 필요한 것이 없다),
# setuid 실행 파일로 권한을 얻지 못하게 하고, 루트 파일 시스템은 읽기 전용(쓸 곳은 tmpfs /tmp뿐), 메모리는
# 2 GB까지. DB 컨테이너는 postgres 이미지의 시작 스크립트가 권한을 바꾸므로 그대로 둔다.
def test_compose_runs_the_app_with_fewer_privileges():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    app = compose["services"]["app"]
    assert "no-new-privileges:true" in app.get("security_opt", [])
    assert app.get("cap_drop") == ["ALL"]
    assert "cap_add" not in app
    assert app.get("read_only") is True
    assert "/tmp" in app.get("tmpfs", [])
    assert app.get("deploy", {}).get("resources", {}).get("limits", {}).get("memory") == "2g"


# 문서(README.md, docs/*.md)에 적힌, 실제 고객을 받는 서비스(service.app:create_app)를 띄우는 uvicorn 명령은
# 모두 접근 로그를 끈다. 모델 없는 오프라인 앱(service.offline)은 시험용이라 대상이 아니다: 보안 스캔이 그
# 접근 로그로 요청 수와 경로별 응답 코드를 센다(docs/service.md "모델 없이 띄우기, 보안 스캔").
def test_documented_service_commands_turn_off_the_access_log():
    docs = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    commands = [
        (doc.name, line)
        for doc in docs
        for line in doc.read_text(encoding="utf-8").splitlines()
        if "uvicorn support_agent.service.app:create_app" in line
    ]
    assert len({name for name, _ in commands}) >= 3  # README, service.md, voice.md
    for name, line in commands:
        assert "--no-access-log" in line, (name, line)
