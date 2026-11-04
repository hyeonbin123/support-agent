"""배포 설정(CI 워크플로, Dockerfile, compose.yaml)과 실행 문서가 지키기로 한 보안·운영 약속을 확인한다."""

# [테스트 묶음] 돌고 있는 서비스가 아니라 저장소에 있는 배포 설정의 글을 읽어 확인한다: GitHub Actions 작업의
# 토큰 권한이 읽기 전용이고 모든 액션이 커밋 SHA로 고정됐는지, 작업끼리 setup-uv 캐시 키를 나눠 쓰지
# 않는지(2026-11-05부터), Docker 이미지가 uv 내려받기 캐시를 이미지 레이어에 남기지 않고 uvicorn 접근 로그를
# 끄는지, compose의 앱 컨테이너가 권한을 줄여 뜨는지, 문서에 적힌 실제 서비스 실행 명령도 접근 로그를 끄는지,
# 문서의 설정 목록이 서비스 설정(Settings)의 필드를 빠짐없이 적는지.
# [왜 중요한가] 고객 쪽 API는 세션 id를 경로에 넣고(/api/sessions/{id}/…) 그 id가 고객의 유일한 열쇠다. 접근
# 로그에 경로가 남으면 로그를 읽을 수 있는 누구나 그 상담을 이어 갈 수 있다(docs/service.md "세션 id가 곧
# 열쇠다"). 태그로 부른 액션은 누가 태그를 다른 커밋으로 옮기면 그 코드가 CI에서 돈다. 이런 설정은 한 줄을
# 지워도 기능 테스트가 하나도 깨지지 않으므로 글자 그대로 고정해 둔다. 설정 목록도 같다: 설정을 더하면서
# 목록을 고치지 않아도 아무 테스트가 실패하지 않는다.
# [방법] YAML은 PyYAML로, Dockerfile과 문서는 줄 단위로 읽는다. Docker·GitHub·네트워크에는 닿지 않는다.
# 설정 목록은 서비스 설정 클래스의 필드 이름과 맞춰 본다(설정 값을 만들지는 않는다). 이미지를 실제로 빌드해
# 컨테이너를 띄워 보는 확인은 이 테스트의 몫이 아니다.

from __future__ import annotations

# json: Dockerfile의 CMD는 JSON 배열(exec 형식)이라 그대로 읽는다. re: `uses:` 줄과 커밋 SHA 모양, 문서의
# `대문자_이름`을 본다. yaml: 워크플로와 compose.yaml을 읽는다(tests/test_service.py도 compose.yaml을 이렇게
# 읽는다). PREFIX·Settings: 서비스 설정의 환경 변수 접두사(SUPPORT_AGENT_)와 필드 목록(model_fields).
import json
import re
from pathlib import Path

import yaml

from support_agent.service.settings import PREFIX, Settings

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


# CI의 audit 작업(2026-11-01부터): 잠금 파일의 기본 그룹(본체 + dev. CI와 서비스 Docker 이미지가 설치하는
# 것)을 판을 고정한 pip-audit으로 공개 권고와 대조한다. 설치 없이(--disable-pip, --no-deps) 잠금 그대로의
# 판만 보고(--frozen), 대조하지 못한 패키지가 있으면 실패한다(--strict). 단계 셸은 bash라 파이프 앞쪽의
# 명령(uv export, grep)이 실패해도 단계가 실패한다(GitHub Actions의 bash는 pipefail을 켠다). 새 권고가 공개된
# 날 코드와 상관없는 push가 막히지 않게 이 작업만 실패로 남는다(continue-on-error). 음성 그룹은 대조하지
# 않는다: MeloTTS가 묶은 판의 권고가 남아 있어 늘 실패한다(docs/voice.md "정한 것과 이유"의 2026-11-01
# 보탬). 이 작업이 빠지면 기본 그룹에 새 권고가 생겨도 아무 검사도 실패하지 않으므로 글자 그대로 고정한다.
def test_ci_audits_the_locked_default_group_with_a_pinned_pip_audit():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    audit = workflow["jobs"]["audit"]
    assert audit.get("continue-on-error") is True
    (step,) = [s for s in audit["steps"] if "pip-audit" in s.get("run", "")]
    assert step.get("shell") == "bash"
    run = step["run"]
    assert "uv export --frozen" in run and "--no-emit-project" in run
    assert "--all-groups" not in run and "--group" not in run  # 기본 그룹만
    assert re.search(r"uvx pip-audit==\d+\.\d+\.\d+ ", run)
    for flag in ("--no-deps", "--disable-pip", "--strict"):
        assert flag in run, flag


# CI 작업끼리 setup-uv 캐시를 같은 키로 나눠 쓰지 않는다. setup-uv의 캐시 키는 러너 OS·해석기 판·의존성 파일
# (pyproject.toml·uv.lock 등) 해시 뒤에 cache-suffix를 붙여 만든다(고정한 setup-uv v10.1.0의
# src/cache/restore-cache.ts computeKeys). 그래서 같은 러너 이미지에서 도는 두 작업이 둘 다 캐시를 켜고
# cache-suffix가 같으면 키가 같다. 먼저 끝난 작업이 그 키로 캐시를 저장하면 다른 작업은 그 캐시에 적중하는데,
# setup-uv는 정확히 같은 키에 적중한 작업의 캐시를 저장하지 않으므로(src/save-cache.ts) 적중한 작업은 실행마다
# 자기 의존성을 다시 내려받는다. 2026-11-01~11-05 CI에서 실제로 그랬다: 몇 초 만에 끝나는 audit 작업이
# pip-audit만 든 캐시를 먼저 저장했고, test 작업은 그 캐시에 적중한 뒤 실행마다 의존성 대부분(11-05 실행에서
# 설치한 48개 가운데 44개)을 다시 받았다.
# enable-cache를 적지 않으면 기본값 auto이고 GitHub 호스트 러너에서는 켜진다. YAML에서 false는 불리언으로
# 읽히지만 따옴표로 쓴 "false"도 setup-uv에는 같은 뜻이라 글자로 바꿔 비교한다.
def test_ci_jobs_do_not_share_a_setup_uv_cache_key():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    cached: dict[str, list[str]] = {}  # cache-suffix → 캐시를 켠 작업 이름들
    for name, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            if not step.get("uses", "").startswith("astral-sh/setup-uv@"):
                continue
            options = step.get("with") or {}
            if str(options.get("enable-cache", "auto")).lower() != "false":
                cached.setdefault(str(options.get("cache-suffix", "")), []).append(name)
    # test 작업은 캐시를 쓴다. 비어 있으면 setup-uv 단계를 찾지 못한 것이라 아래 검사가 아무것도 보지 않는다.
    assert cached
    shared = {suffix: jobs for suffix, jobs in cached.items() if len(jobs) > 1}
    assert not shared, f"같은 setup-uv 캐시 키를 쓰는 작업 {shared}"


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


# docs/service.md "실행"의 설정 목록: "설정은 환경 변수로 한다"로 시작하는 한 줄(문단)의 목록 부분에
# `대문자_이름`으로 적은 이름이 서비스 설정(Settings)의 필드와 같아야 한다. 빠진 설정도, 설정에 없는 이름도,
# 같은 이름을 두 번 적은 것도 실패다. 환경 변수는 접두사 SUPPORT_AGENT_ 뒤에 필드 이름을 대문자로 붙인
# 것이고(Settings.from_env), 목록은 접두사를 뗀 이름으로 적는다. 줄 머리의 괄호에는 그 접두사가 적혀 있어야
# 한다(코드의 PREFIX와 같은 글자).
# 목록 부분은 머리의 괄호를 닫는 `): ` 뒤부터 첫 문장 끝(`. `) 앞까지이고, 그 안은 `, `로 나눈 `이름`뿐이어야
# 한다. 같은 줄의 뒤 문장(범위 검사 안내, 날짜 붙은 보탬)에 적은 이름은 세지 않는다. 2026-11-05 보탬 문장이
# `LOAD_SEED`·`EXPOSE_OPENAPI`를 다시 적기 때문에, 줄 전체에서 세던 앞 판은 목록 부분에서 그 이름이 빠져도
# 통과했다. 그런 줄은 꼭 하나여야 한다(둘이면 어느 쪽이 목록인지 모른다). 목록 모양이 바뀌면(두 줄로 접음,
# 이름 사이에 설명 글) 이름을 잘못 세는 대신 실패하므로, 그때는 이 테스트를 새 모양에 맞춘다. 설정을 더하면서
# 목록을 고치지 않으면 문서만 읽고는 그 설정이 있는지 알 수 없다(2026-11-05까지 LOAD_SEED·EXPOSE_OPENAPI가
# 빠져 있었다).
def test_the_documented_settings_list_names_every_service_setting():
    lines = (ROOT / "docs" / "service.md").read_text(encoding="utf-8").splitlines()
    (paragraph,) = [line for line in lines if line.startswith("설정은 환경 변수로 한다")]
    head, opened, rest = paragraph.partition("): ")
    listed, ended, _ = rest.partition(". ")
    assert opened and ended, f"목록 부분(`): ` 뒤, 첫 `. ` 앞)을 찾지 못함: {paragraph[:80]}"
    assert f"`{PREFIX}`" in head, head
    items = listed.split(", ")
    odd = [item for item in items if not re.fullmatch(r"`[A-Z][A-Z0-9_]*`", item)]
    assert not odd, f"목록에 `이름`이 아닌 항목 {odd}"
    names = [item.strip("`") for item in items]
    twice = sorted({name for name in names if names.count(name) > 1})
    assert not twice, f"두 번 적은 이름 {twice}"
    documented = set(names)
    fields = {name.upper() for name in Settings.model_fields}
    missing, unknown = sorted(fields - documented), sorted(documented - fields)
    assert not missing and not unknown, f"목록에 없는 설정 {missing}, 설정에 없는 이름 {unknown}"
