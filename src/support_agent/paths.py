"""Folders the scripts read and write. `outputs/` is git-ignored; `reports/` is committed."""

# [파일 역할] 저장소 안의 주요 폴더 경로를 한곳에 모아 둔 상수 모듈이다.
# [왜 필요한가] 실행기(run.py), 과제 로더(tasks.py), 분석(analyze.py) 등이 각자 경로를 계산하면 실행 위치(현재
# 작업 폴더)에 따라 경로가 달라질 수 있다. 이 파일의 위치를 기준으로 한 번만 계산해 어디서 실행해도 같은
# 폴더를 가리키게 한다.
# [연결] run.py(OUTPUTS, REPORTS, ROOT), tasks.py·analyze.py·confirm_report.py·sim_report.py(TASKS),
# voice_real.py(OUTPUTS, REPORTS)가 가져다 쓴다.

from pathlib import Path

# 저장소 루트. 이 파일은 src/support_agent/paths.py이므로 parents[0]=src/support_agent, parents[1]=src,
# parents[2]=저장소 루트다. run.py는 이 경로에서 git 명령을 돌려 커밋 해시를 manifest에 남긴다.
ROOT = Path(__file__).resolve().parents[2]
# TASKS: 과제 YAML(smoke·dev·test)이 있는 폴더.
# OUTPUTS: 일반 실행 결과가 쌓이는 곳(.gitignore에 들어 있어 커밋하지 않음).
# REPORTS: --official 공식 측정 기록이 남는 곳. 수치를 나중에 다시 계산할 수 있게 커밋한다(README의 구조·다시
# 재는 방법 절).
TASKS = ROOT / "tasks"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
