#!/usr/bin/env bash
# ZAP 배포판(크로스 플랫폼 패키지, Java 17 이상)으로 zap/automation.yaml을 오프라인 서비스에 돌린다
# (docs/service.md "모델 없이 띄우기, 보안 스캔").
#
#   ZAP_DIR=<zap-2.17.0.jar가 있는 폴더> ADMIN_TOKEN=<토큰> SCAN_SESSION_ID=<id> SCAN_APPROVAL_ID=<n> \
#     zap/run_zap.sh <name>
#
# 서비스는 그 관리자 토큰으로 TARGET(기본 http://127.0.0.1:8062)에서 이미 돌고 있어야 하고, 그 세션에는 승인
# 대기 중인 건이 있어야 한다. 보고서 <name>.json과 <name>.md, ZAP 로그, 새 ZAP 홈 폴더는 work/zap/<name>/에
# 생긴다. 출력은 파일로만 보내고 파이프로는 절대 보내지 않는다.
# [파일 역할] OWASP ZAP 2.17.0의 자동화 계획(zap/automation.yaml)을 오프라인 서비스(모델 자리에 규칙 대역을
# 넣은 service/offline.py)에 돌리는 실행 도우미다. 결과 보고서와 로그를 work/zap/<이름>/에 남긴다.
# [왜 필요한가] 2026-10-05 HawkScan(StackHawk) 체험이 끝나 무료 ZAP으로 같은 스캔(API 설명, 스파이더, 패시브
# 스캔, 기본 정책의 액티브 스캔)을 하게 바꿨다. 두 도구는 규칙 목록이 달라 결과를 직접 비교하지 않고 ZAP
# 기록끼리만 비교한다 (docs/service.md "모델 없이 띄우기, 보안 스캔").
# [흐름] 인자·환경 변수 확인 → 결과 폴더 만들기(이미 있으면 중단) → 관리자 토큰을 ZAP 인증 헤더 환경 변수로
# 넘기기 → java로 ZAP을 띄워 계획 실행 → ZAP의 종료 코드를 그대로 돌려줌.
# [보안] 토큰 같은 비밀값은 계획 파일에 쓰지 않고 실행할 때 환경 변수로만 넘긴다.
set -euo pipefail

# 첫 인자(스캔 이름)는 필수다. 결과 폴더 이름이 된다. root는 이 스크립트 위치에서 구한 저장소 루트라, 어느
# 폴더에서 실행해도 경로가 같다.
name=${1:?usage: zap/run_zap.sh <name>}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# 필요한 환경 변수가 없으면 무엇을 넣어야 하는지 알려 주고 멈춘다. SCAN_SESSION_ID와 SCAN_APPROVAL_ID는 스캔
# 전에 만든 실제 세션과 승인 대기 번호다. 지어낸 번호로는 '없는 세션'(404)에서 끝나 처리기까지 공격이 닿지
# 않으므로, 계획의 requestor 작업이 이 번호로 요청을 한 번씩 보내 스캔 대상에 넣는다.
: "${ZAP_DIR:?set ZAP_DIR to the folder that holds zap-<version>.jar}"
: "${ADMIN_TOKEN:?set ADMIN_TOKEN to the token the service was started with}"
: "${SCAN_SESSION_ID:?set SCAN_SESSION_ID to a session of the scanned database}"
: "${SCAN_APPROVAL_ID:?set SCAN_APPROVAL_ID to the number of a pending approval of that session}"

# 경로를 Java가 읽는 절대 경로로 바꾼다. Git Bash에서는 cygpath -m으로 C:/... 형태(Windows 경로, 슬래시)로
# 바꾸고, 다른 환경에서는 절대 경로 그대로 쓴다. ZAP은 상대 경로를 작업 폴더가 아니라 계획 파일 폴더(zap/)
# 기준으로 풀기 때문에 절대 경로로 넘긴다.
native() {
  local abs
  abs=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
  if command -v cygpath > /dev/null; then cygpath -m "$abs"; else printf '%s\n' "$abs"; fi
}

# 같은 이름의 결과 폴더가 있으면 덮어쓰지 않고 멈춘다. 이전 스캔 기록(docs/service.md 스캔 표의 비교 기준)을
# 지우지 않으려는 것이다. 그다음 ZAP_DIR에서 zap-*.jar를 찾는다.
out=$root/work/zap/$name
if [ -e "$out" ]; then
  echo "$out exists: pick another name" >&2
  exit 1
fi
jars=("$ZAP_DIR"/zap-*.jar)
[ -f "${jars[0]}" ] || { echo "no zap-*.jar in $ZAP_DIR" >&2; exit 1; }
mkdir -p "$out"

# 스캔 대상 주소(기본 127.0.0.1:8062의 로컬 서비스). host는 'http://' 같은 앞부분을 뗀 것이고, 아래에서 포트와
# 경로까지 떼어 호스트 이름만 쓴다.
export TARGET=${TARGET:-http://127.0.0.1:8062}
host=${TARGET#*://}
# ZAP_AUTH_HEADER* 환경 변수를 주면 ZAP이 대상 호스트로 가는 모든 요청에 X-Admin-Token 헤더를 붙인다
# (zap/automation.yaml 참고). 관리자 API까지 인증된 상태로 스캔하려는 것이고, HawkScan의 토큰 인증 자리다.
# 다음 줄은 계획 파일이 ${…}로 읽는 변수(세션·승인 번호, 보고서 폴더와 이름)를 내보낸다.
export ZAP_AUTH_HEADER=X-Admin-Token ZAP_AUTH_HEADER_VALUE=$ADMIN_TOKEN ZAP_AUTH_HEADER_SITE=${host%%[:/]*}
export SCAN_SESSION_ID SCAN_APPROVAL_ID REPORT_DIR REPORT_NAME=$name
REPORT_DIR=$(native "$out")

# ZAP 실행: 힙 2 GB, 화면 없는 명령 모드(-cmd), 원격 측정 끔(-notel), 스캔마다 새 ZAP 홈 폴더(-dir), 업데이트
# 확인 끔, 계획 자동 실행(-autorun). 출력은 모두 zap.log 파일로 보낸다.
# || status=$? 로 실패 코드를 받아 두어, set -e로 바로 끝나지 않고 결과 위치를 알린 뒤 같은 코드로 끝난다.
# ZAP_PORT(선택): ZAP 자신의 프록시 포트(기본 8080)를 바꾼다. 같은 PC에서 다른 ZAP 스캔이 8080을 쥐고 있으면 ZAP이
# "Address already in use"로 바로 끝나기 때문이다(2026-10-15 11차 첫 시도). 스캔 대상(TARGET)과는 상관없다.
port_args=()
if [ -n "${ZAP_PORT:-}" ]; then port_args=(-port "$ZAP_PORT"); fi
status=0
java -Xmx2g -jar "$(native "${jars[0]}")" -cmd -notel "${port_args[@]}" -dir "$(native "$out/home")" \
  -config start.checkForUpdates=false -autorun "$(native "$root/zap/automation.yaml")" \
  > "$out/zap.log" 2>&1 || status=$?
echo "ZAP exit status $status, reports and log in $out"
exit "$status"
