#!/usr/bin/env bash
# Run zap/automation.yaml with a ZAP release (cross-platform package, Java 17 or later) against the offline
# service (docs/service.md, "모델 없이 띄우기, 보안 스캔").
#
#   ZAP_DIR=<folder with zap-2.17.0.jar> ADMIN_TOKEN=<token> SCAN_SESSION_ID=<id> SCAN_APPROVAL_ID=<n> \
#     zap/run_zap.sh <name>
#
# The service must already be running on TARGET (default http://127.0.0.1:8062) with that admin token, and the
# session must have a pending approval. The reports <name>.json and <name>.md, the ZAP log and a fresh ZAP
# home folder go to work/zap/<name>/. Output goes to files only, never to a pipe.
set -euo pipefail

name=${1:?usage: zap/run_zap.sh <name>}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${ZAP_DIR:?set ZAP_DIR to the folder that holds zap-<version>.jar}"
: "${ADMIN_TOKEN:?set ADMIN_TOKEN to the token the service was started with}"
: "${SCAN_SESSION_ID:?set SCAN_SESSION_ID to a session of the scanned database}"
: "${SCAN_APPROVAL_ID:?set SCAN_APPROVAL_ID to the number of a pending approval of that session}"

native() { # the absolute path in a form Java understands (the Windows form under Git Bash). ZAP resolves a
  # relative path against the plan's folder (zap/), not the working directory.
  local abs
  abs=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
  if command -v cygpath > /dev/null; then cygpath -m "$abs"; else printf '%s\n' "$abs"; fi
}

out=$root/work/zap/$name
if [ -e "$out" ]; then
  echo "$out exists: pick another name" >&2
  exit 1
fi
jars=("$ZAP_DIR"/zap-*.jar)
[ -f "${jars[0]}" ] || { echo "no zap-*.jar in $ZAP_DIR" >&2; exit 1; }
mkdir -p "$out"

export TARGET=${TARGET:-http://127.0.0.1:8062}
host=${TARGET#*://}
# ZAP adds this header to every request for the target host (see zap/automation.yaml).
export ZAP_AUTH_HEADER=X-Admin-Token ZAP_AUTH_HEADER_VALUE=$ADMIN_TOKEN ZAP_AUTH_HEADER_SITE=${host%%[:/]*}
export SCAN_SESSION_ID SCAN_APPROVAL_ID REPORT_DIR REPORT_NAME=$name
REPORT_DIR=$(native "$out")

status=0
java -Xmx2g -jar "$(native "${jars[0]}")" -cmd -notel -dir "$(native "$out/home")" \
  -config start.checkForUpdates=false -autorun "$(native "$root/zap/automation.yaml")" \
  > "$out/zap.log" 2>&1 || status=$?
echo "ZAP exit status $status, reports and log in $out"
exit "$status"
