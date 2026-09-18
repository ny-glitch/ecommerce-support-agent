#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"
PYTHON="${PYTHON:-python3}"
DEMO_TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$DEMO_TMP_DIR"' EXIT

questions=(
  "订单 1001 的物流到哪了"
  "退货政策是什么"
  "邮费是多少"
)

for index in "${!questions[@]}"; do
  question="${questions[$index]}"
  response_path="$DEMO_TMP_DIR/turn-$index.sse"
  payload="$("$PYTHON" -c 'import json,sys; print(json.dumps({"message": sys.argv[1]}, ensure_ascii=False))' "$question")"

  printf '\n问题：%s\n' "$question"
  curl --noproxy '*' -N --fail-with-body --silent --show-error --max-time 70 \
    -X POST "$BASE_URL/api/chat" \
    -H 'Content-Type: application/json' \
    -d "$payload" | tee "$response_path"

  "$PYTHON" - "$response_path" <<'PY'
import json
from pathlib import Path
import sys

events = []
name = None
data_lines = []

def flush():
    global name, data_lines
    if name is not None and data_lines:
        events.append((name, json.loads("\n".join(data_lines))))
    name = None
    data_lines = []

for line in Path(sys.argv[1]).read_text(encoding="utf-8").replace("\r\n", "\n").splitlines():
    if not line:
        flush()
    elif line.startswith("event:"):
        name = line[6:].strip()
    elif line.startswith("data:"):
        data_lines.append(line[5:].lstrip())
flush()

names = [item[0] for item in events]
if "error" in names:
    raise SystemExit("stream returned error event")
if len([name for name in names if name == "done"]) != 1 or names[-1:] != ["done"]:
    raise SystemExit("missing done event")
if names[:1] != ["meta"]:
    raise SystemExit("missing meta event")
meta = events[0][1]
done = events[-1][1]
if not isinstance(meta, dict) or done.get("session_id") != meta.get("session_id"):
    raise SystemExit("session_id mismatch")
PY
done
