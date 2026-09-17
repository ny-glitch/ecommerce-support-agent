#!/usr/bin/env bash
set -euo pipefail

base_url="${BASE_URL:-http://127.0.0.1:8000}"
python_bin="${PYTHON:-python3}"
demo_tmp="$(mktemp -d)"

cleanup() {
  rm -f "$demo_tmp/first.sse" "$demo_tmp/second.sse"
  rmdir "$demo_tmp"
}
trap cleanup EXIT

validate_sse() {
  "$python_bin" - "$1" "${2:-}" <<'PY'
import json
import sys
import uuid
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8").replace("\r\n", "\n")
events = []
for block in text.split("\n\n"):
    name = None
    data = []
    for line in block.splitlines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if name and data:
        events.append((name, json.loads("\n".join(data))))

error = next((data for name, data in events if name == "error"), None)
if error is not None:
    code = error.get("code", "UNKNOWN") if isinstance(error, dict) else "UNKNOWN"
    raise SystemExit(f"聊天流失败：{code}")
meta = next((data for name, data in events if name == "meta"), None)
done = next((data for name, data in events if name == "done"), None)
if not isinstance(meta, dict) or done is None:
    raise SystemExit("聊天流不完整：缺少 meta 或 done")
session_id = str(uuid.UUID(meta.get("session_id", "")))
if not isinstance(done, dict) or done.get("session_id") != session_id:
    raise SystemExit("聊天流的 done session_id 不一致")
expected = sys.argv[2]
if expected and session_id != expected:
    raise SystemExit("第二轮未复用原 session_id")
print(session_id)
PY
}

echo "第一轮聊天："
curl --noproxy '*' --fail-with-body --no-buffer --silent --show-error \
  -X POST "$base_url/api/chat" \
  -H 'Content-Type: application/json' \
  -d '{"message":"我叫小林，刚买的耳机有杂音"}' \
  | tee "$demo_tmp/first.sse"
session_id="$(validate_sse "$demo_tmp/first.sse")"

echo "第二轮聊天（复用 session_id）："
curl --noproxy '*' --fail-with-body --no-buffer --silent --show-error \
  -X POST "$base_url/api/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$session_id\",\"message\":\"我叫什么，商品出了什么问题？\"}" \
  | tee "$demo_tmp/second.sse"
validate_sse "$demo_tmp/second.sse" "$session_id" >/dev/null

echo "售后提取："
curl --noproxy '*' --fail-with-body --silent --show-error \
  -X POST "$base_url/api/after-sales/extract" \
  -H 'Content-Type: application/json' \
  -d '{"description":"订单 A123 到货破损，希望换一个新的"}'
echo
