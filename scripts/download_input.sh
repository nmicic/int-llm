#!/usr/bin/env bash
# Download the small names dataset used by Karpathy's microgpt/makemore examples.
# The generated input.txt is intentionally not shipped in the public repo.
set -euo pipefail

OUT="${1:-input.txt}"
URL="${MICROGPT_INPUT_URL:-https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt}"

if [[ -s "$OUT" ]]; then
  echo "$OUT already exists; leaving it unchanged."
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
TMP="${OUT}.tmp.$$"
trap 'rm -f "$TMP"' EXIT

if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$URL" -o "$TMP"
elif command -v python3 >/dev/null 2>&1; then
  python3 - "$URL" "$TMP" <<'PY'
import sys
import urllib.request

url, out = sys.argv[1], sys.argv[2]
urllib.request.urlretrieve(url, out)
PY
else
  echo "error: need curl or python3 to download $URL" >&2
  exit 1
fi

if [[ ! -s "$TMP" ]]; then
  echo "error: downloaded dataset is empty: $URL" >&2
  exit 1
fi

mv "$TMP" "$OUT"
trap - EXIT

echo "Downloaded $OUT from $URL"
