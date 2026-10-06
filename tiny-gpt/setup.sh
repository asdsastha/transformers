#!/usr/bin/env bash
# tiny-gpt setup: ensure .aiml env, install requirements, and download Tiny Shakespeare into data/.
# Safe to re-run. Run by `setup` from this folder or by hand: bash setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REQ_FILE="requirements.txt"
URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
FILE="data/input.txt"
SIZE=1115394  # bytes; the notebook's Part 1 check expects exactly this

if [[ ! -f "$REQ_FILE" ]]; then
  echo "tiny-gpt: missing $REQ_FILE in $SCRIPT_DIR" >&2
  exit 1
fi

# Determine target .aiml environment:
# - explicit AIML_ENV if provided
# - active .aiml environment if the script is run from one
# - otherwise default to ~/aiml/.aiml
if [[ -n "${AIML_ENV:-}" ]]; then
  ENV_DIR="$AIML_ENV"
elif [[ -n "${VIRTUAL_ENV:-}" && "$(basename "$VIRTUAL_ENV")" == ".aiml" ]]; then
  ENV_DIR="$VIRTUAL_ENV"
else
  ENV_DIR="$HOME/aiml/.aiml"
fi

PYTHON_BIN="$ENV_DIR/bin/python"

if ! command -v uv >/dev/null 2>&1; then
  echo "tiny-gpt: uv is required but not installed or not on PATH" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "tiny-gpt: creating .aiml environment at $ENV_DIR"
  uv venv "$ENV_DIR"
fi

# Install into the .aiml env with uv, keeping torch on the cu132 backend.
echo "tiny-gpt: installing Python dependencies from $REQ_FILE"
uv pip install --python "$PYTHON_BIN" --torch-backend cu132 -r "$REQ_FILE"

if [[ -f "$FILE" && $(stat -c %s "$FILE") -eq $SIZE ]]; then
  echo "tiny-gpt: $FILE already present"
  exit 0
fi

mkdir -p data
curl -fsSL "$URL" -o "$FILE.tmp"
if [[ $(stat -c %s "$FILE.tmp") -ne $SIZE ]]; then
  echo "tiny-gpt: downloaded $FILE has the wrong size, expected $SIZE bytes" >&2
  rm -f "$FILE.tmp"
  exit 1
fi
mv "$FILE.tmp" "$FILE"
echo "tiny-gpt: downloaded Tiny Shakespeare to $FILE"
