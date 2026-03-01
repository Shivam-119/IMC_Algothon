#!/usr/bin/env bash
# run_bot.sh — launch the IMCity trading bot
#
# Usage:
#   1. cp .env.example .env
#   2. Edit .env with your credentials
#   3. bash run_bot.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE="$SCRIPT_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env not found."
  echo "  Run:  cp .env.example .env  then fill in your credentials."
  exit 1
fi

# Load .env (export each non-comment line)
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# Validate required vars
for var in CMI_URL CMI_USER CMI_PASS; do
  val="${!var:-}"
  if [[ -z "$val" || "$val" == REPLACE_WITH* ]]; then
    echo "ERROR: $var is not set (or still has the placeholder value)."
    echo "  Edit .env and fill in your credentials."
    exit 1
  fi
done

echo "=== IMCity Bot ==="
echo "  Exchange : $CMI_URL"
echo "  User     : $CMI_USER"
echo "  RapidAPI : ${RAPIDAPI_KEY:+(set)}"
echo ""

# Activate venv
VENV="$SCRIPT_DIR/venv"
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
fi

# Add algothon-templates to PYTHONPATH so bot_template is importable
export PYTHONPATH="$SCRIPT_DIR/algothon-templates:${PYTHONPATH:-}"

exec python3 "$SCRIPT_DIR/src/bot.py"
