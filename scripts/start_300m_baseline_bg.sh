#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs/baseline_300m_wikitext103
mkdir -p logs

LOG_FILE="logs/baseline_300m_wikitext103.log"
PID_FILE="logs/baseline_300m_wikitext103.pid"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Training already running with PID $OLD_PID"
    exit 0
  fi
fi

nohup env \
  WANDB_MODE=offline \
  WANDB_PROJECT=rnn-transformer \
  WANDB_NAME=baseline-300m-wikitext103 \
  ./.venv/bin/python -u train_baseline.py --config configs/baseline_300m_wikitext103.json > "$LOG_FILE" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
echo "Started PID $(cat "$PID_FILE")"
echo "Log: $LOG_FILE"
