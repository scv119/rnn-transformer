#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

LOG_FILE="logs/train_live.log"
MAX_RESTARTS="${MAX_RESTARTS:-50}"
RETRY_DELAY_SEC="${RETRY_DELAY_SEC:-20}"
RESTART_COUNT=0

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting/resuming run (attempt $((RESTART_COUNT + 1)))" | tee -a "$LOG_FILE"

  set +e
  WANDB_MODE="${WANDB_MODE:-offline}" \
  WANDB_PROJECT="${WANDB_PROJECT:-rnn-transformer}" \
  WANDB_NAME="${WANDB_NAME:-baseline-300m-wikitext103}" \
  ./.venv/bin/python -u train_baseline.py \
    --config configs/baseline_300m_wikitext103.json \
    --auto_resume 2>&1 | tee -a "$LOG_FILE"
  RC=${PIPESTATUS[0]}
  set -e

  if [[ $RC -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] training finished successfully" | tee -a "$LOG_FILE"
    exit 0
  fi

  RESTART_COUNT=$((RESTART_COUNT + 1))
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] training exited with code $RC" | tee -a "$LOG_FILE"

  if [[ $RESTART_COUNT -ge $MAX_RESTARTS ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] hit MAX_RESTARTS=$MAX_RESTARTS; exiting" | tee -a "$LOG_FILE"
    exit "$RC"
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] retrying in ${RETRY_DELAY_SEC}s" | tee -a "$LOG_FILE"
  sleep "$RETRY_DELAY_SEC"
done
