#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

CONFIG_PATH="configs/recurrent_shared_moe_300m_wikitext103.json" \
LOG_FILE="logs/recurrent_shared_moe_40e_top2_live.log" \
WANDB_NAME="recurrent-shared-moe-40e-top2-ff2048-wikitext103" \
./scripts/run_300m_resilient.sh
