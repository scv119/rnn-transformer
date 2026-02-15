#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

CONFIG_PATH="configs/recurrent_shared_moe_40e_top3_dff2184_matchactive_short_wikitext103.json" \
LOG_FILE="logs/recurrent_shared_moe_40e_top3_dff2184_matchactive_short_live.log" \
WANDB_NAME="recurrent-shared-moe-40e-top3-dff2184-matchactive-short" \
./scripts/run_300m_resilient.sh
