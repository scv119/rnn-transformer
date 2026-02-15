#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

CONFIG_PATH="configs/recurrent_shared_moe_40e_top2_auxsched_v2_halfparams_wikitext103.json" \
LOG_FILE="logs/recurrent_shared_moe_40e_top2_auxsched_v2_halfparams_live.log" \
WANDB_NAME="recurrent-shared-moe-40e-top2-auxsched-v2-halfparams-wikitext103" \
./scripts/run_300m_resilient.sh
