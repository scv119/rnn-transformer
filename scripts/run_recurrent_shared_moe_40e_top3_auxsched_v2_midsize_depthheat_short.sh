#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

CONFIG_PATH="configs/recurrent_shared_moe_40e_top3_auxsched_v2_midsize_depthheat_short_wikitext103.json" \
LOG_FILE="logs/recurrent_shared_moe_40e_top3_auxsched_v2_midsize_depthheat_short_live.log" \
WANDB_NAME="recurrent-shared-moe-40e-top3-auxsched-v2-midsize-depthheat-short" \
./scripts/run_300m_resilient.sh
