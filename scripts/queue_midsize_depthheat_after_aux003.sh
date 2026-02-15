#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

echo "[$(date '+%F %T')] queue watcher started" >> logs/midsize_depthheat_queue.log

while pgrep -f "recurrent_shared_moe_40e_top2_auxsched_v2_halfparams_aux003_short_wikitext103.json" >/dev/null 2>&1; do
  sleep 60
done

echo "[$(date '+%F %T')] detected aux003 run finished; launching midsize depthheat ablation" >> logs/midsize_depthheat_queue.log
nohup ./scripts/run_recurrent_shared_moe_40e_top2_auxsched_v2_midsize_depthheat_short.sh \
  > logs/recurrent_shared_moe_40e_top2_auxsched_v2_midsize_depthheat_short_launcher.log 2>&1 &
echo "[$(date '+%F %T')] launched" >> logs/midsize_depthheat_queue.log
