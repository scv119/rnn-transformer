#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 train_baseline.py --config configs/baseline_300m_wikitext103.json
