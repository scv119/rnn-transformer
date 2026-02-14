# rnn-transformer baseline

This repo now contains a Tier-A dense baseline aligned with the research plan PDF:
- Model scale: ~300M parameters (`20` layers, `1024` hidden, `16` heads)
- Task: decoder-only language modeling
- Dataset: `wikitext-103-raw-v1`
- Optimizer: `AdamW`
- LR schedule: cosine decay + warmup
- Context length: `1024`

## Files
- `train_baseline.py`: training entrypoint
- `configs/baseline_300m_wikitext103.json`: baseline config
- `scripts/run_300m_baseline.sh`: run script
- `scripts/setup_python_env.sh`: host setup helper

## Setup

```bash
cd /home/chenshen/rnn-transformer
./scripts/setup_python_env.sh
```

## Train

```bash
cd /home/chenshen/rnn-transformer
source .venv/bin/activate
wandb login
export WANDB_PROJECT=rnn-transformer
export WANDB_NAME=baseline-300m-wikitext103
./scripts/run_300m_baseline.sh
```

## Quick smoke test

```bash
cd /home/chenshen/rnn-transformer
source .venv/bin/activate
python train_baseline.py \
  --config configs/baseline_300m_wikitext103.json \
  --max_steps 20 \
  --output_dir runs/smoke_300m
```

## Recurrent Baseline Equivalence

Run the strict equivalence gate (stacked vs recurrent step-indexed):

```bash
cd /home/chenshen/rnn-transformer
source .venv/bin/activate
python scripts/check_recurrent_equivalence.py
```

Success criteria (from plan):
- `max_abs_diff_logits < 1e-5`
- `grad_cosine > 0.999`
