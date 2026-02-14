#!/usr/bin/env python3
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rnn_transformer.recurrent_baseline import (  # noqa: E402
    DecoderConfig,
    RecurrentDecoderLM,
    StackedDecoderLM,
    causal_lm_loss,
    gradient_cosine,
    matched_parameter_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check stacked vs recurrent-step-indexed equivalence")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--atol_logits", type=float, default=1e-5)
    parser.add_argument("--min_grad_cos", type=float, default=0.999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = DecoderConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        dropout=0.0,
    )

    stacked = StackedDecoderLM(cfg).to(device)
    recurrent = RecurrentDecoderLM.from_stacked(stacked, shared_across_steps=False).to(device)

    stacked.eval()
    recurrent.eval()

    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    logits_stacked = stacked(input_ids)
    logits_recurrent = recurrent(input_ids)

    max_abs_diff = float((logits_stacked - logits_recurrent).abs().max().item())

    stacked.zero_grad(set_to_none=True)
    recurrent.zero_grad(set_to_none=True)

    loss_stacked = causal_lm_loss(logits_stacked, input_ids)
    loss_recurrent = causal_lm_loss(logits_recurrent, input_ids)

    loss_stacked.backward()
    loss_recurrent.backward()

    grad_cos = gradient_cosine(matched_parameter_pairs(stacked, recurrent))

    print(f"device={device}")
    print(f"max_abs_diff_logits={max_abs_diff:.8e}")
    print(f"grad_cosine={grad_cos:.8f}")
    print(f"loss_stacked={loss_stacked.item():.8f}")
    print(f"loss_recurrent={loss_recurrent.item():.8f}")

    ok_logits = max_abs_diff < args.atol_logits
    ok_grad = grad_cos > args.min_grad_cos

    if not (ok_logits and ok_grad):
        print("EQUIVALENCE_CHECK=FAIL")
        sys.exit(1)

    print("EQUIVALENCE_CHECK=PASS")


if __name__ == "__main__":
    main()
