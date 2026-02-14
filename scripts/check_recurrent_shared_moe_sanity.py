#!/usr/bin/env python3
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rnn_transformer.recurrent_baseline import DecoderConfig, RecurrentDecoderLM
from train_baseline import RecurrentForCausalLM, pick_shared_moe_shape_for_target


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_decoder_cfg(cfg: dict, use_moe: bool) -> DecoderConfig:
    moe = cfg.get("moe", {})
    return DecoderConfig(
        vocab_size=int(cfg["model"]["vocab_size"]),
        max_seq_len=int(cfg["context_length"]),
        d_model=int(cfg["model"]["n_embd"]),
        n_heads=int(cfg["model"]["n_head"]),
        d_ff=int(cfg["model"]["n_inner"]),
        n_layers=int(cfg["model"]["n_layer"]),
        dropout=float(cfg["model"].get("resid_pdrop", 0.0)),
        moe_num_experts=(int(moe.get("num_experts", 0)) if use_moe else 0),
        moe_top_k=(int(moe.get("top_k", 1)) if use_moe else 1),
        moe_d_ff=(int(moe["d_ff"]) if use_moe and moe.get("d_ff") is not None else None),
    )


def main() -> None:
    cfg_path = Path("configs/recurrent_shared_moe_300m_wikitext103.json")
    cfg = load_cfg(cfg_path)

    indexed_cfg = build_decoder_cfg(cfg, use_moe=False)
    target_params = count_params(RecurrentDecoderLM(indexed_cfg, shared_across_steps=False))

    moe_cfg = cfg.get("moe", {})
    chosen_e, chosen_ff, approx_params = pick_shared_moe_shape_for_target(
        base_cfg=indexed_cfg,
        target_params=target_params,
        num_experts=(int(moe_cfg["num_experts"]) if "num_experts" in moe_cfg else None),
        num_experts_min=int(moe_cfg.get("num_experts_min", 8)),
        num_experts_max=int(moe_cfg.get("num_experts_max", 64)),
        prefer_under_target=bool(moe_cfg.get("strict_match", False)),
    )
    pad = max(0, target_params - approx_params) if bool(moe_cfg.get("strict_match", False)) else 0

    shared_cfg = build_decoder_cfg(cfg, use_moe=True)
    shared_cfg.moe_num_experts = chosen_e
    shared_cfg.moe_d_ff = chosen_ff

    model = RecurrentForCausalLM(RecurrentDecoderLM(shared_cfg, shared_across_steps=True), param_budget_pad=pad)
    total_params = count_params(model)
    if total_params != target_params:
        raise RuntimeError(f"Parameter mismatch: got {total_params:,}, expected {target_params:,}")

    x = torch.randint(0, shared_cfg.vocab_size, (2, 64))
    y = x.clone()
    out = model(input_ids=x, labels=y)
    loss = out["loss"]
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite MoE loss")
    loss.backward()

    print(
        "OK",
        f"target_params={target_params:,}",
        f"experts={chosen_e}",
        f"expert_d_ff={chosen_ff}",
        f"top_k={shared_cfg.moe_top_k}",
        f"pad={pad:,}",
        f"loss={loss.detach().item():.4f}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()
