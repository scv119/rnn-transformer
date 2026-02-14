#!/usr/bin/env python3
"""Lightweight CPU eval for checkpoints while training continues on GPU.

Evaluates cross-entropy loss and perplexity on a dataset split using the same
packing logic as train_baseline.py.
"""

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import load_dataset
from safetensors.torch import load_file as safe_load_file
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, default_data_collator
from transformers.models.auto.tokenization_auto import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rnn_transformer.recurrent_baseline import DecoderConfig, RecurrentDecoderLM
from train_baseline import RecurrentForCausalLM


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPU checkpoint eval (loss + perplexity)")
    p.add_argument("--config", required=True, help="Path to training JSON config")
    p.add_argument("--checkpoint", required=True, help="Checkpoint dir (e.g. runs/.../checkpoint-1000)")
    p.add_argument("--split", default="validation", choices=["validation", "test", "train"])
    p.add_argument("--max_eval_examples", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--dataloader_workers", type=int, default=2)
    return p.parse_args()


def build_model(cfg: Dict[str, Any], tokenizer: AutoTokenizer) -> torch.nn.Module:
    architecture = cfg.get("architecture", "stacked")

    if architecture == "stacked":
        model_cfg = GPT2Config(**cfg["model"])
        model_cfg.bos_token_id = tokenizer.bos_token_id
        model_cfg.eos_token_id = tokenizer.eos_token_id
        model_cfg.pad_token_id = tokenizer.pad_token_id
        return GPT2LMHeadModel(model_cfg)

    shared = architecture in {"recurrent_shared", "recurrent_shared_moe"}
    if architecture not in {"recurrent_indexed", "recurrent_shared", "recurrent_shared_moe"}:
        raise ValueError(f"Unsupported architecture: {architecture}")

    moe_cfg = cfg.get("moe", {})
    use_moe = architecture == "recurrent_shared_moe"
    dcfg = DecoderConfig(
        vocab_size=int(cfg["model"]["vocab_size"]),
        max_seq_len=int(cfg["context_length"]),
        d_model=int(cfg["model"]["n_embd"]),
        n_heads=int(cfg["model"]["n_head"]),
        d_ff=int(cfg["model"]["n_inner"]),
        n_layers=int(cfg["model"]["n_layer"]),
        dropout=float(cfg["model"].get("resid_pdrop", 0.0)),
        moe_num_experts=(int(moe_cfg.get("num_experts", 0)) if use_moe else 0),
        moe_top_k=(int(moe_cfg.get("top_k", 1)) if use_moe else 1),
        moe_d_ff=(int(moe_cfg["d_ff"]) if use_moe and moe_cfg.get("d_ff") is not None else None),
    )
    recurrent = RecurrentDecoderLM(dcfg, shared_across_steps=shared)
    return RecurrentForCausalLM(
        recurrent,
        param_budget_pad=0,
        moe_aux_loss_coef=float(moe_cfg.get("aux_loss_coef", 0.0)),
        moe_balance_log_interval=0,
        gradient_accumulation_steps=1,
    )


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_dir: str) -> None:
    safetensor_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if os.path.exists(safetensor_path):
        state_dict = safe_load_file(safetensor_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {checkpoint_dir}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (first 8: {missing[:8]})")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (first 8: {unexpected[:8]})")


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(cfg["dataset_name"], cfg["dataset_config_name"]) 
    split = ds[args.split]
    if args.max_eval_examples is not None:
        split = split.select(range(min(args.max_eval_examples, len(split))))

    context_length = int(cfg["context_length"])

    def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(batch["text"], return_attention_mask=False)

    tokenized = split.map(
        tokenize_batch,
        batched=True,
        num_proc=(None if int(args.dataloader_workers) <= 1 else int(args.dataloader_workers)),
        remove_columns=split.column_names,
        desc="Tokenizing",
    )

    def group_texts(examples: Dict[str, Any]) -> Dict[str, Any]:
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concatenated["input_ids"]) // context_length) * context_length
        if total_len == 0:
            return {"input_ids": [], "labels": []}
        input_ids = [concatenated["input_ids"][i:i + context_length] for i in range(0, total_len, context_length)]
        return {"input_ids": input_ids, "labels": [x[:] for x in input_ids]}

    packed = tokenized.map(
        group_texts,
        batched=True,
        num_proc=(None if int(args.dataloader_workers) <= 1 else int(args.dataloader_workers)),
        desc="Packing tokens",
    )

    model = build_model(cfg, tokenizer)
    load_checkpoint_weights(model, args.checkpoint)
    model.eval()
    model.to("cpu")

    loader = DataLoader(
        packed,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=int(args.dataloader_workers),
    )

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            out = model(input_ids=input_ids, labels=labels)
            loss = out["loss"].detach().float().item()
            # approximate token count for shifted CE
            valid_tokens = int((labels[:, 1:] != -100).sum().item())
            if valid_tokens <= 0:
                valid_tokens = labels.numel()
            total_nll += loss * valid_tokens
            total_tokens += valid_tokens

    mean_loss = total_nll / max(total_tokens, 1)
    ppl = math.exp(min(20.0, mean_loss))

    print("=== CPU Eval Summary ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"split: {args.split}")
    print(f"tokens: {total_tokens}")
    print(f"loss: {mean_loss:.4f}")
    print(f"perplexity: {ppl:.3f}")


if __name__ == "__main__":
    main()
