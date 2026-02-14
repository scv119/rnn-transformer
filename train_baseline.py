#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import shutil
from typing import Any, Dict

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.rnn_transformer.recurrent_baseline import DecoderConfig, RecurrentDecoderLM


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ~300M GPT baseline on WikiText-103")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional override")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional override")
    parser.add_argument("--max_train_examples", type=int, default=None, help="Optional train text rows slice")
    parser.add_argument("--max_eval_examples", type=int, default=None, help="Optional eval text rows slice")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--auto_resume", action="store_true", help="Auto-resume from last checkpoint in output_dir")
    return parser.parse_args()


def build_training_args(cfg: Dict[str, Any]) -> TrainingArguments:
    use_bf16 = bool(cfg["bf16"])
    use_fp16 = bool(cfg["fp16"])

    if use_bf16 and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        print("bf16 requested but unsupported on this host; disabling bf16.")
        use_bf16 = False

    if use_fp16 and not torch.cuda.is_available():
        print("fp16 requested but CUDA is unavailable; disabling fp16.")
        use_fp16 = False

    report_to = cfg.get("report_to", "none")
    run_name = cfg.get("run_name")

    return TrainingArguments(
        output_dir=cfg["output_dir"],
        do_train=True,
        do_eval=True,
        num_train_epochs=float(cfg["num_train_epochs"]),
        max_steps=int(cfg["max_steps"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        lr_scheduler_type="cosine",
        weight_decay=float(cfg["weight_decay"]),
        adam_beta1=float(cfg["adam_beta1"]),
        adam_beta2=float(cfg["adam_beta2"]),
        adam_epsilon=float(cfg["adam_epsilon"]),
        max_grad_norm=float(cfg["max_grad_norm"]),
        dataloader_num_workers=int(cfg["dataloader_num_workers"]),
        logging_strategy="steps",
        logging_steps=int(cfg["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(cfg["eval_steps"]),
        save_strategy="steps",
        save_steps=int(cfg["save_steps"]),
        save_total_limit=int(cfg["save_total_limit"]),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=bool(cfg["gradient_checkpointing"]),
        report_to=report_to,
        run_name=run_name,
    )


class RecurrentForCausalLM(nn.Module):
    def __init__(self, model: RecurrentDecoderLM, param_budget_pad: int = 0):
        super().__init__()
        self.model = model
        self.param_budget_pad = (
            nn.Parameter(torch.zeros(param_budget_pad), requires_grad=True) if param_budget_pad > 0 else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del attention_mask
        logits = self.model(input_ids)
        out: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            out["loss"] = loss
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pick_shared_moe_shape_for_target(
    base_cfg: DecoderConfig,
    target_params: int,
    num_experts: int | None,
    num_experts_min: int,
    num_experts_max: int,
    prefer_under_target: bool = False,
) -> tuple[int, int, int]:
    shared_dense_cfg = copy.deepcopy(base_cfg)
    shared_dense_cfg.moe_num_experts = 0
    shared_dense_cfg.moe_d_ff = None
    shared_dense_cfg.moe_top_k = 1

    shared_dense_params = count_parameters(RecurrentDecoderLM(shared_dense_cfg, shared_across_steps=True))
    dense_mlp_params = shared_dense_cfg.d_ff * (2 * shared_dense_cfg.d_model + 1) + shared_dense_cfg.d_model
    shared_non_mlp_params = shared_dense_params - dense_mlp_params
    d_model = shared_dense_cfg.d_model

    if num_experts is not None:
        expert_candidates = [int(num_experts)]
    else:
        expert_candidates = list(range(int(num_experts_min), int(num_experts_max) + 1))
        if not expert_candidates:
            raise ValueError("No valid expert candidate count provided.")

    best_choice: tuple[int, int, int] | None = None
    best_under_choice: tuple[int, int, int] | None = None
    for e in expert_candidates:
        numerator = target_params - shared_non_mlp_params - (e * (d_model + 1)) - (e * d_model)
        denominator = e * (2 * d_model + 1)
        ff = max(1, int(round(numerator / denominator)))
        total = shared_non_mlp_params + e * (ff * (2 * d_model + 1) + d_model) + e * (d_model + 1)
        delta = total - target_params

        if best_choice is None or abs(delta) < abs(best_choice[2] - target_params):
            best_choice = (e, ff, total)
        if total <= target_params and (best_under_choice is None or (target_params - total) < (target_params - best_under_choice[2])):
            best_under_choice = (e, ff, total)

    if prefer_under_target and best_under_choice is not None:
        return best_under_choice
    if best_choice is None:
        raise RuntimeError("Unable to derive shared-MoE shape for target parameters.")
    return best_choice


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)

    if args.max_steps is not None:
        cfg["max_steps"] = int(args.max_steps)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    auto_resume_cfg = bool(cfg.get("auto_resume", False))
    auto_resume = bool(args.auto_resume) or auto_resume_cfg

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint is None and auto_resume and os.path.isdir(cfg["output_dir"]):
        resume_from_checkpoint = get_last_checkpoint(cfg["output_dir"])

    if bool(cfg.get("overwrite_output_dir", False)) and os.path.isdir(cfg["output_dir"]) and resume_from_checkpoint is None:
        shutil.rmtree(cfg["output_dir"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    set_seed(int(cfg["seed"]))

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(cfg["dataset_name"], cfg["dataset_config_name"])
    if args.max_train_examples is not None:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_examples, len(dataset["train"]))))
    if args.max_eval_examples is not None:
        if "validation" in dataset:
            dataset["validation"] = dataset["validation"].select(
                range(min(args.max_eval_examples, len(dataset["validation"])))
            )
        if "test" in dataset:
            dataset["test"] = dataset["test"].select(range(min(args.max_eval_examples, len(dataset["test"]))))
    context_length = int(cfg["context_length"])

    def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(batch["text"], return_attention_mask=False)

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        num_proc=(None if int(cfg["dataloader_num_workers"]) <= 1 else int(cfg["dataloader_num_workers"])),
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    def group_texts(examples: Dict[str, Any]) -> Dict[str, Any]:
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])
        total_len = (total_len // context_length) * context_length

        if total_len == 0:
            return {"input_ids": [], "labels": []}

        input_ids = [
            concatenated["input_ids"][i : i + context_length]
            for i in range(0, total_len, context_length)
        ]
        return {"input_ids": input_ids, "labels": [x[:] for x in input_ids]}

    lm_ds = tokenized.map(
        group_texts,
        batched=True,
        num_proc=(None if int(cfg["dataloader_num_workers"]) <= 1 else int(cfg["dataloader_num_workers"])),
        desc="Packing tokens",
    )

    architecture = cfg.get("architecture", "stacked")
    if architecture == "stacked":
        model_cfg = GPT2Config(**cfg["model"])
        model_cfg.bos_token_id = tokenizer.bos_token_id
        model_cfg.eos_token_id = tokenizer.eos_token_id
        model_cfg.pad_token_id = tokenizer.pad_token_id

        model = GPT2LMHeadModel(model_cfg)
        if cfg["gradient_checkpointing"]:
            model.config.use_cache = False
        params = model.num_parameters()
    else:
        shared = architecture in {"recurrent_shared", "recurrent_shared_moe"}
        if architecture not in {"recurrent_indexed", "recurrent_shared", "recurrent_shared_moe"}:
            raise ValueError(f"Unsupported architecture: {architecture}")

        moe_cfg = cfg.get("moe", {})
        use_moe = architecture == "recurrent_shared_moe"
        base_dcfg = DecoderConfig(
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

        param_budget_pad = 0
        if use_moe and bool(moe_cfg.get("match_indexed_params", False)):
            indexed_cfg = copy.deepcopy(base_dcfg)
            indexed_cfg.moe_num_experts = 0
            indexed_cfg.moe_top_k = 1
            indexed_cfg.moe_d_ff = None
            target_params = count_parameters(RecurrentDecoderLM(indexed_cfg, shared_across_steps=False))

            chosen_e, chosen_ff, approx_params = pick_shared_moe_shape_for_target(
                base_cfg=indexed_cfg,
                target_params=target_params,
                num_experts=(int(moe_cfg["num_experts"]) if "num_experts" in moe_cfg else None),
                num_experts_min=int(moe_cfg.get("num_experts_min", 8)),
                num_experts_max=int(moe_cfg.get("num_experts_max", 64)),
                prefer_under_target=bool(moe_cfg.get("strict_match", False)),
            )
            base_dcfg.moe_num_experts = chosen_e
            base_dcfg.moe_d_ff = chosen_ff
            delta = approx_params - target_params

            if bool(moe_cfg.get("strict_match", False)) and delta < 0:
                param_budget_pad = -delta
                approx_params += param_budget_pad
                delta = approx_params - target_params

            print(
                "[INFO] shared_moe budget match: "
                f"experts={chosen_e}, expert_d_ff={chosen_ff}, top_k={base_dcfg.moe_top_k}, "
                f"target={target_params:,}, approx={approx_params:,}, delta={delta:+,}, pad={param_budget_pad:,}"
            )

        recurrent = RecurrentDecoderLM(base_dcfg, shared_across_steps=shared)
        model = RecurrentForCausalLM(recurrent, param_budget_pad=param_budget_pad)
        params = count_parameters(model)

    print(f"Model parameters: {params:,} ({params/1e6:.2f}M)")

    train_args = build_training_args(cfg)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=lm_ds["train"],
        eval_dataset=lm_ds["validation"],
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )

    if resume_from_checkpoint is not None:
        print(f"[INFO] Resuming from checkpoint: {resume_from_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(cfg["output_dir"])
    trainer.save_state()

    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(lm_ds["train"])
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(lm_ds["validation"])
    if "eval_loss" in eval_metrics:
        eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()
