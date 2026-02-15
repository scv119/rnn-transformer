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
    def __init__(
        self,
        model: RecurrentDecoderLM,
        param_budget_pad: int = 0,
        moe_aux_loss_coef_start: float = 0.0,
        moe_aux_loss_coef_end: float | None = None,
        moe_aux_warmup_steps: int = 0,
        moe_aux_decay_steps: int = 0,
        moe_balance_log_interval: int = 0,
        gradient_accumulation_steps: int = 1,
        initial_global_step: int = 0,
    ):
        super().__init__()
        self.model = model
        self.param_budget_pad = (
            nn.Parameter(torch.zeros(param_budget_pad), requires_grad=True) if param_budget_pad > 0 else None
        )
        self.moe_aux_loss_coef_start = float(moe_aux_loss_coef_start)
        self.moe_aux_loss_coef_end = (
            float(moe_aux_loss_coef_end) if moe_aux_loss_coef_end is not None else None
        )
        self.moe_aux_warmup_steps = max(0, int(moe_aux_warmup_steps))
        self.moe_aux_decay_steps = max(0, int(moe_aux_decay_steps))
        self.moe_balance_log_interval = int(moe_balance_log_interval)
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.initial_global_step = max(0, int(initial_global_step))
        self._forward_calls = 0

    def _current_optimizer_step(self) -> int:
        return self.initial_global_step + (self._forward_calls // self.gradient_accumulation_steps)

    def _current_aux_coef(self) -> float:
        if self.moe_aux_loss_coef_end is None or self.moe_aux_decay_steps <= 0:
            return self.moe_aux_loss_coef_start

        step = self._current_optimizer_step()
        if step <= self.moe_aux_warmup_steps:
            return self.moe_aux_loss_coef_start

        progress = min(
            1.0,
            float(step - self.moe_aux_warmup_steps) / float(self.moe_aux_decay_steps),
        )
        return self.moe_aux_loss_coef_start + (
            (self.moe_aux_loss_coef_end - self.moe_aux_loss_coef_start) * progress
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
            ce_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            aux_loss = getattr(self.model, "last_moe_aux_loss", None)
            aux_coef = self._current_aux_coef()
            total_loss = ce_loss
            if aux_loss is not None and aux_coef > 0.0:
                total_loss = total_loss + (aux_coef * aux_loss)
                out["moe_aux_loss"] = aux_loss.detach()
                out["ce_loss"] = ce_loss.detach()

            if self.training:
                self._forward_calls += 1
                should_log_balance = (
                    aux_loss is not None
                    and self.moe_balance_log_interval > 0
                    and (
                        self._forward_calls
                        % (self.moe_balance_log_interval * self.gradient_accumulation_steps)
                        == 0
                    )
                )
                if should_log_balance:
                    expert_load = getattr(self.model, "last_moe_expert_load", None)
                    if expert_load is not None:
                        load = expert_load.detach().float().cpu()
                        min_load = float(load.min().item())
                        max_load = float(load.max().item())
                        entropy = float((-(load * (load + 1e-9).log()).sum() / math.log(load.numel())).item())
                        top_vals, top_idx = torch.topk(load, k=min(4, load.numel()))
                        top_desc = ", ".join(
                            f"e{int(idx)}={float(val):.3f}" for idx, val in zip(top_idx.tolist(), top_vals.tolist())
                        )
                        print(
                            "[MOE] "
                            f"ce={float(ce_loss.detach().item()):.4f} "
                            f"aux={float(aux_loss.detach().item()):.4f} "
                            f"coef={aux_coef:.4f} "
                            f"opt_step={self._current_optimizer_step()} "
                            f"balance_entropy={entropy:.4f} "
                            f"load_min={min_load:.4f} "
                            f"load_max={max_load:.4f} "
                            f"top_load=[{top_desc}]"
                        )

            out["loss"] = total_loss
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

    resume_global_step = 0
    if resume_from_checkpoint is not None:
        trainer_state_path = os.path.join(resume_from_checkpoint, "trainer_state.json")
        if os.path.isfile(trainer_state_path):
            try:
                with open(trainer_state_path, "r", encoding="utf-8") as f:
                    trainer_state = json.load(f)
                resume_global_step = int(trainer_state.get("global_step", 0))
            except Exception:
                resume_global_step = 0

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
        aux_coef_start = float(moe_cfg.get("aux_loss_coef_start", moe_cfg.get("aux_loss_coef", 0.0)))
        aux_coef_end_raw = moe_cfg.get("aux_loss_coef_end")
        aux_coef_end = float(aux_coef_end_raw) if aux_coef_end_raw is not None else None
        aux_warmup_steps = max(0, int(round(int(cfg["max_steps"]) * float(moe_cfg.get("aux_warmup_frac", 0.0)))))
        aux_decay_end_steps = max(
            aux_warmup_steps,
            int(round(int(cfg["max_steps"]) * float(moe_cfg.get("aux_decay_end_frac", 0.0)))),
        )
        aux_decay_steps = max(0, aux_decay_end_steps - aux_warmup_steps)
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
        elif use_moe:
            print(
                "[INFO] shared_moe fixed shape: "
                f"experts={base_dcfg.moe_num_experts}, expert_d_ff={base_dcfg.moe_d_ff}, "
                f"top_k={base_dcfg.moe_top_k}, active_ffn={base_dcfg.moe_top_k * base_dcfg.moe_d_ff}, "
                f"aux_loss_coef_start={aux_coef_start}, "
                f"aux_loss_coef_end={aux_coef_end if aux_coef_end is not None else aux_coef_start}, "
                f"aux_warmup_steps={aux_warmup_steps}, "
                f"aux_decay_steps={aux_decay_steps}, "
                f"balance_log_interval={int(moe_cfg.get('balance_log_interval', int(cfg['logging_steps'])))}"
            )

        recurrent = RecurrentDecoderLM(base_dcfg, shared_across_steps=shared)
        model = RecurrentForCausalLM(
            recurrent,
            param_budget_pad=param_budget_pad,
            moe_aux_loss_coef_start=aux_coef_start,
            moe_aux_loss_coef_end=aux_coef_end,
            moe_aux_warmup_steps=aux_warmup_steps,
            moe_aux_decay_steps=aux_decay_steps,
            moe_balance_log_interval=int(moe_cfg.get("balance_log_interval", int(cfg["logging_steps"]))),
            gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
            initial_global_step=resume_global_step,
        )
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
