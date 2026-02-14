import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecoderConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    dropout: float = 0.0
    moe_num_experts: int = 0
    moe_top_k: int = 1
    moe_d_ff: int | None = None


class TopKMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int):
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2 for MoE.")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts].")

        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        flat_x = x.reshape(-1, d_model)

        router_logits = self.router(flat_x)
        topk_scores, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        topk_gates = torch.softmax(topk_scores, dim=-1).to(flat_x.dtype)

        mixed = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            expert_mask = topk_indices == expert_id
            if not expert_mask.any():
                continue

            token_indices, topk_slot = torch.where(expert_mask)
            expert_in = flat_x.index_select(0, token_indices)
            expert_out = expert(expert_in)
            gates = topk_gates[token_indices, topk_slot].unsqueeze(-1)
            mixed.index_add_(0, token_indices, expert_out * gates)

        return mixed.view(bsz, seq_len, d_model)

    def output_projections(self) -> list[nn.Linear]:
        return [expert[2] for expert in self.experts]


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        moe_num_experts: int = 0,
        moe_top_k: int = 1,
        moe_d_ff: int | None = None,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        if moe_num_experts > 0:
            expert_d_ff = int(moe_d_ff if moe_d_ff is not None else d_ff)
            self.mlp = TopKMoE(d_model, expert_d_ff, moe_num_experts, moe_top_k)
            self.mlp_output_projs = self.mlp.output_projections()
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )
            self.mlp_output_projs = [self.mlp[2]]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln_1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_out)

        h = self.ln_2(x)
        mlp_out = self.mlp(h)
        x = x + self.dropout(mlp_out)
        return x


class StackedDecoderLM(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.d_ff,
                    cfg.dropout,
                    moe_num_experts=cfg.moe_num_experts,
                    moe_top_k=cfg.moe_top_k,
                    moe_d_ff=cfg.moe_d_ff,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self._apply_residual_scaling(cfg.n_layers)
        with torch.no_grad():
            self.lm_head.weight.copy_(self.tok_emb.weight)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            nn.init.normal_(module.out_proj.weight, mean=0.0, std=0.02)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _apply_residual_scaling(self, n_layers: int) -> None:
        scale = 1.0 / math.sqrt(2.0 * n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=0.02 * scale)
            for proj in block.mlp_output_projs:
                nn.init.normal_(proj.weight, mean=0.0, std=0.02 * scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device), diagonal=1
        )

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        return self.lm_head(x)


class RecurrentDecoderLM(nn.Module):
    def __init__(self, cfg: DecoderConfig, shared_across_steps: bool = False):
        super().__init__()
        self.cfg = cfg
        self.shared_across_steps = shared_across_steps

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        if shared_across_steps:
            self.shared_block = DecoderBlock(
                cfg.d_model,
                cfg.n_heads,
                cfg.d_ff,
                cfg.dropout,
                moe_num_experts=cfg.moe_num_experts,
                moe_top_k=cfg.moe_top_k,
                moe_d_ff=cfg.moe_d_ff,
            )
            self.step_blocks = None
        else:
            self.step_blocks = nn.ModuleList(
                [
                    DecoderBlock(
                        cfg.d_model,
                        cfg.n_heads,
                        cfg.d_ff,
                        cfg.dropout,
                        moe_num_experts=cfg.moe_num_experts,
                        moe_top_k=cfg.moe_top_k,
                        moe_d_ff=cfg.moe_d_ff,
                    )
                    for _ in range(cfg.n_layers)
                ]
            )
            self.shared_block = None

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self._apply_residual_scaling(cfg.n_layers)
        with torch.no_grad():
            self.lm_head.weight.copy_(self.tok_emb.weight)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            nn.init.normal_(module.out_proj.weight, mean=0.0, std=0.02)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _apply_residual_scaling(self, n_layers: int) -> None:
        scale = 1.0 / math.sqrt(2.0 * n_layers)
        if self.shared_across_steps:
            nn.init.normal_(self.shared_block.attn.out_proj.weight, mean=0.0, std=0.02 * scale)
            for proj in self.shared_block.mlp_output_projs:
                nn.init.normal_(proj.weight, mean=0.0, std=0.02 * scale)
        else:
            for block in self.step_blocks:
                nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=0.02 * scale)
                for proj in block.mlp_output_projs:
                    nn.init.normal_(proj.weight, mean=0.0, std=0.02 * scale)

    @classmethod
    def from_stacked(cls, stacked_model: StackedDecoderLM, shared_across_steps: bool = False) -> "RecurrentDecoderLM":
        cfg = stacked_model.cfg
        recurrent = cls(cfg, shared_across_steps=shared_across_steps)

        recurrent.tok_emb.load_state_dict(stacked_model.tok_emb.state_dict())
        recurrent.pos_emb.load_state_dict(stacked_model.pos_emb.state_dict())
        recurrent.ln_f.load_state_dict(stacked_model.ln_f.state_dict())
        recurrent.lm_head.load_state_dict(stacked_model.lm_head.state_dict())

        if shared_across_steps:
            recurrent.shared_block.load_state_dict(stacked_model.blocks[0].state_dict())
        else:
            for i in range(cfg.n_layers):
                recurrent.step_blocks[i].load_state_dict(stacked_model.blocks[i].state_dict())

        return recurrent

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device), diagonal=1
        )

        for i in range(self.cfg.n_layers):
            if self.shared_across_steps:
                x = self.shared_block(x, attn_mask=attn_mask)
            else:
                x = self.step_blocks[i](x, attn_mask=attn_mask)

        x = self.ln_f(x)
        return self.lm_head(x)


def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def matched_parameter_pairs(stacked: StackedDecoderLM, recurrent: RecurrentDecoderLM) -> list[tuple[torch.Tensor, torch.Tensor]]:
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    pairs.append((stacked.tok_emb.weight, recurrent.tok_emb.weight))
    pairs.append((stacked.pos_emb.weight, recurrent.pos_emb.weight))
    pairs.append((stacked.ln_f.weight, recurrent.ln_f.weight))
    pairs.append((stacked.ln_f.bias, recurrent.ln_f.bias))

    if recurrent.shared_across_steps:
        return pairs

    for i in range(stacked.cfg.n_layers):
        sb = stacked.blocks[i]
        rb = recurrent.step_blocks[i]
        for (_, sp), (_, rp) in zip(sb.named_parameters(), rb.named_parameters()):
            pairs.append((sp, rp))

    return pairs


def gradient_cosine(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    lhs: list[torch.Tensor] = []
    rhs: list[torch.Tensor] = []

    for lp, rp in pairs:
        if lp.grad is None or rp.grad is None:
            continue
        lhs.append(lp.grad.detach().reshape(-1))
        rhs.append(rp.grad.detach().reshape(-1))

    if not lhs:
        return 0.0

    lg = torch.cat(lhs)
    rg = torch.cat(rhs)
    return float(F.cosine_similarity(lg, rg, dim=0).item())
