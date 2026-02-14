# Research Discussion — Ablations + Stopping Strategy

Date: 2026-02-14

## Current Context Snapshot
- Active run: `recurrent_shared_moe_40e_top2_ff2048_wikitext103`
- Recent train loss: `4.373 -> 4.368 -> 4.367 -> 4.318 -> 4.346 -> 4.312` (healthy downward trend with normal noise)
- Eval loss trajectory:
  - epoch `0.14`: `5.611`
  - epoch `0.2801`: `4.767`
  - epoch `0.4201`: `4.357`
- Baseline CPU eval reference (`checkpoint-19000`, validation subset 13299 tokens):
  - loss `2.6623`, ppl `14.329`

Interpretation: MoE run is improving steadily and is still far from baseline final quality, so this run should continue to collect meaningful learning curves.

---

## Should We Ablate This Current Run?
Recommendation: **Do not interrupt this run now**. Let it progress to a stronger checkpoint window first (e.g., 3k/5k/10k steps).

Reason:
- We need this run as a stable anchor for later ablation comparisons.
- Early-stage interruption reduces comparability and weakens conclusions.

---

## High-Value Ablations (Ranked)

### A1) Aux loss coefficient (`moe.aux_loss_coef`)
- Values: `0.0`, `0.005`, `0.01` (current), `0.02`
- Goal: quantify quality vs routing-balance tradeoff.
- Key metrics: eval loss, aux stability, expert-load entropy, max expert load.

### A2) Routing top-k
- Values: `top_k=1` vs `top_k=2` (current)
- Goal: quality/efficiency tradeoff; top-1 should be faster/sparser but may reduce quality.

### A3) Number of experts
- Values: `num_experts=20`, `40` (current), `60`
- Keep `active_ffn` budget logic explicit when comparing.
- Goal: test redundancy and conditional capacity scaling.

### A4) Expert FFN size (`moe.d_ff`)
- Values: `1024`, `2048` (current), maybe `3072`
- Goal: understand capacity split between router diversity and expert width.

### A5) Shared vs indexed depth (architecture-level)
- Compare to recurrent-indexed baseline under same token budget.
- Goal: isolate effect of shared-MoE design vs recurrent-depth alone.

---

## Smaller Iterations + Adaptive Stop Criteria
Recommendation: **Yes** — use stage gates to avoid wasting GPU time.

### Stage-gate schedule
- Fast probes: run ablations to `1k` steps first.
- Promote only promising configs to `5k`.
- Final candidates to `full/long` run.

### Suggested promote/stop rules
At each eval checkpoint (every 500 steps), compute:
- `best_eval_so_far`
- moving improvement over last 2 evals

Stop early if **all** are true:
1. No eval improvement > `0.02` for 3 consecutive evals, and
2. Train loss slope is near-flat, and
3. Routing metrics are not improving (entropy/load imbalance plateau).

Promote config if **any** are true:
1. Eval loss beats current reference by >= `0.05` at same step budget, or
2. Same eval loss but meaningfully better routing balance / efficiency.

---

## Practical Experiment Matrix (Lean)
Phase 1 (quick screen, 1k steps each):
1. `aux=0.0` (sanity lower bound)
2. `aux=0.005`
3. `aux=0.01` (control)
4. `aux=0.02`
5. `top_k=1` with aux best from above

Phase 2 (promoted configs, 5k steps):
- Best 2 from Phase 1
- Add `num_experts=20/60` around winner

Phase 3 (long run):
- 1–2 winners only

---

## What to Track for Publication-Quality Conclusions
- Quality: eval loss + ppl over steps
- Efficiency: tokens/sec, wall-clock to target loss
- MoE health: aux, entropy, min/max/top expert load, token drop (if any)
- Stability: gradient norm, NaN events

---

## Immediate Next Actions
1. Continue current run to at least next checkpoint milestone (>= 3k step suggested).
2. Prepare 1k-step ablation configs for `aux_loss_coef` sweep.
3. Add simple stage-gate script/criteria for automatic stop/promote decisions.
4. Keep plotting with deduped-step parser (already fixed).
