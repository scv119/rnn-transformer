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

---

## New Discussion: Smaller Model While Keeping Per-Token Activation Similar
Question discussed: whether to try a much smaller model (e.g., half parameters or less) while keeping per-token activation budget roughly the same.

Recommendation: **Yes, this is a strong next step** after we secure enough checkpoints from the current control run.

### Why this experiment is high-value
- Tests whether gains come from conditional-compute efficiency vs just larger total capacity.
- If quality is retained with much smaller total params, it strengthens both practical and research claims.

### Expected gains if successful
1. Lower memory footprint (model + optimizer states + checkpoints).
2. Cheaper/faster experimentation cycles (more sweeps under same budget).
3. Better serving economics potential (lower cost at similar quality).
4. Stronger paper narrative: "similar quality at much lower total parameters with matched active compute."

### What counts as a "good" result
At matched step/token budget versus control:
- **Excellent:** <3% quality drop with ~50% fewer params.
- **Good:** 3–7% drop with ~50% fewer params.
- **Weak:** >10% drop unless major latency/memory gains compensate.

### Suggested implementation notes
- Keep `top_k` fixed (e.g., 2) for clean comparison.
- Reduce total experts and/or expert width to hit ~0.5x params first.
- Keep training recipe fixed (LR schedule, data slice, eval cadence).
- Run stage-gated (1k -> 5k -> long) using same stop/promote policy above.

---

## Decision Update: Aux-Schedule Run (2026-02-15)

### Observation that triggered the decision
At matched steps, baseline still outperforms current MoE control:
- step `1500`: baseline eval `4.279` vs MoE eval `4.357`
- step `2000`: baseline eval `3.961` vs MoE eval `4.105`
- step `2500`: baseline eval `3.786` vs MoE eval `3.915`

Interpretation: the gap is not only because MoE has trained fewer total steps; optimization/routing regularization likely still needs tuning.

### Decision
Launch a new MoE run with a stronger early load-balance regularization schedule, then taper:
- aux coefficient schedule: `0.05 -> 0.01`
- warmup hold: first `5%` of steps
- decay-to-end by `30%` of steps
- keep architecture fixed for comparability (`num_experts=40`, `top_k=2`, `d_ff=2048`)

### New tracked experiment
- run name: `recurrent-shared-moe-40e-top2-auxsched-wikitext103`
- config: `configs/recurrent_shared_moe_40e_top2_auxsched_wikitext103.json`
- script: `scripts/run_recurrent_shared_moe_40e_top2_auxsched_300m.sh`
- output dir: `runs/recurrent_shared_moe_40e_top2_auxsched_wikitext103`
- live log: `logs/recurrent_shared_moe_40e_top2_auxsched_live.log`

### Implementation notes recorded
- training code now supports schedule fields:
  - `aux_loss_coef_start`
  - `aux_loss_coef_end`
  - `aux_warmup_frac`
  - `aux_decay_end_frac`
- schedule is resume-aware via checkpoint `global_step`.
- `[MOE]` logs now report current `coef` and `opt_step` with balance stats.

### Comparison protocol update
- For plots/comparisons, prefer run directories (`runs/...`) over appended live logs to avoid mixed-session artifacts.
