# Why more compute stops helping neural CFR+

This note gives one explanation for the central problem in the neural CFR+
work so far: exploitability falls, levels out, and then often *rises*, while
more iterations or longer runs do not help. The behavior was seen on the 18-claim
game (exact evaluation) and the 69-claim game (approximate BR). It builds on
[from exact CFR+ to neural CFR+](neural_cfr_plus_from_exact.md) and the
September 2026 CPU experiments linked there.

**Status: a hypothesis based on analysis, not yet tested directly.** It matches
the existing evidence well (section 3). Section 5 proposes a cheap experiment
that would confirm or refute it.

## 1. Summary

There is no evidence of an implementation bug: the root value checks, the
streamed-vs-old traversal comparison, and the exact oracles all agree. The
issue is structural. The neural update has an **error that does not shrink
with the iteration count**, while the true signal it needs to track does
shrink. Past some iteration the error dominates, so the current policy
is increasingly shaped by it, not by regret. Linear averaging then gives
those late iterates the most weight. More iterations drive the run toward
the update's own biased steady state, not toward equilibrium.

## 2. The mechanism

### 2.1 The update feeds on its own output

The regret network approximates `R⁺_t / t` and is trained toward

```text
target_t = ReLU( (t-1)/t · ReLU(net_{t-1}) + r_t / t )
```

where `net_{t-1}` is **the previous network's prediction**, not a stored table
(see `_regrets_and_strategy` / the streaming target in
[`neural_cfr_plus_gpu.py`](../../liars_poker/algo/neural_cfr_plus_gpu.py)).
Write the network's error as `e_t = net_t − R⁺_t/t`. Each iteration, the
update adds some fresh error `δ_t` (sampling noise surviving the clip, clipping
bias, finite-step fitting and SGD noise), so roughly

```text
e_t ≈ (t-1)/t · e_{t-1} + δ_t
    = Σ_{s≤t} (s/t) · δ_s          (unrolled)
```

The weight on an old error, `s/t`, decays only as fast as new errors are
added: old errors are **inherited, not averaged away**. Exact CFR+ has
`δ_t = 0`, so this never comes up.

### 2.2 Two kinds of per-iteration error behave differently

**(a) Errors on the new-regret term, which scale like `1/t`.** Sampling noise
in `r_t` enters as `r_t / t`. With per-record clipping, an action whose
expected instantaneous regret is near zero gets a positive bias of about
`E[ReLU(X)] = σ/√(2π) ≈ 0.4σ` per record, where `X ~ N(0, σ²)`. After the
`1/t` scaling this is `δ_s ≈ 0.4σ/s`, and

```text
e_t ≈ Σ_s (s/t) · 0.4σ/s = 0.4σ        — a constant floor.
```

**(b) Errors from fitting, which do not scale with `1/t`.** A fixed number of
Adam steps at a fixed learning rate leaves an error `δ_s` of roughly constant
size `ε` each iteration, whatever `t` is. If those errors were independent
with mean zero,

```text
Var(e_t) ≈ ε² · Σ_s (s/t)² ≈ ε² t / 3   — e_t grows like ε·√t (random-walk drift).
```

If they had a systematic component, the growth would be linear in `t`.
Realistically, fitting errors are correlated between iterations and partly
limited by network smoothness, so `√t` is a caricature. The qualitative point
survives: nothing in the update pulls accumulated fitting error back to zero.

### 2.3 Meanwhile the signal shrinks

CFR+ bounds cumulative regret by `O(√t)`, so the quantity the network
stores, `R⁺_t / t`, is at most `O(1/√t)`. It goes to zero as the policy
approaches equilibrium. The strategy depends on the **ratios** between
positive regrets, so what matters is error *relative to* signal:

```text
signal   ~ 1/√t      (shrinking)
error    ~ constant (clip bias)  or  growing (fitting drift)
```

These must cross. Before the crossing, CFR+ makes progress. After it, the
current strategy is shaped mainly by accumulated error. The linearly weighted
average (weight `t`) then fills up with those error-dominated iterates, so
**average-policy exploitability can rise**, not just stall.

### 2.4 What the clip bias does to the policy

In equilibrium, every action in a player's mix has expected instantaneous regret
near zero; that is what indifference means. Case (a) raises *all* such actions,
and mildly bad ones with `|μ| ≲ σ`, to a common floor of about `0.4σ`. The true
regrets that separate them shrink toward zero. The current policy drifts toward
**uniform over every action that isn't clearly bad**, losing the finely balanced
mixes an equilibrium needs. The result is a biased steady state, not divergence,
and its location depends on `σ`, not on how long you train.

This is a known weakness of CFR+ under sampling; the regret floor was designed
for exact updates. To our recollection, the Deep CFR paper builds on Linear
CFR instead of CFR+, and Monte Carlo variants usually use Linear or Discounted
CFR. (Worth checking against the papers.)

## 3. How this matches the evidence

| Observation | Explanation under this hypothesis |
| --- | --- |
| 18- and 69-claim runs improve, bottom out, then worsen | Signal `~1/√t` crosses a constant or growing error; linear averaging then weights late, error-dominated iterates most |
| A learning-rate drop to `1e-4` helped on 18 claims (≈0.024 → ≈0.014) | Smaller fitting error `ε`: lower floor, later crossing |
| Optimizer reset alone did not help | It does not change the size of `δ` |
| [Aggregate-then-clip](../experiments/2026-09-28_01-04-50_cfr_plus_neural_clip_order_cpu.md) helps substantially | Clipping the mean of `n` samples shrinks the clip bias by about `√n` |
| …but the [18-claim run](../experiments/2026-09-28_13-29-00_18_claim_target_sampling_factorial.md) remains ~10× above tabular CFR+ (≈0.024 vs 0.00197) | Rarely visited, deep information sets have `n ≈ 1`, so no `√n` benefit there; fitting drift (b) is untouched |
| 4,096 traversals help *per iteration* but not reliably *per minute* | More samples shrink `σ` but not `ε`; each iteration costs more |
| 4,096 traversals help clip-each little | Per-record clipping does not average its bias away |
| Tabular sampled CFR+ worked well (June 2026) | A table has no fitting error, and summing per information set avoids the per-record clip |
| Regret-fit validation error is tiny but the policy is poor | Targets are mostly the network's own output; it can fit itself well while drifting |
| Neural current play differs from the exact [shadow ledger](../experiments/2026-09-27_23-58-30_cfr_plus_shadow_neural_cpu.md) (root TV ≈ 0.2–0.3) | Accumulated error changes the action ratios |
| 6-claim runs had not deteriorated by 800 iterations | `ε` is small relative to the signal on this game; the crossing would come later |

## 4. What this implies for fixes

The problem is that **the network is asked to store a running sum by
learning from its own previous output**. Tuning a long run cannot change
that. Changes that could turn compute back into quality, roughly ordered by
how directly they target the cause:

1. **Remove the self-reference.** Store instantaneous regrets `(I, r_s, weight s)`
   in a large (reservoir) buffer and regress the network onto their weighted
   mean. This is Deep CFR with Linear CFR weighting. Apply the positive part
   only during regret matching. The error then becomes ordinary regression
   error on a growing dataset, which shrinks with data and capacity; it does
   not build up. The older Deep CFR plateau (≈0.084 on 18 claims) predates the
   larger networks, positive-weighted loss and GPU traversal, so a rematch
   with the current setup is warranted.
2. **If you keep the current update, shrink `δ` on a schedule.** Decay the
   learning rate and grow traversals with `t`: the standard condition for a
   noisy iterative update to converge. The single learning-rate drop was a
   crude version of this and helped.
3. **Use a base algorithm that does not clip noisy sums.** Linear or Discounted
   CFR store raw (unclipped) cumulative regret, removing the clip bias (a)
   entirely instead of merely reducing it, as aggregation does.

## 5. A cheap test

**Tabular noise injection, with no neural networks.** On the 6- or 18-claim
game, run exact tabular CFR+ in the same normalized form, adding a
per-iteration perturbation:

```text
R/t ← ReLU( (t-1)/t · R/t + r_t/t + δ_t )
```

Try (i) zero-mean `δ_t` of fixed size `ε`, (ii) a fixed positive bias, and
(iii) clip-style bias `≈ 0.4σ/t` from per-record clipping of noisy `r_t`.
Evaluate the average policy exactly. The hypothesis predicts:

- (i) and (ii) reproduce *improve, then deteriorate*, with the turning point moving
  earlier as `ε` increases;
- (iii) gives a plateau whose level scales with `σ`;
- decaying `ε` with `t` removes the deterioration.

**Shadow-ledger check on the neural run.** Using the
[existing shadow-ledger script](../../scripts/shadow_neural_cfr_plus_cpu.py),
track the positive regret mass the network assigns to actions whose exact
ledger regret is zero. The hypothesis predicts that this share rises over `t`.

If both tests agree, the 69-claim runs were not short of compute. The algorithm
was converging to its own biased steady state, and the next step is a
different update, not a longer run.
