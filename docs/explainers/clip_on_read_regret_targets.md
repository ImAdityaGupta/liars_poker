# Clip on read: unclipped regret targets for neural CFR+

**Status: implemented in the trainer; one 18-claim CPU control is running, and CUDA smoke has not run.** This note explains the idea and its assumptions. It builds on [from exact CFR+ to neural CFR+](neural_cfr_plus_from_exact.md) and the [neural CFR+ code map](neural_cfr_plus_code_map.md). The motivating comparison, with a worked loss example, is in the [regret fit-steps sweep](../experiments/18_claim/2026-09-30_00-55-10_18_claim_regret_fit_steps_sweep.md).

*Update (30 September 2026): at 18 claims, clip on read was 1.2–1.4× more exploitable than `aggregate_then_clip`. See [aggregate-then-clip, clip on read, and the hybrid](aggregation_and_clip_on_read.md) for why repeated information sets favour aggregation, when clip on read should help, and the hybrid that combines them.*

## 1. The idea

CFR+ clips the cumulative regret at zero **after** the iteration's increment has been added:

```text
R_t(I,a) = max(0, max(0, R_(t-1)(I,a)) + g_t(I,a)).
```

The neural trainer only sees sampled estimates `G_k` of `g_t`, one per visit to `I`. The question is where the outer `max(0, ·)` goes relative to averaging those samples. There are three options:

| `regret_target_mode` | Stored target for visit `k` | What the network learns | Where the outer clip happens |
| --- | --- | --- | --- |
| `clip_each_record` (historical) | `max(0, old + G_k)` | The mean of clipped samples: biased upwards | Before fitting, on every sample |
| `aggregate_then_clip` (current best) | `max(0, old + mean_k G_k)`, the same for every visit to `I` | The clipped group mean | Before fitting, per group of **identical** feature rows |
| **`clip_on_read`** (proposed) | `old + G_k`, **unclipped** and possibly negative | The unclipped mean, via squared error | **When the output is used**: regret matching, and as next iteration's `old` |

Here `old = max(0, previous network output)` in all three modes; this part does not change. In cumulative units the target is `old + G_k`; in normalized units it is `((t-1)/t)·old + G_k/t`.

Squared-error regression converges to the mean of its targets. If the network can fit `I` independently and the samples have the same effective weighting, clip on read learns `old + mean G`, and reading it gives `max(0, old + mean G)`. That matches the `aggregate_then_clip` target **in this idealised case**. With a shared network, limited fit steps and importance weights, their fitted policies can differ even at frequently visited information sets.

## 2. Why bother

1. **Singletons.** Grouping merges only identical feature rows within one iteration. An information set visited once gets `max(0, old + G_1)`: one noisy sample clipped, i.e. the old per-record bias. Clip on read keeps that sample's negative evidence. The network can average it with similar information sets through generalisation *before* any clipping. About half of visited 18-claim information sets have fewer than one expected visit per update even at 4,096 roots. At 69 claims almost every visit is a singleton.
2. **No grouping pass.** `aggregate_then_clip` runs `torch.unique` over the whole regret buffer each player update. That is why it is guarded to CPU, or to CUDA for the 18-claim spec only. At 69 claims, the grouping memory is prohibitive. Clip on read has no such step, so it works on any device and spec.
3. **It is how VR-DeepDCFR+ fits regrets.** [Their code](https://github.com/rpSebastian/DeepPDCFR) regresses `d_t·max(0, R_prev) + advantage` with plain MSE and clips only when computing strategies.
4. **A step towards DCFR.** Discounted CFR keeps *signed* regrets. A neural DCFR would need unclipped values. Here, a negative prediction is not carried forward as negative regret because `old` is clipped before every increment. DCFR would also require discount factors and a different averaging rule; see the separate [discounting experiment](../experiments/18_claim/2026-09-30_00-55-09_18_claim_tabular_discounting.md).

## 3. The loss must become plain squared error

`_train_model` weights each regret entry by `1 + regret_positive_weight · [target > 1e-6]`, with default weight 0.5, and normalises each row by its total weight. Under `aggregate_then_clip`, every visit to `I` has the same target. The weight then only shifts emphasis *between* entries, which is harmless to each entry's mean.

Under clip on read, each visit carries its own noisy, possibly negative sample, and the weight depends on the sample's **sign**. The fitted mean therefore moves towards the positive samples: the clipping bias again, in a softer form. Two samples `+1` and `−1` with true mean 0 fit to `0.2` with raw weights, or about `0.11` after our per-row normalisation.

Clip on read therefore requires `regret_positive_weight = 0`. The per-row normalisation is then harmless: every visit to one information set has the same legal mask, and so the same denominator. If emphasis is wanted later, the weight must depend only on quantities fixed before sampling, such as `old`.

## 4. Invariants the implementation must keep

- `old` is read as `max(0, network output)` wherever it enters a target. **Do not remove the ReLU on `old`.** Removing it would give a different algorithm, with negative regret carried forward.
- Everything that turns regret outputs into a strategy already applies `max(0, ·)` before normalising, with a uniform fallback. This must remain true.
- Illegal-action entries of targets remain zero, and the loss stays masked to legal entries.
- Importance weights from traverser-action sampling (inverse path inclusion) still multiply each row's loss. Weighted least squares then learns the importance-weighted mean, matching `aggregate_then_clip`'s weighted group mean.
- Checkpoints from existing modes must load unchanged.

## 5. Implementation map

Line numbers refer to the working tree on 30 September 2026 and will drift. Search for the quoted code.

### `liars_poker/algo/deep_cfr_plus.py`

1. **Constructor, mode validation** (`if regret_target_mode not in {"clip_each_record", "aggregate_then_clip"}`, near line 405). Add `"clip_on_read"`.
2. **Constructor guards** (near lines 419–448):
   - `clip_on_read` must **not** be subject to the `aggregate_then_clip` device/spec guard (`cuda_aggregate_supported`, `_CUDA_AGGREGATE_SPEC`). It is allowed on CPU and CUDA, for any spec, with either traversal backend.
   - The `cumulative` guard currently requires `aggregate_then_clip` plus the device/spec condition. Change it to allow `regret_target_mode in {"aggregate_then_clip", "clip_on_read"}`. Keep the device/spec condition only for `aggregate_then_clip`.
   - Raise `ValueError` if `clip_on_read` is combined with `regret_increment_reach_mode` other than `"none"`. The visit-based modes need per-group visit counts, which only exist under grouping.
   - Raise `ValueError` if `clip_on_read` is used with `regret_positive_weight != 0`. An explicit error is better than silently ignoring the setting, since the default is 0.5.
3. **Recursive traversal** `_traverse` (the block computing `old_scaled`, near lines 1048–1059). It currently always applies `target = np.maximum(target, 0.0)`. Apply that clip only for `clip_each_record`. For `clip_on_read`, store the raw target, with illegal entries still zeroed.
4. **`_train_regret`** (near line 1066). No aggregation for `clip_on_read`: go straight to `_train_model`, and do not group the validation buffer either.
5. **`_train_model` and `_validation_metrics_for`** (`entry_weight = 1.0 + self.regret_positive_weight * (y > 1e-6)`, near lines 1193 and 1259). These need no code change if the constructor enforces a weight of 0. Add a comment explaining why. Validation MSE under clip on read compares against noisy per-visit targets, so its floor is the sample variance, not zero. `support_accuracy`, which compares `pred > 0` with `target > 0`, becomes a noisy diagnostic.
6. **Checkpoint config** (`checkpoint_dict` stores `regret_target_mode`, and `load_checkpoint` defaults it). No change is needed. Add a small public method, e.g. `set_regret_target_mode(mode, regret_positive_weight)`. It should re-run the same validation as the constructor, so fork runners can switch an existing checkpoint to `clip_on_read` safely. Runners currently mutate attributes directly, as in `trainer.regret_train_steps = arm` in `run_cfr_plus_18_gpu_fit_forks.py`. Switching an `aggregate_then_clip` checkpoint to `clip_on_read` is valid: its outputs were trained on clipped targets, and `old` is read through a ReLU anyway.

### `liars_poker/algo/neural_cfr_plus_gpu.py`

7. **Both target constructions:** the non-streamed path near lines 743–756 and the streamed path near lines 1156–1169. Both now use:

   ```python
   targets = (
       raw_targets if self.trainer.regret_target_mode in {"aggregate_then_clip", "clip_on_read"}
       else torch.relu(raw_targets)
   ) * legal_mask
   ```

   The condition includes `regret_target_mode in {"aggregate_then_clip", "clip_on_read"}`, so both modes store raw targets before any grouping. `old_scaled = torch.relu(regret_values) * legal_mask` remains unchanged.
8. **`_regrets_and_strategy`** already applies `torch.relu` before regret matching; no change. **`_claim_edges`** priority sampling ranks claims by raw `regret_values`. Ranking by raw values is consistent with clipped reading (positive before non-positive), so no change is needed.

### Unchanged, but check

- `_strategy_from_features` and `current_policy_dense` in `deep_cfr_plus.py`, and `NeuralRegretMatchingPolicy._legal_probs` in `liars_poker/policies/neural_regret.py`, all apply `max(0, ·)` before normalising. Confirm, and do not change.
- `liars_poker/algo/cfr_plus_tabular_fork.py` requires `aggregate_then_clip`. Keep it that way. A table is an exact per-information-set averager, so for a table clip on read and aggregate-then-clip are identical.
- Audit scripts that define the sampled target **S** from the regret buffer: `scripts/audit_cfr_plus_18_late_update.py`, `scripts/audit_cfr_plus_neural_targets_cpu.py`, `scripts/shadow_neural_cfr_plus_cpu.py`. Under clip on read the buffer holds raw per-visit targets. The audit's S should be the weighted group mean, clipped: the network's ideal target. Check how each script obtains S before reusing it with this mode.

### Useful logging

Per player update, log:
- the fraction of fitted outputs at visited rows that are negative;
- the mean and standard deviation of raw targets;
- the fraction of distinct information sets visited once.

The last number says how much of the buffer is in the regime where clip on read and grouping differ.

## 6. Verification before any long run

1. **Raw-target identity.** With the same checkpoint and RNG state, one iteration in `aggregate_then_clip` and one in `clip_on_read` must store identical raw buffer targets **before** grouping. Only the grouping step and the loss weight should differ.
2. **Equivalence at repeated information sets.** On a small synthetic buffer where every information set appears many times, fit a network to convergence under both modes. The ReLU of the predictions should agree within tolerance. Use enough capacity that each information set can be fitted independently.
3. **Loss.** With `regret_positive_weight=0`, `_train_model`'s regret loss must equal the masked per-row mean squared error times the normalised importance weights.
4. **Guards.** `clip_on_read` constructs on CPU and CUDA for the 18- and 69-claim specs. Visit-based reach modes raise; a non-zero positive weight raises.
5. **Smoke.** Run one full iteration on CUDA for the 18-claim and 69-claim specs, with finite losses. Record peak GPU memory, which should be lower than `aggregate_then_clip` at 18 claims. Save and reload a checkpoint and confirm the mode round-trips.

## 7. First experiment

Run one from-scratch cumulative conditional `clip_on_read` arm with plain MSE, 4,096 roots and seed 17 on CPU. Compare its exact average-policy exploitability to the historical cumulative conditional 4,096-root run on the same port 8765 dashboard. The [experiment note](../experiments/18_claim/2026-09-30_10-37-00_18_claim_clip_on_read_and_aggregation_mse.md) records settings and limitations. This is a practical recipe comparison: the old arm also differs in positive-target loss weighting, and it ran earlier under different machine load. The tabular discounting experiment remains a separate, unimplemented proposal: clip on read still clips the *previous* prediction, and neither discounts old regrets nor changes average-policy weighting.
