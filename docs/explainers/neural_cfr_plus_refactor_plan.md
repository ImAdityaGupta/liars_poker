# Neural CFR+ refactor plan

**Status (30 September 2026):** a small, behaviour-preserving refactor is committed (`a66eefc`). The structural refactor described below is **deliberately delayed**. It should be done in stages, each triggered by a concrete feature that needs it, not all at once while experiments are running. This note records what exists, which future features, mostly from [VR-DeepDCFR+ / VR-DeepPDCFR+](https://arxiv.org/abs/2511.08174), the code must accommodate, and the order in which to reshape it.

It complements the [code map](neural_cfr_plus_code_map.md), which describes the code as it is, and supersedes earlier versions of this note.

## 1. What has been done

Commit `a66eefc`:

- **One regret-target formula.** The recursive traversal and both batched paths (streamed and non-streamed) call `DeepCFRPlusTrainer.make_regret_record`, which uses [`cfr_plus_targets.make_regret_target`](../../liars_poker/algo/cfr_plus_targets.py). The table discount trainer overrides `make_regret_record` to store the sampled advantage directly. Its buffer's `targets` column therefore holds advantages, not targets.
- **Explicit regret reads.** Traversal and current-policy evaluation read regrets through `regret_values_tensor`, backed by a [`NetworkRegretReader` or `TableRegretReader`](../../liars_poker/algo/regret_readers.py). The table variants no longer intercept `_forward`.
- **Tests.** [`tests/test_cfr_plus_refactor.py`](../../tests/test_cfr_plus_refactor.py) has nine CPU checks: formulas, table seeding, streamed and non-streamed iterations, and checkpoint round trips. Run them with `python -m unittest tests.test_cfr_plus_refactor`; pytest is not installed.
- **Equivalence with the pre-refactor code.** A one-off comparison ran identical seeded iterations under the VM's pre-refactor `deep_cfr_plus.py`, `neural_cfr_plus_gpu.py` and `cfr_plus_tabular_fork.py`, and under the refactored code. It covered 12 configurations:
  - clip-each, aggregate and clip-on-read targets;
  - normalized and cumulative units;
  - streamed traversal and a claim cap of 1;
  - `visit_count`;
  - uniform and linear strategy weighting;
  - the recursive backend;
  - the table fork in both units.

  Buffers, losses, network parameters and table rows were **bit-identical** on CPU. CUDA was not compared. That comparison was not committed; Stage 0 below turns it into permanent fixtures.

## 2. What remains awkward

1. **Mode flags instead of a rule.** `regret_target_mode`, `regret_accumulation_mode`, `regret_increment_reach_mode`, `regret_positive_weight` and a device/spec guard jointly define one choice, the regret update rule. `_validate_regret_target_mode` keeps the combinations legal, but each new rule adds more flags.
2. **Regret matching is implemented five times:**
   - `GPUDeepCFRPlusTraverser._regrets_and_strategy`;
   - `DeepCFRPlusTrainer._strategy_from_features`;
   - `current_policy_dense`;
   - `_regret_matching_tensor`, for validation;
   - `NeuralRegretMatchingPolicy._legal_probs`, for the exported current policy.

   All five apply `max(0, ·)` with a uniform fallback. Predictive CFR and an argmax fallback would each have to change all five consistently.
3. **The traverser has no extension points.** Sampling policy, value estimates at unexpanded actions (currently zero or the `"call"` value), and what gets recorded are all fixed inside `neural_cfr_plus_gpu.py`. The learned baseline and outcome sampling both need to change these.
4. **One class owns everything.** `DeepCFRPlusTrainer` owns regret networks and fitting, the average network and its reservoir, the recursive traversal, grouping, checkpoints and diagnostics. The table variants subclass it and inherit neural machinery they do not use.
5. **Misleading names.** `neural_cfr_plus_gpu.py`, `GPUDeepCFRPlusTraverser` and `traversal_backend="gpu_native"` mean batched tensors on any device. `run_cfr_plus_18_target_order_cpu_overnight.py` is now the general 18-claim runner. The CUDA guard names a game spec (`_CUDA_AGGREGATE_SPEC`) where it means a memory limit.
6. **The recursive backend** duplicates traversal and target logic. Only a June notebook uses it.
7. **Runners duplicate infrastructure.** Eight or so runners each implement checkpoints, snapshots, resume and manifests. About a dozen scripts reach into private trainer members.

None of these blocks the currently queued experiments. That is why the structural work is delayed.

## 3. Features the code should be able to absorb

This section drives the design. Each row is something we may want, and says what it requires from the code, not how to implement it.

| Feature | Source | What it needs from the code |
| --- | --- | --- |
| **Clip on read** | Paper (done) | A rule that records unclipped targets, plain MSE, and a clipped read. *Exists as a mode.* |
| **DCFR+ discount of old regret** | Paper, α = 2. Their code uses `+1.5` in the denominator where the paper has `+1`. | Rule: `target = d_t · max(0, old) + advantage`. Only visited information sets are discounted in the neural version. |
| **DCFR with signed regrets** | Brown and Sandholm | Rule with signed storage. Positive and negative old values discounted differently (`t^α/(t^α+1)`, `t^β/(t^β+1)`); regret matching reads positive parts only. |
| **Quadratic or power averaging** | Paper, γ = 2 | Strategy-record weight `t^γ`. *`quadratic` exists; generalise to a power.* |
| **Predictive regret matching (PDCFR+)** | Paper | A second per-player network fitted to the *instantaneous* advantage. The current strategy becomes regret matching on `max(0, d·max(0, R) + r̂)`. So "strategy from regrets" must be able to combine **two** stores. This touches traversal, current-policy compilation and the exported current policy. |
| **Argmax fallback** | Paper's code | When no regret is positive, play the highest-regret action rather than uniform. A property of the one shared regret-matching function. |
| **Learned history baseline Q** | Paper; VR-MCCFR; see [learned history baselines](learned_history_baselines.md) | Traversal must compute **history features** (both hands plus public claims). It calls a value model at visited nodes to fill in values of unexpanded actions and correct the sampled one, at traverser **and** opponent nodes. It emits **transitions** into a persistent buffer. Q is trained after each player update with an expected-SARSA target under the *next* strategy, so the baseline trainer needs the strategy function. Options: retrain from scratch, a target network, a best-loss snapshot. |
| **Traverser action sampling / outcome sampling** | Paper uses outcome sampling with ε = 0.6; we use claim caps | A **sampling policy** separate from the current strategy: exploration mixture, caps, or one action. Inclusion probabilities feed the baseline correction and record weights. Outcome sampling is the one-action case with a baseline. |
| **Old regret from a frozen network at fit time** | Paper's `target_model` | An alternative to recording the target during traversal. Records carry only the advantage, and `old` is read from a frozen copy of the pre-update network when fitting. It avoids storing targets, but costs a forward pass per fitted batch. |
| **Average network refit from scratch** | Paper: every 3 iterations, 5,000 steps, MSE on probabilities | Average learner options: reinitialise schedule, loss type, steps. Low priority, since the current average network is adequate. |
| **Fit-quality diagnostics** | Our fit-steps sweep | Public hooks: collect one update's samples without fitting, fit on given samples, read regrets at arbitrary features, measure change at unvisited information sets. |
| **Table and network regret stores on the same traversal** | Our tabular fork | Swap the regret store while holding traversal and averaging fixed. This is today's fork, made a configuration. |

The hardest rows are **predictive regret matching** and the **learned baseline**. The former needs a pluggable strategy function; the latter needs pluggable traversal. They define the shape of the refactor.

## 4. Target architecture

```text
             ┌────────── strategy function ◄── regret store(s) [+ predictor]
             │                  ▲
sampling ────┤                  │ update(samples, rule)
policy       ▼                  │
        traversal engine ──► regret samples ──► regret update rule
             │    ▲         strategy samples ──► average learner
             │    └── value model (optional baseline Q) ◄── transitions
             ▼
        iteration coordinator ──► run controller (checkpoints, snapshots, evaluation)
```

| Component | Owns | Replaces |
| --- | --- | --- |
| **Strategy function** | One regret-matching implementation, with a fallback option (uniform or argmax), optionally combining a cumulative store with a predictor | The five regret-matching copies |
| **Traversal engine** | Batched game progression on CPU or CUDA; streamed and unsplit schedules sharing one semantics; history features when a value model is present | `GPUDeepCFRPlusTraverser` and the recursive backend |
| **Sampling policy** | Which traverser and opponent actions are followed, and their inclusion probabilities | Cap/fraction/schedule flags inside the traverser |
| **Value model** (optional) | Q network, transition buffer, training schedule and targets | Zero or `"call"` baselines |
| **Regret update rule** | How `old` is read (clipped, discounted, signed), how it combines with the advantage, where clipping happens, whether grouping is needed, and the required loss weighting | The mode flags and `_validate_regret_target_mode` |
| **Regret store** | Network (models, optimisers, fit buffer, fitting) or table (keys, values, seeding, lazy discount clocks); raw reads | Regret parts of `DeepCFRPlusTrainer`; `TabularRegretFork`; `TabularDiscountTrainer` |
| **Average learner** | Reservoir, record weighting, fitting schedule, export of `NeuralPolicy` | Strategy parts of `DeepCFRPlusTrainer` |
| **Coordinator** | Alternating player updates and freezing policies during traversal; calls the components in order | `run_iteration` |
| **Run controller** | Budgets, snapshots, checkpoints of trainer plus run state, evaluation scheduling, resume | The per-experiment runner loops |

### Where `old` is computed

Today the traversal already reads each visited information set's regret, to choose its strategy. `make_regret_record` reuses that value to form the target immediately. The buffer stores only the target, which needs no extra memory and is bit-exact.

Keep this as the default. Formalise it as the traversal handing `(old_raw, advantage, mask)` to the **rule**, not to trainer flags. The table discount rules already use the same hook to record the bare advantage. The paper's approach, reading `old` from a frozen network at fit time, becomes an *alternative rule*, to be measured before adoption. At 69 claims it roughly doubles regret-network inference, and it is not bit-identical to the current path.

### Compatibility

- Constructor keyword arguments keep working by translating to a rule.
- Checkpoint formats (trainer version 2, fork format 3, discount format 1) load through a translation layer.
- The resume path for an old VM run is its own code checkout; new code need not resume every historical run. Before deploying, verify one real VM checkpoint loads and continues identically in a fresh checkout.
- Keep aliases for renamed modules and for `traversal_backend="gpu_native"`, because checkpoints store it.

## 5. Stages and their triggers

Each stage ends with the Stage 0 fixtures passing **bit-identically on CPU** for all configurations that still exist. A learning curve is compared only where a stage intentionally changes arithmetic.

| Stage | Work | Trigger |
| --- | --- | --- |
| **0. Fixtures** | Commit golden outputs from current `HEAD` for the configurations in section 1, plus the discount rules and uniform/linear/quadratic weighting. Include a small script that regenerates them, and a CUDA smoke comparison within tolerance. Add a one-line comment on `TabularDiscountTrainer.make_regret_record` saying its "targets" are advantages. | **Now.** Cheap, and everything else depends on it |
| **1. Rule and strategy function** | Introduce `RegretUpdateRule`, translating existing flags into it. Route all five regret-matching copies through one strategy function with a fallback option. | The first neural rule beyond today's modes (neural DCFR+ / DCFR), predictive matching, or argmax fallback |
| **2. Traversal engine** | Separate sampling policy and optional value-model hooks; history features; transition emission; one semantics for streamed and unsplit schedules; rename with aliases; remove the recursive backend. | Work starting on the learned baseline, outcome sampling, or new claim-sampling schemes for 69 claims |
| **3. Stores and average learner** | `NetworkRegretStore` and `TableRegretStore`; separate the average learner; express the fork and discount trainers as configurations; move 18-claim-specific harnesses to `liars_poker/experimental/`. | Predictive CFR (a second store per player), average-network schedule changes, or the next table-versus-network comparison |
| **4. Run controller** | Shared checkpoint/snapshot/resume/evaluation loop; checkpoints that include run-controller state; one config-driven 18-claim runner; public diagnostic hooks for audit scripts; archive superseded runners. Port the 69-claim adaptive controller only when those runs restart. | Starting 69-claim runs again, or the next time a runner must be written from scratch |

Stages 1 and 2 are independent and can happen in either order. Stage 3 benefits from Stage 1's rule object.

## 6. Gates

**Correctness** (required for every stage):

1. Stage 0 fixtures are bit-identical on CPU, and within tolerance on CUDA.
2. Checkpoints of every existing format load. A real VM checkpoint continues one iteration identically to the old checkout.
3. `current_policy_dense()`, `average_policy()` and exported `NeuralRegretMatchingPolicy` outputs are unchanged for a fixed checkpoint.
4. A short 18-claim exact-exploitability comparison, only where arithmetic intentionally changed. Compare by iteration and by wall time; single snapshots scatter about ±15%.

**Scaling** (separate, for 69-claim readiness, not a merge requirement): peak live rows, buffer use, chunk counts, and time in traversal, target formation and fitting, at cap 16/24 and larger traversal counts on the GPU intended for those runs.

## 7. Out of scope

- The exact solvers and best responses: `cfr_exact_dense.py`, `cfr_plus_dense.py`, `br_exact*.py`, `br_exact_dense_to_dense.py`, `DenseTabularPolicy`. Also the `increment_transform` hook used by the tabular bridge.
- Approximate best responders, evaluation code, policy serialization formats.
- Original Deep CFR (`deep_cfr.py`), except shared buffer utilities.
- Changing any learning behaviour **as part of** a refactor stage. New rules and features land in separate commits after the stage that makes room for them, so any change in a learning curve has one cause.

## Related notes

[Code map](neural_cfr_plus_code_map.md) · [Clip on read](clip_on_read_regret_targets.md) · [Learned history baselines](learned_history_baselines.md) · [Tabular discounting experiment](../experiments/18_claim/2026-09-30_00-55-09_18_claim_tabular_discounting.md) · [Regret fit-steps sweep](../experiments/18_claim/2026-09-30_00-55-10_18_claim_regret_fit_steps_sweep.md)
