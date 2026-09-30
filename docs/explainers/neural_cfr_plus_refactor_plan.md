# Refactor plan: neural CFR+ trainer and its experimental variants

**Status: core refactor implemented locally and checked on CPU.** This note records the original proposal and the concrete implementation below. It complements the [code map](neural_cfr_plus_code_map.md).

Neural target arithmetic, fitting and checkpoint formats are preserved. The tabular discount trainer now stores the sampled advantage directly in its transient regret buffer; those buffers are cleared each iteration and are not checkpointed. Long learning-curve equivalence remains unmeasured.

## Review and decision, 30 September 2026

The duplication in section 4.1 was real. Recursive traversal, batched traversal and streamed traversal built the same regret target separately. They now call the trainer's `make_regret_record`, which uses [one target calculation](../../liars_poker/algo/cfr_plus_targets.py) for neural runs. The table discount diagnostic overrides that hook to store the sampled advantage directly. It no longer subtracts the old regret from a target to recover the advantage. Grouping, fitting, strategy records, checkpoint formats and the existing constructor arguments are unchanged.

All regret reads made by normal traversal and current-policy evaluation now use the explicit [regret reader](../../liars_poker/algo/regret_readers.py). A neural trainer reads its networks; a table fork reads its table and optionally seeds unseen rows from the frozen network. This removes both table subclasses' `_forward` interception. The existing network models and optimizers are retained for checkpoint compatibility.

The [CPU checks](../../tests/test_cfr_plus_refactor.py) cover the old NumPy and Torch target formulas, table seeding, direct sampled-advantage updates, streamed and non-streamed traversal, and neural, table-fork and discount-table checkpoint round trips. Nine checks passed locally. They do not establish parity on CUDA or over a long training curve. The refactor has not been deployed to a VM.

The remaining proposal is too broad for a behaviour-preserving cleanup now:

- Adding `old_raw` to every traversal record would increase peak memory on the 69-claim game. A single target calculation gives us a place to change the neural rule later without paying that cost now.
- A full `RegretUpdateRule` object and `RegretStore` ownership interface would require changing fitting, checkpoint loading and audit scripts together. The flags still describe distinct choices, including loss weighting. A single rule object should be introduced only when a new algorithm needs it and its semantics are clear.
- Fitting remains with the existing trainer and table subclasses. Moving it behind another abstraction should wait until a concrete new fitting rule requires it. Replacing all runners with a config engine would be a separate migration.
- Keep the recursive backend and historical import paths. The recursive path is still referenced by June notebooks. Renaming a module or backend adds compatibility work without helping the current experiments.
- Keep the existing CUDA guard until a memory model for grouping exists. Replacing a known bound with a guess could cause another out-of-memory run.

This local change does **not** implement neural discounted CFR. The proposed six-arm tabular discount experiment can still use the same runner, but it should be reviewed separately: its table update rules operate on sampled conditional advantages and its average is a fitted neural policy, unlike the bridge's exact average. No code has been deployed to the VM by this refactor.

## Original larger proposal (deferred)

The sections below describe the initial design for future reference. In particular, their "implementation needed" and equivalence-test lists are proposals, not completed work.

## 1. Scope

**In scope**

- `liars_poker/algo/deep_cfr_plus.py`: `DeepCFRPlusTrainer`, its buffers, target construction, fitting and checkpoints.
- `liars_poker/algo/neural_cfr_plus_gpu.py`: the batched traversal.
- `liars_poker/algo/cfr_plus_tabular_fork.py` and `liars_poker/algo/cfr_discount_tabular.py`: table-based experimental variants.
- The `scripts/run_cfr_plus_18_*` runners and scripts that reach into trainer internals.

**Out of scope: do not change**

- The exact solvers and best responses: `cfr_exact_dense.py`, `cfr_plus_dense.py`, `br_exact*.py`, `br_exact_dense_to_dense.py`, and `DenseTabularPolicy`. Their behaviour and interfaces stay as they are. The existing `increment_transform` hook in `CFRPlusDense._update_player`, used by the tabular bridge, stays.
- Approximate best responders, evaluation code, policy classes and `serialization.py` formats.
- The average-strategy learning: its reservoir, loss and weighting options, including the new `quadratic`. It may move with the trainer but must behave identically.
- `deep_cfr.py` (original Deep CFR), except for any shared buffer code it exports.

## 2. Preconditions

1. **Commit the working tree.** Everything since `bfd16d0` is uncommitted: clip on read, quadratic weighting, the discount trainer, runners and docs.
2. **Resolve the local/VM difference.** On 30 September the VM's `deep_cfr_plus.py` and `neural_cfr_plus_gpu.py` lacked the local quadratic-weighting change, and the VM had no `cfr_discount_tabular.py`. Record which revision each live VM run uses in its manifest before deploying refactored code.
3. **Do not deploy refactored code under a running experiment.** Live runs resume from checkpoints using whatever code is on the VM. Refactor locally, prove equivalence (section 6), then deploy between runs.

## 3. How the code is organised today

| File | Responsibility | Notes |
| --- | --- | --- |
| `deep_cfr_plus.py` (about 1,580 lines) | `RecentBuffer` and `DeviceRecentBuffer`. `DeepCFRPlusTrainer`: configuration and validation (`_validate_regret_target_mode`), networks and optimisers, recursive Python traversal (`_traverse`), target grouping (`_aggregate_regret_targets`, including visit-count reach modes), fitting (`_train_model`), `run_iteration`, checkpoints (`CHECKPOINT_VERSION = 2`) | Algorithm, storage, traversal and I/O in one class |
| `neural_cfr_plus_gpu.py` (about 1,285 lines) | `GPUDeepCFRPlusTraverser`: tensor-batched external sampling, action sampling and inclusion corrections, streamed and non-streamed paths, record accumulation | Runs on **CPU and CUDA**. Selected by `traversal_backend="gpu_native"`. Both paths compute regret targets. |
| `cfr_plus_tabular_fork.py` | `TabularRegretFork(DeepCFRPlusTrainer)`: replaces the regret networks with a table seeded from a network checkpoint | Hard-coded to the 18-claim spec. Intercepts `_forward` when called with a regret network. |
| `cfr_discount_tabular.py` | `TabularDiscountTrainer(TabularRegretFork)`: zero-initialised table with CFR, CFR+, DCFR+, exact DCFR and visited-only DCFR, with lazy discounting | Recovers the increment as `mean_raw − relu(old)` because the traverser has already built a target |
| `training/deep_cfr_plus.py`, `training/neural_runs.py`, `eval/profile_cfr_plus_branching.py` | Library-level users of the trainer | Must keep working |

Runners that construct trainers: `run_cfr_plus_18_target_order_cpu_overnight.py` (now the general 18-claim CPU runner despite its name), `run_cfr_plus_18_gpu_fit_forks.py`, `run_cfr_plus_18_oens_followups.py`, `run_cfr_plus_18_tabular_regret_fork.py`, `run_cfr_plus_18_tabular_discount.py`, and the three `run_cfr_plus_69_claim_*.py` scripts. Each has its own loop for checkpoints, snapshots, evaluation and resuming.

Scripts that use **private** trainer members, such as `_train_regret`, `regret_buffers`, `_aggregate_regret_targets`, `_forward`, `regret_nets` or `_gpu_traverser`:
- `audit_cfr_plus_18_late_update.py`
- `audit_cfr_plus_neural_targets_cpu.py`
- `diagnose_neural_cfr_plus_cpu.py`
- `run_cfr_plus_18_gpu_fit_forks.py`
- `run_cfr_plus_18_oens_followups.py`
- `run_cfr_plus_18_target_order_cpu_overnight.py`
- the three 69-claim runners
- `shadow_neural_cfr_plus_cpu.py`
- `smoke_cfr_plus_18_cuda_aggregate.py`
- `verify_cfr_plus_18_reach_weighted.py`

## 4. Problems

1. **The regret-target formula exists in three places, plus an undo.** The recursive `_traverse` (around `deep_cfr_plus.py:1061`), the non-streamed traverser path (around `neural_cfr_plus_gpu.py:741`) and the streamed path (around `neural_cfr_plus_gpu.py:1155`) each compute `old`, choose normalized or cumulative units, and decide whether to clip. `TabularDiscountTrainer._train_regret` then subtracts `relu(old)` again to recover the increment. Every new update rule touches all four: clip on read did, and DCFR or a learned baseline would. The traverser should not know what a target is.
2. **Mode flags multiply.** `regret_target_mode` × `regret_accumulation_mode` × `regret_increment_reach_mode` × `regret_positive_weight` × device and spec gives many combinations, several invalid. Validation is now centralised, which is an improvement. The underlying problem is that one conceptual choice, "which update rule", is spread over five parameters.
3. **Tables are attached by interception.** The fork recognises regret networks by object identity inside `_forward`, and replaces `_train_regret`. The discount trainer layers lazy discounting on top of the same interception. This works, but depends on internal call paths staying unchanged. It also forces the fork to inherit neural fitting machinery it does not use.
4. **Names mislead.** `neural_cfr_plus_gpu.py`, `GPUDeepCFRPlusTraverser` and `traversal_backend="gpu_native"` all mean "batched tensors on any device". The general 18-claim runner is called `..._target_order_cpu_overnight.py` and gains arms through an `extra_arms.json` side file. The spec-specific CUDA guard (`_CUDA_AGGREGATE_SPEC`) encodes a memory concern as a game identity.
5. **The recursive backend keeps a third implementation alive.** `traversal_backend="recursive"` appears to be used only by `notebooks/june_2026/deep_cfr_batched_gpu_validation.ipynb`. It duplicates traversal and target logic. It cannot run `aggregate_then_clip`, the mode used by the main cumulative runs, and it has not been exercised by any September experiment.
6. **Runners duplicate infrastructure.** Around five 18-claim runners and three 69-claim runners each implement atomic checkpoints, snapshots, resume checks, manifests and timing. Several reach into private trainer state.
7. **Experimental harnesses sit beside library solvers.** The fork and discount trainers are 18-claim diagnostic hybrids that depend on trainer internals. In `algo/` they look like peers of the exact solvers, which they are not.

## 5. Target design

### 5.1 The traverser emits ingredients, not targets

At each traverser decision, the traversal records a **regret sample**:

| Field | Meaning |
| --- | --- |
| `features`, `legal_mask` | As now |
| `old_raw` | The regret store's raw output at this information set before this update, i.e. the value currently fed through `relu` |
| `advantage` | Sampled conditional advantage `action_values − node_value`, masked to legal actions |
| `weight` | Inverse path inclusion probability, as now |

The iteration number is known to the trainer. The traverser no longer reads `regret_target_mode` or `regret_accumulation_mode`. Strategy records are unchanged.

**Storage cost.** Storing `old_raw` adds one action-width float column per row. That is negligible at 18 claims. At 69 claims (70 columns, float32) it adds about 280 MB per million rows. If that becomes a problem, `old_raw` could instead be recomputed at fit time from a frozen copy of the pre-update network, which is what VR-DeepDCFR+ does. Start with stored `old_raw`, because it makes the equivalence with today's targets exact and easy to test.

### 5.2 One update-rule object

A `RegretUpdateRule` (a small class or frozen dataclass) owns everything the scattered flags own now:

- how `old` is read, normally `relu(old_raw)`;
- how old and new combine: normalized `((t−1)/t)·old + advantage/t`, cumulative `old + advantage`, or discounted variants;
- where the clip goes: each record, per group of identical information sets, or on read;
- whether grouping is required, including the visit-fraction and visit-count reach modes, which need group counts;
- the loss entry weighting it requires: plain MSE for clip on read, the positive-entry weight otherwise;
- its own validation of compatible devices, stores and reach modes.

Existing configurations map onto named rules, and the constructor keeps accepting the current keyword arguments by translating them into a rule. Every existing runner and checkpoint therefore keeps working unchanged. Table-only rules (vanilla CFR, CFR+, DCFR+, exact and visited-only DCFR) live in the same module so their formulas are defined once. Whether the neural and table versions can share code beyond the formulas is for the implementer to decide.

### 5.3 A regret-store interface

```text
RegretStore
    values(pid, features) -> raw regret outputs      # used by traversal and policies
    update(pid, samples, rule, iteration) -> stats   # fit or write
    state_dict() / load_state_dict()
```

- `NetworkRegretStore` owns the regret networks, optimisers, recent buffers, grouping and `_train_model`'s regret branch.
- `TableRegretStore` owns the 18-claim table key encoding and optional seeding from a network checkpoint; this is today's fork. It also owns lazy discount state for the discount rules.

The trainer composes a traverser, a regret store, the unchanged average-strategy learner and a rule. `run_iteration()` keeps its signature and returned diagnostics. This removes the `_forward` interception: the traverser asks the store for values, whatever the store is.

### 5.4 Renames, with compatibility

- `neural_cfr_plus_gpu.py` becomes, for example, `batched_traversal.py` with class `BatchedCFRPlusTraverser`. Keep a thin re-export module under the old name.
- `traversal_backend="gpu_native"` becomes `"batched"`. Accept `"gpu_native"` as an alias forever, because checkpoints store it.
- Replace `_CUDA_AGGREGATE_SPEC` with an explicit estimate of grouping memory. Allow CUDA grouping when it fits and raise a clear error when it would not.
- Remove the recursive backend, once it is confirmed that nothing but the June validation notebook uses it. Mark that notebook historical, or pin it to the old commit.

### 5.5 Placement of experimental code

Create `liars_poker/experimental/`, or a similarly named package, for spec-specific diagnostic harnesses: the table fork and the discount trainer, once they are expressed through `TableRegretStore`. The exact solvers stay in `algo/`. Before moving any class, check whether any checkpoint pickles its import path. The fork and discount checkpoints appear to store only tensors and dictionaries, but confirm this by loading a real checkpoint from the VM. If any does pickle a path, keep a shim.

### 5.6 One runner with shared infrastructure

- Extract a shared helper for atomic checkpoints and state files, rolling snapshots, measured-time budgets, resume validation, manifests with git revision and source hashes, and non-finite-loss checks.
- Replace the 18-claim runners with one config-driven runner. An arm is a JSON or YAML file giving spec, trainer settings, rule, store, device, budgets, snapshot intervals, and optionally a source checkpoint with overrides for forks. The overrides are what `extra_arms.json`, the fit-fork runner's `trainer.regret_train_steps = arm` and `set_regret_target_mode` do today.
- Keep the old runners runnable until their live experiments finish, then move them to `scripts/archive/`. The 69-claim runners can be ported later or archived; they are not on the current path.
- Give audit scripts public accessors on the trainer rather than private members: "collect one update's samples without fitting", "fit on given samples", "evaluate regret outputs for these features".

## 6. Equivalence tests: write these first

There is no `tests/` directory today. Before changing code, add `pytest` tests that record **golden outputs from the current code**. Run each after every phase.

1. **One iteration per configuration.** Use a fixed seed, CPU, and a small checkpoint on the six-claim spec (`ranks=3, suits=2, hand_size=1`), plus one on the 18-claim spec. Cover at least:
   - `clip_each_record`, `aggregate_then_clip` and `clip_on_read` in both normalized and cumulative units, where valid;
   - `visit_fraction` and `visit_count`;
   - uniform, linear and quadratic strategy weighting;
   - streamed and non-streamed traversal;
   - full expansion and one action-cap setting.

   Compare buffer rows (features, targets, masks, weights), the regret and strategy losses, and the post-fit parameters: exactly on CPU, and within tolerance on CUDA.
2. **Tables.** For each discount-trainer rule, run 20 iterations from zero. Compare the table rows and `last_discount_iteration`. Compare lazy against eager discounting on a small table to floating-point tolerance. Run the fork from a small seeded checkpoint.
3. **Checkpoints.** Load a trainer checkpoint at version 2, a fork checkpoint (format 3) and a discount checkpoint (format 1) produced by the old code, including at least one real VM checkpoint. Save them with the new code and reload. Continue one iteration and compare with the old code's continuation.
4. **Public behaviour.** `current_policy_dense()`, `average_policy()` and `NeuralRegretMatchingPolicy` outputs are unchanged for a fixed checkpoint.

Golden files should be small. Store hashes or compressed arrays under `tests/golden/`, generated by a script that is itself committed.

## 7. Phases

Each phase ends with every test in section 6 passing and with no change to any learning curve.

| Phase | Work | Risk |
| --- | --- | --- |
| 0 | Commit; record VM revisions; add golden tests against current code | None |
| 1 | Traverser emits ingredients; a `RegretUpdateRule` builds targets; constructor keyword arguments translate to rules; delete the three duplicated target blocks and the subtract-`relu(old)` workaround | Medium: touches both traversal paths |
| 2 | Add `RegretStore`; move networks and fitting into `NetworkRegretStore`; re-express the fork and discount trainers with `TableRegretStore`; move them to the experimental package | Medium: checkpoint compatibility |
| 3 | Renames with aliases; memory-based CUDA guard; remove the recursive backend | Low |
| 4 | Shared runner infrastructure and the config-driven runner; public accessors for audits; archive superseded runners | Low for algorithms, moderate effort |
| 5 | Documentation: update the [code map](neural_cfr_plus_code_map.md), the [clip-on-read note](clip_on_read_regret_targets.md) and the [neural CFR+ guide](neural_cfr_plus_from_exact.md) to the new names | None |

## 8. What this makes easier afterwards

- **Neural DCFR variants** become a new rule. Clip on read already stores signed values, and a DCFR rule would change only how `old` is read and discounted.
- **The learned history baseline** changes how the traverser computes `advantage` and node values, at opponent and traverser nodes. With targets gone from the traverser, that is a traversal change only. See [learned history baselines](learned_history_baselines.md).
- **New experiments become configuration files**, not new runners.
