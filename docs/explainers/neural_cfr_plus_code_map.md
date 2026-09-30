# Neural CFR+ code map

This is a reading guide to **this repository's implementation**. For the algorithmic derivation and its approximations, see [from exact CFR+ to neural CFR+](neural_cfr_plus_from_exact.md). Start with one call to `run_iteration()` and follow the functions it calls; the run scripts choose parameters but do not implement the update.

## Read one iteration in this order

| Code | What to look for |
| --- | --- |
| [`GameSpec`](../../liars_poker/core.py) and [`rules_for_spec`](../../liars_poker/env.py) | The deck, available claims, legal moves, and payoff rules. |
| [`InfosetEncoder` and `NeuralMLP`](../../liars_poker/policies/neural.py) | Inputs are private-hand rank counts and public claim-history bits. The network outputs one number per action, including `CALL`; a legal-action mask is applied later. |
| [`DeepCFRPlusTrainer.__init__`](../../liars_poker/algo/deep_cfr_plus.py) | All trainer parameters, the two regret networks, the two average-strategy networks, replay buffers, and optimizers. Search for `def __init__` inside `DeepCFRPlusTrainer`. |
| [`DeepCFRPlusTrainer.run_iteration`](../../liars_poker/algo/deep_cfr_plus.py) | The main loop: increment iteration; traverse and fit regrets for player 1, then player 2; finally fit both average-strategy networks. Search for `def run_iteration`. |
| [`GPUDeepCFRPlusTraverser`](../../liars_poker/algo/neural_cfr_plus_gpu.py) | Batched traversal: `_sample_deals`, `_features`, `_regrets_and_strategy`, `_claim_edges`, `_terminal_values`, and `_run_traversals_streaming`. The file name says GPU, but this path also runs on CPU when the trainer's device is CPU. |
| [`make_regret_target`](../../liars_poker/algo/cfr_plus_targets.py) | One target calculation used by recursive, batched and streamed traversal. It reads raw old regret, adds the sampled conditional advantage in normalized or cumulative units, and applies record-level clipping when selected. |
| [`NetworkRegretReader` and `TableRegretReader`](../../liars_poker/algo/regret_readers.py) | Explicit source of current regrets. The table fork uses the table reader, including lazy seeding or discounting, without intercepting `_forward`. |
| [`_train_model`](../../liars_poker/algo/deep_cfr_plus.py) | Masked, positive-weighted squared error for regrets; cross-entropy for strategy records. |

`run_iteration()` clears the **recent regret buffer** before each player's traversal, and fits that player's regret network using this iteration's records. The **strategy reservoir** persists across iterations and stores historical strategy examples. Thus old regret information mostly lives in network weights; old strategy information also lives in replay. For a playable policy, [`NeuralRegretMatchingPolicy`](../../liars_poker/policies/neural_regret.py) uses the regret networks for the *current* strategy, while `NeuralPolicy` in `policies/neural.py` uses the strategy networks for the *learned average*.

## The update to inspect most closely

At a traverser's decision, `_regrets_and_strategy` applies regret matching to the network output: keep positive predictions on legal actions, normalize them, and fall back to uniform legal play if none is positive. Traversal estimates each expanded action's return. All traversal paths then use the same target function:

```python
node_values = (strategy * action_values).sum(dim=1)
instant_regret = (action_values - node_values[:, None]) * legal_mask
targets = trainer.make_regret_record(regret_values, instant_regret, legal_mask)
```

For the historical neural default, the trainer's hook calls `make_regret_target` and computes `relu((t-1)/t * relu(regret_values) + instant_regret/t)` on legal actions. Cumulative and aggregate-first modes use different settings of the same function. The tabular discount trainer instead stores `instant_regret` directly. **For a neural trainer, `regret_values` is yesterday's network prediction, not an independently stored cumulative-regret table.** This is the feedback loop to keep in mind when a run initially improves and later plateaus. The non-streamed traversal is another implementation option in the same file; `traversal_streaming` selects the path.

For comparison, [`CFRPlusDense._update_player`](../../liars_poker/algo/cfr_plus_dense.py) updates explicit tabular regrets with exact game values. Comparing the two makes clear what is sampled, predicted, and clipped in the neural version.

## Parameters: which knob changes what?

The concrete 69-claim configuration is in [`run_cfr_plus_69_claim_adaptive.py`](../../scripts/run_cfr_plus_69_claim_adaptive.py): `SPEC`, `PHASES`, and `TRAINER_KWARGS`. Its phase controller can change learning rate, traversals, action cap, fit steps, and optimizer state. The trainer constructor gives the complete parameter list.

| Parameter | Meaning |
| --- | --- |
| `traversals_per_player` | Sampled starting deals for each player's update in one CFR+ iteration. Passed to `run_iteration`, not the constructor. |
| `traversal_batch_size` | How many of those deals are traversed together. This mainly affects execution and memory. |
| `traverser_action_sample_schedule` | Maximum number of legal *claim* actions sampled at successive traverser decisions. This changes statistical coverage and variance. `CALL` is handled separately. Full traverser-action expansion requires no schedule, count, or fraction cap. |
| `traverser_action_sample_mode`, `traverser_action_baseline` | How claims are selected and how unselected actions are estimated. These can materially change the learning signal. Read `_claim_edges` and the action-value correction in the traversal code. |
| `traversal_streaming`, `traversal_live_row_budget`, `traverser_action_chunk_size` | Which traversal implementation runs and how much expansion is held live or processed per chunk. These primarily control memory and throughput; they do not mean fewer sampled actions by themselves. |
| `batch_size` | Minibatch size for *network fitting*. It is independent of `traversal_batch_size`. |
| `regret_train_steps`, `strategy_train_steps` | Optimizer steps per outer iteration, for each regret net and each strategy net respectively. More steps cost wall time and may improve fit to the available records. |
| `regret_buffer_capacity`, `strategy_buffer_capacity` | Memory for current regret examples and historical strategy examples. The regret buffer is cleared each iteration; merely enlarging it does not create a long-term regret ledger. |
| `regret_positive_weight` | Extra squared-error weight on positive regret targets. |
| `regret_target_mode` | `clip_each_record` is the historical default. `aggregate_then_clip` groups raw updates by information set before clipping; `clip_on_read` retains signed targets and clips predictions when forming a policy. See the [neural clip-order experiment](../experiments/2026-09-28_01-04-50_cfr_plus_neural_clip_order_cpu.md). |
| `regret_accumulation_mode` | `normalized` uses `(t-1)/t` on the old prediction and `1/t` on the new advantage; `cumulative` adds an unscaled advantage. |
| `strategy_weighting` | `uniform`, `linear` and `quadratic` give an iteration's strategy records weights `1`, `t` and `t^2` respectively. |
| `learning_rate`, `regret_hidden_sizes`, `strategy_hidden_sizes` | Optimizer scale and separate network capacities. |

To separate algorithm from configuration while reading, trace one iteration with a small `GameSpec`, `device="cpu"`, and `traversal_backend="gpu_native"`. Check the returned record's traversal time, record counts, action-sampling fraction, and fit losses. The [sampled-target](../experiments/2026-09-27_23-28-05_cfr_plus_sampled_targets_cpu.md) and [shadow-ledger](../experiments/2026-09-27_23-58-30_cfr_plus_shadow_neural_cpu.md) notes explain which of those diagnostics agree with an independent exact reference—and which do not.
