# 18-claim O/E/S/N audit and tabular-regret fork

## Questions

This work asks two connected questions about neural CFR+ on the 18-claim game:

1. As a normal neural CFR+ run continues, do one-step target and fitting errors change in a way that helps explain the average policy's progress or plateau?
2. If the regret networks are replaced by stored tabular regrets late in training, does the average policy keep improving?

The first is a diagnostic of one-step regret updates. The second is a longer intervention on the regret representation. They are separate experiments and should not be read as one controlled comparison.

## O/E/S/N audit

The fresh normal run uses seed 31 and the 18-claim `r4_s4_h2_hp2pt_ss` game. It uses the established neural CFR+ settings, including 4,096 root traversals per player update, aggregate-then-clip regret targets, normalized `1/t` units, 24 regret fitting steps and six strategy fitting steps. Every 15 measured training minutes, the run saves a resumable checkpoint and policy snapshot, measures exact exploitability of the average and current policies, and audits a cloned copy of the next player-1 update. The audit does not change the live trainer or its random-number state.

For each visited information set:

- **O** is the old policy/regret prediction before the update.
- **E** is the ideal one-step target using exact conditional action advantages `g`, while retaining the old regret prediction.
- **S** is the target from the normal sampled traversal data.
- **N** is the regret network after fitting S.

The comparisons separate the local stages: O→E is the intended update, E→S is sampling error, and S→N is fitting error. E→N compares the final fitted result with the exact one-step target. The audit also reports total-variation and KL distances with equal, reach, and `1/[-log(q)]` weights. These are local policy distances at one update, not direct exploitability estimates. A low distance does not guarantee that strategically important states are weighted enough, and one seed cannot establish a predictive relationship.

### Normal-run results through 360 minutes

The planned initial run completed 360 measured training minutes at iteration 8,727. Its exact average-policy exploitability was 0.02212 and current-policy exploitability was 0.08704. The average had reached 0.02073 at minute 345 and was 0.02564 at minute 330, so the last stretch was not monotonic. The much higher current-policy value reinforces that current exploitability is a noisy companion metric, not a substitute for the average-policy result.

At the minute-360 audit, among visited information sets, equal-weight mean TV was 0.01336 for O→E, 0.00201 for E→S, 0.00682 for S→N, and 0.00609 for E→N. On that audit, the sampled targets were close to exact, while fitting moved the policy partway back toward the old policy; the fitted policy nevertheless remained closer to E than O was. This is a one-update observation. It does not show that the regret network is the cause of the longer-term plateau, nor that the next live update will match the cloned audit.

### Continuation in progress

The same seed-31 trainer was resumed from its 360-minute checkpoint toward 720 **total measured training minutes**. At the latest VM check (29 September 2026, 23:13 UTC), it had reached 367.8 total minutes and iteration 8,890; the 375-minute monitor was still ahead. The continuation runs in tmux session `oens_normal_extend`, with dashboard `oens_dashboard` on port 8767. Its run directory is `artifacts/cfr_plus_18_oens_followups/main_20260929/normal`.

This is a continuation of the original trajectory, not a new 360-minute run. Further O/E/S/N audits and exact evaluations will be appended as the continuation reaches each 15-minute boundary.

## Why the exact-G rescue was retired

An earlier rescue fork started from the seed-17 neural checkpoint at iteration 4,072 (330 minutes). It replaced sampled conditional action values with exact `g` at the same sampled/visited information sets, leaving the other training machinery in place. After 120.6 additional measured minutes it had reached only iteration 4,214. Average exploitability moved from 0.02603 at the source to 0.02261 at total minute 390, then rose to 0.02826 by minute 450. The transient improvement did not persist.

The run was also prohibitively slow: median iteration time was 51.23 seconds, of which 49.14 seconds was in the regret-training phase containing the exact dense calculation, versus 2.10 seconds for traversal and 0.05 seconds for strategy fitting. It generated only 142 additional CFR+ iterations in two hours. With no same-checkpoint ordinary continuation, the result did not isolate whether exact `g` helps. It is best treated as an expensive, inconclusive pilot, not evidence that exact values cannot help. We stopped pursuing it and moved to a more direct test of the regret representation.

## Active follow-up: replace the regret networks with a tabular regret state

The current fork starts from the seed-31 O/E/S/N run's 300-minute checkpoint, at iteration 7,417. The source average policy's exact exploitability was 0.02224. The fork keeps the neural strategy/average-policy network and its replay state, but replaces the regret networks with a lazy tabular store:

1. On first lookup of an information set, initialize its table row from the frozen source regret network's prediction.
2. At each iteration, collect the usual 4,096-root sampled regret records.
3. For each visited information set, aggregate its conditional targets, then clip once and store the updated regret vector in the table. The source settings use normalized regret units. Unvisited entries stay at their frozen source predictions.
4. Continue training the strategy network and evaluating the exact average policy every 15 fork-training minutes.

This is **not exact tabular CFR+** and does not use exact `g` or exact reach. It keeps sampled conditional updates and the neural average. It asks whether removing repeated regret-network fitting and prediction drift helps the late trajectory when regrets are stored explicitly. If it improves, that implicates some part of the regret-network approximation/fitting loop, but it will not distinguish those components by itself. If it does not, the remaining causes include sampled updates, the normalized `1/t` scale, coverage, and the learned average.

The fork is configured for 600 additional measured minutes, with exact average-policy evaluation and a resumable checkpoint every 15 minutes. At the 120-minute evaluation it had reached iteration 19,781 and exact average exploitability 0.00584, down from 0.02224 at the source checkpoint. The latest training log at the VM check had advanced to 131.7 minutes and iteration 20,987; its next exact evaluation/checkpoint was due at 135 minutes. Recent iterations took about 0.6 seconds, with roughly 0.53 seconds in traversal, 0.01 seconds in the tabular regret update and 0.05 seconds in strategy fitting. This is a strong early result, but it is one continuation, not yet evidence about the full ten-hour curve or the specific reason it improved.

The fork runs in tmux session `cfr18_tabular_fork`, with its average-policy curve overlaid on the cumulative dashboard on port 8765. Its VM directory is `artifacts/cfr_plus_18_tabular_regret_forks/oens_0300m`. The source checkpoint is preserved separately in that directory.

## How to read the combined evidence

- The minute-360 O/E/S/N audit found that, on visited sets, this sampled update was close to the exact-`g` target, and network fitting did not erase the entire update. That weakens the simple story that sampled `g` or one-step fitting error alone explains the neural plateau.
- The exact-G rescue produced too few iterations and no lasting improvement, so it was not a useful long-run test.
- The tabular-regret fork is more informative about the regret representation and is currently improving quickly. Because it retains sampled traversal and neural averaging, its success would show that a full neural regret model is not necessary for this late-stage continuation; it would not establish that the other approximations are harmless at larger game sizes.
- Compare fork and source at both iteration and measured time, and remember their policies diverge after the fork. The source checkpoint is the starting point, not a concurrent control.

## Files and run records

- Normal/OENS run and audits: `artifacts/cfr_plus_18_oens_followups/main_20260929/normal/`
- Retired exact-G pilot: `artifacts/cfr_plus_18_oens_followups/main_20260929/exact_g/`
- Active tabular-regret fork: `artifacts/cfr_plus_18_tabular_regret_forks/oens_0300m/`
- Implementation: [`cfr_plus_tabular_fork.py`](../../liars_poker/algo/cfr_plus_tabular_fork.py) and [`run_cfr_plus_18_tabular_regret_fork.py`](../../scripts/run_cfr_plus_18_tabular_regret_fork.py)
