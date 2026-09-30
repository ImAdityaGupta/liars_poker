# 18-claim sampled tabular regret: clipping and discounting

## Question

The neural CFR+ runs often improve quickly and then flatten. Before adding another rule to a neural network, we want to know whether a different **regret update** or **average-policy weighting** helps when the regrets themselves are stored exactly in a table. This removes regret-network fitting from the comparison while keeping the same noisy traversal signal.

The outcome is **exact exploitability of the learned average policy** at matched CFR iterations and measured training minutes. Lower is better. We will use one seed (17) for each arm. A small difference is a lead for another experiment, not a reliable ranking across random seeds.

## What is sampled

This uses the 18-claim game: `ranks=4, suits=4, hand_size=2`, with `RankHigh`, `Pair`, `TwoPair`, and `Trips`, and suit symmetry. Each iteration samples 4,096 root deals for each traverser, samples opponent actions, and fully expands the traverser's actions. At a visited information set, the update is the mean sampled **conditional advantage** over that iteration's visits. At an unvisited information set, the sampled increment is zero.

This is the regret update of **arm 4 (`conditional`)** in the [tabular bridge experiment](2026-09-29_01-42-50_18_claim_tabular_bridge.md): reach is used only as a visited/not-visited gate, with no estimated reach multiplier. The regret table starts at zero. All arms store **cumulative regret**: there is no division by iteration `t`, and no `N/K` multiplier. Regret matching reads the positive part of the stored row, including for rules that retain negative entries.

One important difference from the bridge remains: this faster runner fits a **neural average-policy network** from strategy records. The bridge computed the average exactly. A gap between this experiment and bridge arm 4 could therefore come from average-policy fitting, the different number of roots, or other implementation details. The experiment does not isolate the averaging step.

## Six arms

Write `g_hat` for the mean conditional advantage at a visited information set. The following rule applies once per iteration to a visited row. `t` is the CFR iteration number.

| Arm | Regret rule | Weight of iteration `t` in the average |
| --- | --- | --- |
| V: vanilla CFR | `R <- R + g_hat`; keep negative entries | `1` |
| A: CFR+ | `R <- max(0, R + g_hat)` | `t` |
| B: CFR+, quadratic average | Same regret rule as A | `t^2` |
| C: DCFR+ | `R <- max(0, d_t R + g_hat)`, where `d_t = (t-1)^2 / ((t-1)^2 + 1)` | `t^2` |
| D: DCFR, exact decay | Multiply **old** positive entries by `(t-1)^1.5/((t-1)^1.5+1)` and old negative entries by `1/2`, then add `g_hat` | `t^2` |
| E: DCFR, visited-only decay | Same update as D on visited rows | `t^2` |

For C and D, **discounting still occurs on iterations with no visit**. We calculate it lazily when a row is next read or updated; this should equal eager discounting of every row each iteration. For E, both the increment and the discount happen only on a visit. This tests whether the sparse-update shortcut changes the result. V, A and B leave unvisited rows unchanged.

The [DCFR paper](https://arxiv.org/abs/1809.04040) motivates separate positive and negative discounts and quadratic averaging. The [VR-DeepDCFR+ paper](https://arxiv.org/abs/2511.08174) motivates the positive-only discount in C. Their published results do not predict the outcome here: our sampled conditional increments omit the counterfactual reach magnitude.

## How to interpret the comparisons

| Compare | Main difference | What an improvement would suggest |
| --- | --- | --- |
| V vs A | Clipping and average weighting both change | The CFR+ package helps under this sampling scheme; the comparison alone cannot identify which part. |
| A vs B | Average weights only | Give later strategy records more weight in a neural follow-up. |
| B vs C | Discount old positive regret | Old sampled regret is worth forgetting faster. |
| B vs D | Signed regret and sign-specific discount | Keeping negative evidence may be useful. |
| D vs E | Discount all rows versus visited rows | If D wins, a neural implementation needs a way to decay predictions at unvisited states. |

The average is itself fitted from a 2,000,000-record strategy replay buffer, so single snapshot differences may reflect average-network noise. Compare trajectories and windowed values, and use a second seed or an exact-average confirmation if two arms are close. A tabular win is not yet a neural win: a network must also fit the chosen state over many iterations.

## Run design

- **Seed:** 17 for every arm, with the same initial network and traversal RNG seed.
- **Regret table:** zero at the start of each arm; one row per 18-claim information set. The regret networks are instantiated by the existing trainer but are not fitted or used for regret prediction.
- **Strategy fit:** 256 by 256 MLP, six steps per iteration, batch 1,024, learning rate `1e-3`, two-million-record replay buffer.
- **Budget:** 180 measured training minutes per arm, six arms, roughly 18 CPU training hours if run sequentially. Exact evaluation and checkpoint time are outside this budget. Actual iteration counts depend on VM load and the rule.
- **Monitoring:** exact exploitability of the saved average policy every 15 measured minutes; a rolling resumable checkpoint at the same interval; training and evaluation JSONL logs; comparison graph with logarithmic exploitability on the vertical axis.
- **Pause/resume:** the runner checks a `PAUSE` file between iterations and saves a checkpoint before exiting. Starting it again in the same output directory resumes incomplete arms.

The implementation is in [the tabular discount trainer](../../liars_poker/algo/cfr_discount_tabular.py) and [the runner](../../scripts/run_cfr_plus_18_tabular_discount.py). The runner can select one arm or run the six sequentially. **The experiment has not been launched.**

The neural CFR+ cleanup now builds neural targets in [one shared function](../../liars_poker/algo/cfr_plus_targets.py) for all three traversal paths. The tabular discount trainer uses the same regret-record hook but stores conditional advantages directly. This avoids recovering `g_hat` by subtracting old regrets, with no extra buffer column or checkpoint migration. The runner's outputs and table update rules are unchanged. Before using the results to choose a neural discount rule, confirm that the tabular curves remain sensible and remember that the average policy here is learned, not exact.

## Results

Pending. No conclusion about which rule improves exploitability yet.
