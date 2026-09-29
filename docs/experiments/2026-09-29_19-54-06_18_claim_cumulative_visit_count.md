# 18-claim neural CFR+: cumulative conditional updates multiplied by visits

## Question

The [cumulative-regret comparison](2026-09-29_11-14-31_18_claim_cumulative_regret_scale.md) found that the conditional update `clip(old + mean(G))` learned well, while `clip(old + (N/K) mean(G))` barely learned. Does keeping the visit-count variation while removing the small `1/K` scale recover useful learning?

## Method

At each visited information set, let `N` be the number of sampled roots that visit it, `K=4,096` the roots per player update, and `mean(G)` the mean sampled conditional action advantage. The new target is:

`clip(old + N × mean(G))`

The earlier comparison uses `clip(old + mean(G))` (conditional) or `clip(old + (N/K) × mean(G))` (visit fraction). A set with `N=0` has no training target. The new target is the sum of its sampled fresh advantages added to the previous **cumulative** regret prediction, followed by one clipping operation. The trainer retains the same 18-claim game, seed 17, full action expansion, 4,096 roots, 512-by-512 regret network, 256-by-256 strategy network, 24 and 6 fit steps, and four-million-row regret buffer as the cumulative conditional run. As in the previous arms, a visited information set appears `N` times in the fitting buffer, so fitting also gives it more weight.

This is not a pure test of whether reach weighting is theoretically useful: target magnitudes and fitting behavior change. It is a practical test of whether the neural model can learn in cumulative units when sampled visits set the relative scale of regret increments. Exact tabular regret matching would be insensitive to a common positive rescaling of all regrets; limited neural fitting need not be.

## What different results would mean

- If `N` learns substantially better than `N/K`, the tiny target scale was an important part of the earlier failure.
- If `N` beats conditional per iteration and per training minute, using visit frequency may improve the learned policy on this game.
- If `N` is unstable or worse than conditional, its larger and more uneven targets may be difficult for the current network and fixed fitting budget. This would not by itself show that reach is unimportant.
- If both conditional and `N` improve early but later deteriorate, the visit multiplier has not resolved the long-term plateau.

The primary outcome is **exact average-policy exploitability**, shown on port 8765 against both measured training minutes and CFR+ iteration. Policies are saved every 15 measured minutes and evaluated by the independent dashboard. The runner keeps a rolling full checkpoint every 15 measured minutes and logs iteration costs and losses. Its target is 600 measured training minutes; evaluation and checkpoint time are excluded from that budget. The arm is saved under `artifacts/cfr_plus_18_cumulative_regret/main_20260929/n4096` on the VM. [The launcher](../../scripts/launch_cfr_plus_18_cumulative_visit_count.sh) can resume it after an interruption.

## Results

Launched on the CPU VM at 19:57 UTC on 29 September 2026 in tmux session `cfr18_n4096`. The first minute reached iteration 17 with finite fitting losses; the larger regret loss scale is expected from the larger target magnitude and is not itself an exploitability result. The new arm appears on port 8765 with a 600-minute target. Exact evaluation will start at the first 15-minute snapshot.

The current cumulative conditional arm continues independently toward its 930-minute total target. Compare both at matching minutes and iterations. They begin from seed 17 but diverge after the first regret update. No exploitability conclusion is available yet.
