# GPU neural CFR+: fitting steps, learning rate and schedules

**Status (30 September 2026): open questions and a proposed plan, no results yet.** Neural CFR+ now runs about eight times faster per iteration on the VM's GPU than on CPU. That leaves several training settings we have never chosen deliberately. This note:

- sets out what is known;
- gives the reasoning that should guide the choices;
- proposes cheap diagnostics to run **before** any long training runs.

It builds on the [regret fit-steps sweep](../experiments/18_claim/2026-09-30_00-55-10_18_claim_regret_fit_steps_sweep.md), the [tabular discounting screen](../experiments/18_claim/2026-09-30_00-55-09_18_claim_tabular_discounting.md) and [from exact CFR+ to neural CFR+](neural_cfr_plus_from_exact.md).

The open questions:

1. How many average-strategy fit steps per iteration, and should that change over training?
2. How many regret fit steps per iteration, and should that change?
3. How do those two interact?
4. Should the learning rate change over training?
5. Should traversals per iteration change over training, and how do we move from fully expanding traverser actions to sampling them? This matters most at 69 claims and is only sketched here.

## 1. Average-strategy fitting can be separated from training

### The key fact

The average-strategy network is **never used during training**. Traversal plays the *current* strategy, obtained by regret matching on the regret networks. The average network only matters for the policy we evaluate or deploy. So fitting it online, every iteration:

- does not change the regret trajectory at all;
- costs time;
- produces a policy whose quality depends on how well a network can follow a moving target from a small fraction of the data.

Currently each iteration adds roughly 100,000–400,000 strategy records: about 92,000 for one player and 400,000 for the other at 4,096 roots. They go into a 2,000,000-record reservoir, and the network takes 6 Adam steps of 1,024 records. It sees a few percent of each iteration's new data.

The alternative is to keep collecting the reservoir and fit the average network **only when a policy is needed**: from scratch, or warm-started, for as many steps as it takes to converge. This is a proposal for our implementation. The [VR-DeepDCFR+ paper](https://arxiv.org/pdf/2511.08174) reports 5,000 average-network steps with batch size 2,048 on a 1,000,000-record reservoir, but its stated algorithm trains the average network within each CFR iteration; it does not establish an every-three-iterations offline-refit schedule.

This removes question 1 from the joint optimisation. It also mostly dissolves question 3, since the only remaining interaction is that better regrets give better policies to average. What is left is "how much fitting does an offline refit need", which can be answered without any training run.

### Why this matters now

The tabular discounting screen suggests the online average network is costing a lot. Every arm there stores regrets exactly in a table, with the same conditional CFR+ update as bridge arm 4. Interpolated exact exploitability at matched iterations:

| Average | Roots | 1,200 iterations | 2,400 iterations |
| --- | ---: | ---: | ---: |
| Exact | 128 | 0.036 | — |
| Exact | 256 | 0.023 | — |
| Exact | 512 | 0.019 | — |
| Exact | 1,024 | 0.016 | 0.011 |
| Neural, online | 4,096 | about 0.020 | 0.018 |

With an exact average, more roots help steadily. The neural-average arm has four times the bridge's roots, yet per iteration performs like the 256–512-root exact runs. By about 16,000 iterations it is roughly twice as exploitable as the bridge's extrapolated curve. The traversal paths were checked to produce identical records on fixed deals, but the runs also differ in root count. Average fitting is a plausible cause of the gap, not yet an isolated explanation.

Two controls launched on 30 September test this directly:

- **`exact4096`:** exact average, 4,096 roots. It should match or beat the bridge if averaging is the cause.
- **`neural1024`:** neural average, 1,024 roots.

Both run under `artifacts/cfr_plus_18_batched_bridge_controls/main_20260930` on the VM. Snapshot noise also matters here. Learned averages scatter by about ±15% between snapshots, and in the discounting screen the five CFR+ variants bunched together in a way consistent with a common averaging floor hiding differences between regret rules.

The neural average is not a hard floor. The tabular regret fork used it and still reached about 0.0035 by 72,000 iterations. But if it roughly halves progress per iteration, part of the neural runs' apparent plateau near 0.0075 may be averaging rather than regret learning.

### Proposed diagnostic: an offline average-network study

The [standalone experiment plan](../experiments/18_claim/2026-09-30_21-47-00_18_claim_offline_average_fitting.md) uses the exact average, online network, and strategy reservoir from **the same** `exact4096` checkpoint. This removes the root-count and traversal differences from the main comparison.

Trainer checkpoints save both players' strategy reservoirs (`strategy_buffers` in `checkpoint_dict`), along with the online network. The `exact4096` checkpoint additionally saves the exact accumulated average. Begin with its preserved 30-minute checkpoint: compare the exact policy, saved online network, a warm refit, and a fresh refit at fixed architecture and loss. Expand fit steps only if the first screen improves; compare later checkpoints from this same run when available. The live trainer overwrites its checkpoint every 15 training minutes, so later stages must be copied before their next overwrite.

This isolates the **policy-extraction** question without changing the regret trajectory. If refitting still leaves a large gap to the exact average, more optimizer steps alone are insufficient; reservoir coverage, model capacity, target weighting and the loss remain separate possibilities. A larger reservoir's GPU memory cost depends on feature and action dimensions and must be measured before use, especially at 69 claims.

## 2 and 4. Regret steps and learning rate are one question

With Adam, each step moves parameters by roughly the learning rate, fairly independently of target scale. The total change a fit can make is therefore governed by **steps × learning rate**, while the noise left in the fit is governed mainly by the **learning rate**. Choosing them separately risks attributing to one what the other did.

### What the target units imply

- **Normalized units** (`((t−1)/t) old + g/t`): targets shrink as training proceeds, so a fixed learning rate becomes relatively coarser. This is consistent with the June result that dropping the learning rate helped normalized runs, from about 0.024 to about 0.014.
- **Cumulative units** (`old + g`): positive regrets grow, so a fixed learning rate becomes relatively *finer*, an implicit annealing. This is consistent with cumulative runs doing well without any schedule, and it means the June learning-rate result may not transfer.
- **Late training:** at low exploitability, the current policy depends on small differences between action regrets. A lower noise floor (smaller learning rate) and more precise fitting should matter more late than early. The prior is a learning rate that decays late plus a modest increase in steps, but there is no direct evidence in cumulative units.
- **A principled alternative:** normalise regret targets by a running scale estimate, so outputs stay near size 1. This is the "PopArt" idea (Van Hasselt et al., 2016), which the VR-DeepDCFR+ paper cites for exactly this problem of values spanning orders of magnitude. A learning-rate schedule would then only need to handle noise.

### What has been measured

- 24, 96 and 384 regret steps per iteration showed no clear difference in exact average exploitability. However, the runs ended at 23,700–36,500 iterations. The same-checkpoint tabular fork, the infinite-fit reference at visited information sets, improved clearly only **after about 40,000 iterations**. The sweep was too short to settle the question. Longer continuations are planned in the fit-steps note.
- In trend, S96 looked no better than S24: slope +0.05 against −0.24 over their shared range. This is weak one-seed evidence that harder fitting may disturb predictions at the roughly 99% of information sets not visited in a given update.

### Proposed diagnostic: one-iteration fit measurements across training stages

From checkpoints at an early, a middle and a late stage, run a single regret update under a small grid:

| Variable | Values |
| --- | --- |
| Regret steps | 24, 96, 384 |
| Learning rate | `1e-3`, `3e-4`, `1e-4` |

For each cell, measure:

- **Held-out fit error:** fit on one traversal's targets and score against an independent traversal of the same frozen policy. A gap between training and held-out error means the network is fitting sampling noise.
- **Distance to the exact one-step target:** exact conditional advantages from the dense solver, compared with the network after fitting (the E–N distance of the [late-update audit](../experiments/18_claim/2026-09-29_13-47-16_18_claim_late_checkpoint_update_audit.md)), in cumulative units.
- **Drift at unvisited information sets:** change in the regret-matched policy at information sets *not* in the buffer, weighted by exact reach. A table has zero drift by construction; this is the interference hypothesis measured directly.

Each cell takes seconds to minutes on the GPU. The results show which settings fit visited information sets well **without** disturbing unvisited ones, and whether the best setting shifts with training stage, which directly answers "should it change over time". Only the two or three most promising schedules then need long runs.

## Cost on the GPU

On the RTX 4060 Ti, an S24 iteration at 4,096 roots takes about 0.33 s:

| Phase | Time per iteration |
| --- | ---: |
| Traversal | about 0.22 s |
| Regret fitting (24 steps, both players) | about 0.09 s |
| Strategy fitting (6 steps, both players) | about 0.02 s |

On CPU, traversal was about 2.1 s of 2.7 s. Fitting is now cheap relative to traversal. At the current 512×512 regret networks:

- doubling regret steps costs roughly 25% more time per iteration;
- strategy steps are nearly free.

Larger networks change this: a 2048×2048 step took about 5 ms against about 1 ms. The balance has moved towards more fitting and more roots than the CPU-era defaults. That is only worth paying for where the diagnostics above show a benefit.

## 5. Traversals over time and traverser-action sampling (sketch)

- **More roots later.** On the GPU, 4,096 roots traverse in about 0.2 s, so growing the root count over training is cheap. It reduces target noise as the per-iteration signal shrinks, and at 30 claims it raises how often each visited information set repeats, from 1.6 to 2.4 rows per information set when going from 1,024 to 4,096 roots. Test it as one extra arm in the long runs below.
- **Sampling traverser actions.** Needed at 69 claims, where fully expanding every claim is costly. Earlier results that looked bad for sampling all used per-record clipping, which turns the extra variance of inverse-probability estimates directly into bias. The comparison should be repeated under cumulative targets with grouping or [clip on read](clip_on_read_regret_targets.md). The principled fix, if claim caps still lose, is a [learned history baseline](learned_history_baselines.md). At 30 claims, full expansion is still affordable, so sampling there is rehearsal for 69.

## Proposed order

1. **Offline average-network study** on existing checkpoints (section 1). This decides whether future comparisons should use online averages at all.
2. **One-iteration regret-fit diagnostics** at several training stages (section 2). This narrows regret steps and learning rate to a few candidate schedules.
3. **A few long GPU runs** of those candidates, with each snapshot's average refit offline, so that averaging noise does not hide regret differences. Include one growing-roots arm.
4. **Sampling** at 30 claims as rehearsal, then 69 claims (section 5).

Each step is cheaper than the next and narrows what the next must test. Until step 1 is done, treat small differences between neural runs' exploitability as possibly caused by the average network rather than the regret update.
