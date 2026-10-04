# Aggregate-then-clip, clip on read, and the hybrid

This note explains, with two small examples, how the three ways of building neural CFR+ regret targets differ. It says when each is right and why a hybrid should combine their strengths. It complements [clip on read](clip_on_read_regret_targets.md), which specifies the implementation, and the [clip-order experiments](../experiments/18_claim/2026-09-28_13-29-00_18_claim_target_sampling_factorial.md) that motivated aggregation.

## 1. What every mode is trying to compute

For one information set `I` and action `a`, CFR+ with sampled increments wants:

```text
new regret = max(0, old + average of this iteration's sampled advantages)
```

Here `old` is the previous regret, read through `max(0, ·)`. The order matters: **average the noisy evidence first, then clip once**. Clipping each noisy sample before averaging throws away negative samples and biases the regret upwards.

The three target modes differ only in **where the clip happens relative to the averaging**, and in **who does the averaging**: explicit grouping code, or the network during fitting.

| Mode | Target stored for each visit | Who averages | Where the clip happens |
| --- | --- | --- | --- |
| `clip_each_record` | `max(0, old + G_k)` | The network | Before averaging, on every sample |
| `aggregate_then_clip` | `max(0, old + mean G)` over visits to the **identical** information set | Grouping code, for identical information sets only | After grouping, before fitting |
| `clip_on_read` | `old + G_k`, unclipped, possibly negative | The network | When the network output is used: regret matching and next iteration's `old` |

## 2. Example 1: repeated visits to the same information set

In one iteration, information set `I` is visited three times. `old = 0.2`, and the sampled advantages for action `a` are `+1.0`, `−0.6` and `−0.8`, whose mean is −0.13. CFR+ wants `max(0, 0.2 − 0.13) = 0.067`.

| Mode | Stored targets | What a perfect fit learns | Result read by regret matching |
| --- | --- | --- | --- |
| Clip each | 1.2, 0, 0 | 0.4 | 0.4: biased upwards |
| Aggregate then clip | 0.067, 0.067, 0.067 | 0.067 | 0.067 ✅ |
| Clip on read | 1.2, −0.4, −0.6 | 0.067 | 0.067 ✅ if the fit really averages |

With a perfect fit, aggregation and clip on read agree. **But our fits are far from perfect.** Each player update takes 24 minibatches of 1,024 rows from about 54,000 rows, under half a pass. For this information set the network may see only one of the three rows:

- **Aggregation:** every row carries the correct average, so any row pulls the network the right way.
- **Clip on read:** each row carries its own noisy sample. The fitted value ends up noisy around 0.067. That noise is then clipped on read, and a clipped noisy value is biased upwards. The noise also carries forward, because the fitted value becomes next iteration's `old`.

So for repeated information sets, aggregation does the averaging exactly and explicitly. Clip on read relies on the optimizer and does it worse.

## 3. Example 2: similar information sets, each visited once

Now suppose 100 **different** information sets are strategically alike, for example the same claim history with 100 similar private hands. For action `a` at each:

- the true advantage is −0.1, a mildly bad action; CFR+ wants its regret to stay at 0;
- one sample is `+0.9` or `−1.1`, 50/50;
- `old = 0`;
- each information set is visited **exactly once** this iteration. This is the typical situation at 69 claims.

**Aggregation never combines different information sets.** Each group has one member, so grouping changes nothing. Every row is clipped on its own: about half the targets are 0.9 and half are 0.

The network cannot memorise 100 separate noisy labels. Because the information sets look alike, it effectively fits their average, about **+0.45**. It now believes this bad action has substantial positive regret throughout that region, and the next strategy plays it. This is the per-record clipping bias again. The averaging happened inside the network, *after* the clip.

**Clip on read** stores 0.9 or −1.1, unclipped. The network averages those to about **−0.1**. Reading `max(0, −0.1)` gives 0, which is correct.

If all 100 visits had been to *one* information set, aggregation would have averaged them exactly and been correct too. For a genuinely isolated single visit, with no similar information sets sharing the network's prediction, the two modes also give the same result: `max(0, stored value)` is 0 either way.

## 4. The whole difference

**Aggregation removes clipping bias only when the repeated evidence comes from identical information sets. When the network does the averaging, across similar but non-identical information sets, the targets must be unclipped for that averaging to be correct.**

| | Evidence repeated at the *same* information set | Evidence spread over *similar* information sets |
| --- | --- | --- |
| Aggregate then clip | Averages exactly, then clips ✅ | Clips first; the network averages clipped values: biased ❌ |
| Clip on read | Relies on the optimizer to average; noisy at our fit budget ⚠️ | The network averages raw values, then the output is clipped ✅ |

Which column dominates depends on the game and root count. From one iteration's traversal under a trained policy at 4,096 roots:

| Game | Player 1 rows whose information set appears in a group of 10 or more | Rows from single-visit information sets |
| --- | ---: | ---: |
| 18 claims | 86% | 5% |
| 30 claims | 31% | 28% (47% at 1,024 roots) |
| 69 claims | not measured; expected near 0% | expected to dominate |

## 5. What the 18-claim runs show

At 18 claims the left-hand column dominates, so aggregation should win, and it does. Cumulative conditional, 4,096 roots, seed 17, exact average-policy exploitability as window geometric means:

| Iterations | Aggregate then clip | Clip on read | Ratio |
| --- | ---: | ---: | ---: |
| 1k–2.5k | 0.0216 | 0.0242 | 1.12 |
| 2.5k–4k | 0.0167 | 0.0201 | 1.21 |
| 4k–5.5k | 0.0129 | 0.0157 | 1.22 |
| 5.5k–7k | 0.0121 | 0.0174 | 1.44 |
| 7k–9k | 0.0118 | 0.0165 | 1.40 |

Clip on read's regret loss was about 0.45, against about 0.0035 for aggregation. That is the network chasing individual noisy samples rather than their group means. The comparison also changed the loss weighting (next section), so it is a recipe comparison, not a clean test of clip placement alone.

## 6. Loss weighting

Aggregate runs give target entries above zero **1.5 times** the squared-error weight of zero entries (`regret_positive_weight=0.5`). With aggregation this is harmless: all visits to one information set carry the same target, so the weight only changes emphasis *between* entries.

With unclipped per-visit targets, the same weight would favour individual **samples** by their sign, pulling the fitted average upwards. Two targets `+1` and `−1` with mean 0 would fit to 0.2. Clip on read therefore uses plain squared error, and so should any mode in which the network averages across differing targets, including the hybrid below.

## 7. The hybrid: aggregate, then clip on read

Combine the two ✅ cells:

1. **Group identical information sets** within the iteration and compute the weighted mean of their raw targets `old + G_k`, exactly as `aggregate_then_clip` does now.
2. **Do not clip the group mean.** Store it unclipped on every row of the group.
3. **Clip on read**, as clip on read does.
4. Use **plain squared error**.

Repeated information sets get an exact, low-noise average. Similar single-visit information sets keep their negative evidence, so the network's pooling across them is unbiased. In code this is `aggregate_then_clip` without the final `torch.relu` on the group mean in `_aggregate_regret_targets`, with `regret_positive_weight=0`.

It is the best of both **in principle**, with three caveats:

- **The positive weight is lost.** Aggregate runs use it; the hybrid cannot. If the weight is part of why aggregation wins at 18 claims, the hybrid gives that part back. An `aggregate_then_clip` run with plain squared error isolates this.
- **The generalisation assumption is untested.** The advantage at single visits exists only if the network really pools similar information sets as in example 2. If it keeps them separate, clipping on read changes nothing there.
- **Grouping still costs what it costs now.** At 69 claims exact duplicates are rare, so grouping buys little while needing a pass over a very large buffer. Plain clip on read may be the practical choice there. At 30 claims the buffer is small, about 156,000 player-1 rows per update, and the hybrid is cheap.

## 8. Tests

1. **`aggregate_then_clip` with plain squared error**, 18 claims, against the existing aggregate run. This separates the loss weight from clip placement.
2. **The hybrid at 18 claims.** It should match aggregation. A loss would mean the positive weight, or storing negative values, costs something.
3. **Aggregate against hybrid against clip on read at 30 claims**, at 4,096 and 1,024 roots. This is where single visits are common enough for the hybrid's advantage, if real, to appear. Evaluation must use approximate best responses with a fixed, adequate budget.
