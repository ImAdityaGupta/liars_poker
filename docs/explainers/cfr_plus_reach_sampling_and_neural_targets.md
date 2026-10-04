# Where reach goes in exact, sampled, and neural CFR+

This note answers one question: **if exact CFR+ adds an opponent-and-chance
reach-weighted regret, how can sampled CFR+ do that without explicitly
calculating reach, and does our neural CFR+ do the same thing?**

Short answer: a sampled **table** can preserve reach by adding a value on
visits and zero on non-visits. Our online **neural target** instead fits a
value conditional on a visit; visit frequency affects the fitting loss, but
does not explicitly scale that target's cumulative regret increment. These
are different uses of the same sampled traversals.

## 1. Two quantities at one information set

Fix an iteration `t`, an updating player `p`, and one of that player's
information sets `I`. An information set may contain several hidden histories
`h`, such as different opponent hands that the player cannot distinguish.

Let `u_t(h,a)` be the expected continuation payoff if we force action `a` at
`h` and otherwise follow the current policies. The continuation payoff
already averages over later chance events and opponent decisions. Let
`u_t(h)` be the payoff under the current mixture of actions at `I`.

For each hidden history, define its **counterfactual reach weight**:

```text
w_t(h) = probability of chance events leading to h
       * probability of the opponent's actions leading to h.
```

The updating player's own earlier action probabilities are excluded. Now
define:

```text
q_t(I)   = sum over h in I of w_t(h)
g_t(I,a) = [sum over h in I of w_t(h) * (u_t(h,a) - u_t(h))] / q_t(I)
r_t(I,a) = q_t(I) * g_t(I,a).
```

`g_t` is the **conditional advantage** given counterfactual arrival at `I`;
`r_t` is the **counterfactual regret increment**. `q_t(I)` applies to the
information set before selecting `a`, so it has no `a` argument. If
`q_t(I) = 0`, the conditional quantity is undefined but the counterfactual
increment is zero.

The exclusion of the player's own reach matters. If the player currently
avoids a branch, CFR can still assess how to play *if the player chose to
take that branch*. That is why sampling all updating-player actions is the
simple external-sampling case.

## 2. Exact dense CFR+: add `q * g`, then clip

At each iteration the exact solver computes action values throughout the
game tree and updates each regret table entry:

```text
R_t(I,a) = max(0, R_(t-1)(I,a) + q_t(I) * g_t(I,a)).
```

It does not usually materialize an array named `q_t`. In
[`cfr_plus_dense.py`](../../liars_poker/algo/cfr_plus_dense.py), the
opponent-likelihood arrays `Lopp`, the hand/blocker matrices, and backward
value calculation put the required reach factors into `action_vals`.
`CFRPlusDense._update_player` adds `action_vals - V_state` directly to `R`
and clips `R` after the addition. Those value differences are already in
counterfactual units.

Example: if `q_t(I) = 0.1` and `g_t(I,a) = +0.5`, the increment is `+0.05`.
The `+0.5` tells us how good action `a` is *at* `I`; the `0.1` tells us how
much that improvement contributes to whole-game regret this iteration.

## 3. Sampled table CFR+: visits and non-visits jointly estimate `q * g`

Now do `K` independent root traversals for the same fixed policy. Sample
chance and opponent actions; expand the updating player's actions. For root
traversal `k`, write:

```text
Z_k(I) = 1 if this root traversal visits I, otherwise 0
G_k(I,a) = its sampled conditional advantage if it visits I.
```

In this game, claim indices strictly increase, `CALL` ends the game, and an
information set includes the full public claim history. Thus **one root
traversal can visit a particular information set at most once**, even when
it expands all of the updating player's actions. Assuming the root sampling
matches the exact chance distribution, use:

```text
estimated counterfactual increment
    = (1/K) * sum over k=1..K of Z_k(I) * G_k(I,a).
```

Every non-visit contributes **zero**. Under the matching sampling
distribution, the expectation of this increment is `q_t(I) * g_t(I,a)`.
There is no need to form an explicit `q_hat`. If you do form one, it is
approximately `number of visits / K`; multiplying it by the **mean over
visits** gives the same expression. The single `Z_k(I)` indicator is valid
here because the same root cannot visit `I` twice.

For `K = 100`, suppose ten roots visit `I`, each reporting `G_k(I,a) = +1`:

```text
sampled table: (10 * 1 + 90 * 0) / 100 = +0.1.
```

The table receives `+0.1` before CFR+ clips its cumulative regret. Dividing
by the ten **visits** instead would give `+1`, a conditional update. That is
a different learning rule.

There are two qualifications. Actual sampled `G_k` values vary because
hidden hands and continuations vary, so sampling introduces both reach and
value noise. Also, CFR+ clipping is nonlinear: even an unbiased *pre-clip*
increment need not produce an unbiased *clipped* result at finite `K`.

In this repository, root deal sampling and hand normalization must be checked
against the dense solver before claiming that `visits / K` numerically equals
the exact `q_t(I)` for every encoded information set. Sampling traverser
actions adds another inclusion correction. Neither qualification changes the
basic distinction between dividing by all roots and dividing by visits.

## 4. Our neural CFR+ trainer: visits become training records

The GPU traversal in
[`neural_cfr_plus_gpu.py`](../../liars_poker/algo/neural_cfr_plus_gpu.py)
calculates, on each **visited** traverser row:

```python
instant_regret = action_values - node_value
raw_target = ((t - 1) / t) * old_network_regret + instant_regret / t
```

It clips the target per record by default, or groups current-iteration
records before clipping in the experimental `aggregate_then_clip` mode. The
regret network is fitted to those records. In
[`deep_cfr_plus.py`](../../liars_poker/algo/deep_cfr_plus.py),
`run_iteration` **clears the regret buffer before collecting each player's
new records**. It keeps the cumulative state in the network parameters,
not in a table of per-root updates or a replay of all previous iterations.

Suppose `K = 100`, ten roots visit `I`, and all ten report advantage `+1`.
The network receives **ten labels**, each containing `+1/t` as the fresh
term. It receives no zero labels for the 90 misses. If it could fit `I`
perfectly in isolation, ten identical labels have the same optimum as one:
the fresh term in its prediction is `+1/t`, not `+0.1/t`.

Here is the same example at `t = 10`, starting from zero cumulative regret:

| Update | Unscaled new regret | Scaled value represented for regret matching |
| --- | ---: | ---: |
| Exact or correctly normalized sampled table, with `q = 0.1` and `g = 1` | `0.1` | `0.1 / 10 = 0.01` |
| Ideal fit to this trainer's ten visited targets | `1` if multiplied back by `t` | `1 / 10 = 0.1` |

That tenfold difference is **not** caused by the `/t`: both rows use it.
It comes from using `g` rather than `q * g` for the new cumulative increment.

Visit frequency still has two effects in the real shared network. Ten
records give `I` more influence on the *fitting loss* than one record. A
never-visited `I` has no direct target this iteration, although shared
parameters may change its prediction. Neither effect is the same operation
as adding `Z_k * G_k / K` to an exact table.

When traverser actions are sampled, the GPU code uses inclusion corrections
for sampled child values and inverse path-inclusion weights for deeper
records. Those compensate for **traverser-action subsampling**. The stored
`path_probability` does not track the probability of the opponent's sampled
actions and is not an explicit `q_t(I)` multiplier.

## 5. Why the difference may matter only sometimes

Regret matching depends on the *relative* positive regrets of actions at
one information set. If `q_t(I)` is the same positive constant on every
iteration, removing it merely scales that information set's whole regret
vector. The current policy there is unchanged under exact arithmetic.

If the opponent changes how often it reaches `I`, then `q_t(I)` changes over
time. The two updates can then prefer different actions. For example:

| Iteration | `q_t(I)` | Conditional advantages `(A, B)` |
| --- | ---: | ---: |
| 1 | `1` | `(+1, -1)` |
| 2 | `0.01` | `(-2, +2)` |

After **clipping after each iteration**, exact counterfactual CFR+ has
regrets `(0.98, 0.02)`; the conditional update has `(0, 2)`. They choose
very different next strategies. The `B = 0.02` entry matters: its negative
first-iteration regret was clipped to zero *before* its small positive
second-iteration increment. An earlier conversational example incorrectly
gave `B = 0` by clipping only at the end.

This demonstrates a possible algorithmic difference, not proof that our
neural trainer fails because of it. Finite network fitting, clipping of
individual records, action sampling, and strategy averaging can each affect
the resulting policy. The relative importance of the missing reach factor
has to be measured against **exact exploitability** on a tractable game.

## 6. What Deep CFR does differently

Original [Deep CFR](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf)
is a separate neural CFR algorithm, not this online neural CFR+ trainer. It
stores sampled instantaneous-advantage records from **many iterations** in
a bounded reservoir and refits its advantage network on that historical
memory. With equal numbers of root traversals per iteration and equal record
weights, an information set's fitted mean across retained records is
proportional to:

```text
sum_t q_t(I) * g_t(I,a) / sum_t q_t(I).
```

The denominator is shared across actions at `I`, so regret matching ignores
it. Deep CFR can therefore obtain reach weighting through historical record
frequency without feeding `q_t(I)` into each label. Its linear-weighting
variant additionally weights records by iteration. Finite reservoir size,
approximate fitting, and rare information sets still cause problems.

Our trainer retains no such historical regret-record distribution. It fits
new targets based on the previous network prediction. This is compact and
fast, but means Deep CFR's historical-frequency argument does not by itself
justify the online neural CFR+ target. Deep CFR also does not reproduce the
per-iteration clipping of CFR+ merely by averaging historical records.

## 7. Why `/t` is separate from reach

If `R_t` is exact cumulative clipped regret and `S_t = R_t / t`, then:

```text
S_t = max(0, ((t - 1) / t) * S_(t-1) + (q_t * g_t) / t).
```

This is just a change of scale: exact regret matching gives the same policy
from `R_t` and `S_t`. In our neural target, the fresh term is instead a
sampled conditional advantage divided by `t`. The `/t` makes network output
scale easier to manage; it does **not** supply the missing reach weight.
Approximate fitting makes even this mathematically harmless rescaling an
empirical question, because the fresh label gets small late in training.

## 8. The diagnostic bridge

The first clean comparison on the 18-claim game is:

```text
0. Exact table:             exact q_t(I) * exact g_t(I,a)
1. Sampled reach only:      visit fraction * exact oracle g_t(I,a)
2. Sampled reach and value: (1/K) * sum over roots of Z_k(I) * G_k(I,a)
3. Conditional table:       mean G_k(I,a) over visits only
4. Neural regret state:     train on visited targets, with either an
                            independent table ledger or the previous network
                            supplying the old cumulative regret.
```

Use `K = 1,024` root traversals per player per iteration first, matching one
arm of the [330-minute 18-claim experiment](../experiments/18_claim/2026-09-28_13-29-00_18_claim_target_sampling_factorial.md), then check `K = 4,096`.
Fully expand traverser actions until the reach/value comparison is understood.
Keep aggregate-then-clip versus clip-each and traverser-action sampling as
later, separately named switches. Compare frozen-policy target estimates
first, then exact current- and average-policy exploitability in free-running
tests. The [bridge plan](neural_cfr_plus_regret_units_and_bridge.md) describes
those later comparisons.

The key measurement is whether the counterfactual sampled table approaches
the exact table as `K` grows, and whether the conditional table departs from
it. That gives us a direct test of the reach-weighting concern before asking
the neural network to learn anything.
