# From exact CFR+ to this repository's neural CFR+

This guide is meant to answer a specific question: **what, exactly, did we replace when we moved from a tabular CFR+ solver to neural networks, and how could those replacements stop helping as training continues?** It describes the implemented algorithm, not a generic promise that all algorithms called “Deep CFR” behave alike.

Start with [the runnable orientation notebook](../../notebooks/sep_2026/repo_reorientation.ipynb) if the game rules or artifacts are unfamiliar. For an entry point into the implementation, see the [neural CFR+ code map](neural_cfr_plus_code_map.md). The most relevant implementation files are [the game and rules](../../liars_poker/env.py), [exact dense CFR+](../../liars_poker/algo/cfr_plus_dense.py), [the neural trainer](../../liars_poker/algo/deep_cfr_plus.py), and [GPU traversal](../../liars_poker/algo/neural_cfr_plus_gpu.py).

CPU experiments carried out after this guide are recorded in [sampled regret targets](../experiments/18_claim/2026-09-27_23-28-05_cfr_plus_sampled_targets_cpu.md), [the exact shadow ledger](../experiments/18_claim/2026-09-27_23-58-30_cfr_plus_shadow_neural_cpu.md), [the neural clip-order intervention](../experiments/18_claim/2026-09-28_01-04-50_cfr_plus_neural_clip_order_cpu.md), [its longer run](../experiments/18_claim/2026-09-28_01-17-33_cfr_plus_neural_clip_order_long_cpu.md), and [the depth target audit](../experiments/18_claim/2026-09-28_01-47-35_cfr_plus_neural_depth_target_audit_cpu.md). Each note states the question, possible outcomes, measured results, and limits.

### How to read this guide

The same words are used for several different quantities, so keep this small
dictionary nearby:

| Symbol or word | Meaning here |
| --- | --- |
| `I` | Everything the acting player can distinguish: their private hand and public claim history. |
| `σ_t(I,a)` | Probability that the **current** strategy chooses legal action `a` at iteration `t`. |
| `v_t(I,a)` | Counterfactual value of choosing `a` now and following the iteration's strategies afterward. |
| `r_t(I,a)` | This iteration's advantage `v_t(I,a) - Σ_b σ_t(I,b)v_t(I,b)`. It may be negative. |
| `R⁺_t(I,a)` | Nonnegative *cumulative* CFR+ regret after iteration `t`. |
| `R⁺_t/t` | The scaled quantity approximated by this repository's regret network. |
| `σ̄_T` | Reach-weighted historical **average** strategy; usually the final policy we evaluate. |
| `traversal` | One sampled starting deal and its explored continuation for one traversing player; it can create many training records. |
| `fit step` | One Adam minibatch update to a neural network, separate from a CFR+ iteration. |

There are three nested activities: an **outer CFR+ iteration** changes the
strategy; **traversals** collect the evidence for that change; **fit steps**
teach networks to reproduce the resulting targets. Confusing these counts can
make a run with more computation look like it learned more, even if it simply
performed more work inside fewer outer iterations.

## 1. What is the game state, and what can a player know?

An actual position includes both private hands and the public sequence of claims. A player observes **their own hand and the public claims**, but not the opponent's hand. The policy therefore acts on an *information set* (`InfoSet(pid, hand, history)`), not the fully revealed game position. Chance draws cards without replacement. Each new claim must have a higher index than the previous one; after a claim, the next player may call. A terminal call pays `+1` to the winner and `-1` to the loser in the training traversals.

The `GameSpec` fixes ranks, suits, cards per hand, claim families, and suit symmetry. Suit symmetry lets the solver represent a private hand by rank counts. The neural [infoset encoder](../../liars_poker/policies/neural.py) concatenates those counts with a bit for each previously made claim. The bits retain the entire public sequence because legal claims strictly increase: a set of claim indices has only one possible order. For the 69-claim spec, the input has 6 hand-count features plus 69 history bits; the network predicts one value for `CALL` and one for each of the 69 claims. Its legal-action mask discards actions unavailable at that history.

The hard part is **not** merely that there are 69 possible claims. There are many possible public histories, and a claim can be good for one private hand and disastrous for another. A successful network has to generalize strategically across those combinations.

## 2. What does CFR+ want to compute?

At one infoset `I`, suppose the current policy is `σ(I)` and `v(I,a)` is the value of forcing action `a` now, then following the current policies afterward. The current node value is

```text
v(I) = Σ_a σ(I,a) v(I,a)
instantaneous regret r(I,a) = v(I,a) - v(I).
```

Here “regret” is **not the neural-network training loss**. It asks how much better or worse an action would have been than the policy's mixture at this infoset in the current iteration. The payoff is measured from the traversing player's point of view, so a positive regret means the action would have improved that player's result.

For example, with `CALL` and `RAISE`, suppose the policy is `(0.5, 0.5)` and their values are `(0.6, -0.2)`. The node value is `0.2`, so the instantaneous regrets are `(0.4, -0.4)`. CFR+ adds them to cumulative regrets and clips *each action's cumulative regret* at zero:

```text
R⁺_t(I,a) = max(0, R⁺_(t-1)(I,a) + r_t(I,a)).
```

The next current strategy is proportional to positive regrets. Starting from zero, the example gives cumulative regrets `(0.4, 0)` and the next policy picks `CALL`. If later play makes `RAISE` profitable, its cumulative regret can become positive and the current policy can change again. Thus even a perfectly implemented CFR+ current strategy can oscillate.

The values are **counterfactual**: a player's regret at `I` is weighted by the chance and *opponent* reach of `I`, without multiplying by that player's own probability of reaching `I`. Imagine your present strategy reaches `I` only 1% of the time, but conditional on being there it routinely makes a terrible call. If your own 1% reach also suppressed the regret update, you would barely learn to fix that mistake. CFR still asks what you *would* gain by changing the decision at `I`. In contrast, when constructing the **average policy**, your own reach matters: an iteration in which you almost never go to `I` should contribute little to the played average at `I`. The exact weighted average at an infoset is

```text
average_σ_T(I,a) = [Σ_(t=1)^T t · own_reach_t(I) · σ_t(I,a)]
                   / [Σ_(t=1)^T t · own_reach_t(I)].
```

This is not generally the simple arithmetic mean of the action probabilities. If one iteration's policy reaches an infoset with probability `0.1` and another reaches it with probability `0.9`, they should not contribute equally there. CFR+ uses later iterations more heavily as well. The convergence argument concerns cumulative regret and the **average** strategy; it does not promise that every later snapshot has lower exploitability than the previous one.

### A tiny game tree: what is actually hidden?

The following is a schematic of a real kind of decision in this game, with
only two possible claims shown. It is **not** a complete `GameSpec`:

```text
chance deals P1's hand and P2's hand privately
  P1: claim C0
    P2: CALL             → reveal hands; C0 was true or false
    P2: claim C1
      P1: CALL           → reveal hands; C1 was true or false
```

At P2's first decision, the complete game state includes both hands. P2's
information set includes **P2's hand and the fact that P1 claimed C0**. All
possible P1 hands consistent with P2's cards and that public history are
merged into that one information set. The claim C0 is evidence: a P1 policy
that makes C0 much more often with strong hands changes the probabilities of
those hidden hands. Card blockers matter too, because a card in P2's hand
cannot also be in P1's. Exact traversal handles both effects when it values
`CALL`. A network sees P2's private hand and the claim history, then has to
learn the same strategic distinction from data.

### One CFR+ update by hand

Imagine an information set with just `CALL` and `RAISE`. The numbers below
are illustrative **counterfactual action values**, already summed over hidden
states and the relevant chance/opponent reach. They are not terminal payoffs
from one fixed deal.

| Iteration | Current `σ(CALL, RAISE)` | Action values | Mixture value | Instant regrets | Cumulative `R⁺` after clipping |
| ---: | --- | --- | ---: | --- | --- |
| 1 | `(0.5, 0.5)` | `(0.6, -0.2)` | `0.2` | `(0.4, -0.4)` | `(0.4, 0)` |
| 2 | `(1, 0)` | `(-0.2, 0.6)` | `-0.2` | `(0, 0.8)` | `(0.4, 0.8)` |

At the start, all regrets are zero, so regret matching plays uniformly. After
iteration 1, only `CALL` has positive cumulative regret, so the **next**
current strategy chooses `CALL` with probability one. In iteration 2,
`RAISE` turns out to be better, even though it was not selected by the current
strategy. CFR+ still evaluates it and increases its cumulative regret. The
strategy for iteration 3 becomes `(0.4/1.2, 0.8/1.2) = (1/3, 2/3)`.
Regrets belong to **actions at an information set**, not to the realized
trajectory alone; that is why exploring the traverser's alternatives matters.

The clipping is on the *running sum*. For an action with old `R⁺ = 0.4` and
new `r = -0.7`, the next regret is `max(0, 0.4 - 0.7) = 0`. CFR+ does not add
`max(0, r)` to the old regret; doing so would discard the negative evidence.
It also does not replace the regret table with this iteration's `r`.

### Why these are counterfactual values

Take a full history `h` inside information set `I`. Its probability under the
current play can be factored into chance reach `π_c(h)`, the opponent's reach
`π_-i(h)`, and the acting player's own reach `π_i(h)`. Informally, a CFR
action value sums over hidden histories using

```text
counterfactual weight of h = π_c(h) × π_-i(h)
                            (the player's π_i(h) is left out)
```

and includes the continuation value after forcing the action. If chance reach
is `0.1`, opponent reach is `0.5`, and your own reach is `0.01`, the history
contributes weight `0.05` to the counterfactual calculation, **not**
`0.0005`. The exact tables can therefore improve a choice at an information
set your current policy almost never visits. `CFRPlusDense._update_player`
computes this with exact opponent reach and card-blocker matrices; its local
`action_vals - V_state` is already expressed in this counterfactual value
scale. There is no additional multiplication by the player's own reach in the
regret update.

The average policy asks a different question: *how often would this player's
historical strategies actually reach I?* Suppose iteration 1 uses `CALL`
with probability one at `I` but reaches `I` with own probability `0.1`;
iteration 2 uses `RAISE` with probability one and reaches it with own
probability `0.9`. With linear iteration weights `1` and `2`, the accumulated
weights are `1 × 0.1 = 0.1` for `CALL` and `2 × 0.9 = 1.8` for `RAISE`.
The average therefore plays `CALL` with probability `0.1 / 1.9 ≈ 0.053`,
not `0.5`. The exact implementation stores these weighted action sums in
`SS0` and `SS1` and normalizes them when constructing the average policy.

Why should averaging help at all? Roughly, regret matching tries to make
each player's accumulated advantage from changing decisions small. In a
two-player zero-sum game, when **both** players have small average regret,
their suitably averaged strategies have little incentive for either player
to deviate: they approach a Nash equilibrium. Exact CFR can relate overall
regret to the sum of counterfactual regrets across information sets. This is
the source of the method's appeal. It is **not** a claim that each current
strategy is close to equilibrium, that exploitability falls at every
snapshot, or that the same guarantee automatically survives sampled,
neural, finite-step updates. In the neural trainer we have to check whether
the approximate updates still make the right regrets small.

### What the exact code does

[Dense CFR+](../../liars_poker/algo/cfr_plus_dense.py) has explicit arrays indexed by public history, private hand, and action. At each iteration it computes values throughout the game, updates one player's regrets, then the other's, and adds their reach-weighted current strategies to average accumulators. It can do the chance and opponent-hand accounting exactly. The resulting [dense policy](../../liars_poker/policies/tabular_dense.py) can be evaluated with an [exact best response](../../liars_poker/algo/br_exact_dense_to_dense.py).

That becomes impossible to store or traverse directly at 69 claims. A dense history index alone scales like `2^claim_count`. The neural method keeps the same *questions*—values, regrets, and average strategy—but changes how it answers and stores them.

## 3. The three substitutions in neural CFR+

| Question | Exact dense solver | Implemented neural solver |
| --- | --- | --- |
| Where is cumulative regret stored? | A table entry for every represented infoset and action | A regret network for each player predicts a regret-like vector from encoded infoset features |
| How is a traversal evaluated? | Sum the relevant game-tree branches and chance/opponent possibilities | Draw private deals, sample opponent actions, expand traverser's actions (possibly sample some of those too), and back up returns |
| Where is the average policy stored? | Explicit own-reach-weighted strategy sums | A strategy network for each player fits strategy records kept in a finite reservoir |

It helps to picture one outer iteration as a data-generation step followed by supervised fitting:

```text
regret networks at iteration t-1
           │
           ▼
compute current strategies by regret matching
           │
           ▼
sample deals and traverse for P1 ──► P1 regret examples ──► fit P1 regret net
           │
           ▼
sample deals and traverse for P2 ──► P2 regret examples ──► fit P2 regret net
           │
           └───────────────────────► strategy examples ─────► fit average nets
```

The code uses alternating player updates. It does not fit a regret network *during* one player's traversal, so that traversal sees a frozen policy. P1's regret network is fitted before P2's traversal, matching the alternating update order. See [`run_iteration()`](../../liars_poker/algo/deep_cfr_plus.py).

### One complete neural outer iteration, in plain language

Assume the trainer has finished iteration `t-1` and is about to run iteration
`t`. It owns **four networks**: P1 regret, P2 regret, P1 average strategy,
and P2 average strategy. Each regret network outputs a number for each
possible action. These are *not* action probabilities; positive outputs are
converted to a current strategy by regret matching. Each average network
outputs logits, converted to probabilities over legal actions by softmax.

1. Increase the outer iteration counter to `t`. Clear P1's recent regret
   records. Keep its learned network weights and Adam state. Do **not** clear
   either historical strategy reservoir.
2. Draw `traversals_per_player` fresh starting deals for P1's update, in
   chunks of `traversal_batch_size`. At P1 decisions, evaluate P1's legal
   alternatives; at P2 decisions, sample from P2's current policy. Each P1
   decision visited creates a P1 regret target. Each P2 decision visited
   creates a P2 *strategy* example for the historical average.
3. Fit P1's regret network to those targets for `regret_train_steps` Adam
   minibatches. This changes P1's current policy.
4. Do the symmetric P2 traversal and fit, now using the updated P1 regret
   network. It creates P2 regret examples and P1 historical strategy
   examples. This is **alternating** updating, so the two players' traversals
   within one outer iteration need not use the exact same pair of policies.
5. Fit both average-strategy networks on their persistent strategy
   reservoirs, for `strategy_train_steps` minibatches per network. The
   average nets do **not** determine the traverser's action values; the
   regret nets determine current play during traversal.

`traversals_per_player=1,024` thus means 1,024 sampled root deals for P1
**and** 1,024 for P2 *per outer iteration*. It does not mean 1,024 regret
records or 1,024 optimizer steps. A single root deal can branch into many
traverser decisions and produce many records.

The exact solver performs the same conceptual regret/average bookkeeping,
but it can enumerate all represented histories and hidden hands instead of
learning four functions from sampled records. The neural update inherits
historical regret through P1/P2 network weights, while the exact update
inherits it through explicit `R0`/`R1` arrays.

### A. Sampling a traversal

The trainer draws a batch of complete private deals. For one traversing player, it explores their available actions at a reached decision while sampling one action from the other player's current strategy. This is **external sampling**: “external” chance and opponent choices are sampled; the traverser's choices are expanded. Each leaf reached by a call has an exact payoff for that sampled deal. Backing up those payoffs gives noisy estimates of the action values at the visited infosets.

“Full action expansion” does **not** mean exact CFR: opponent actions and private deals are still sampled. In the GPU code, even traverser actions may be capped: with cap 16, at most 16 legal *claims* are sampled at a traverser decision; `CALL`, when legal, is handled exactly. Claims with fewer than 16 legal alternatives are fully expanded. To keep an action-value estimate unbiased *before later nonlinear operations*, a selected claim with inclusion probability `q` gets an inverse-`q` correction:

```text
estimated_value(a) = baseline(a)
                   + selected(a)/q(a) · [sampled_child_value(a) - baseline(a)].
```

With no baseline, an unselected claim contributes zero to that sample's action-value vector. At deeper traverser nodes, reaching the node also depended on earlier action selections; an inverse product of their inclusion probabilities weights the training record. [The sampler and correction](../../liars_poker/algo/neural_cfr_plus_gpu.py) implement these ideas. The streamed traversal breaks expanded edges into chunks to bound *live GPU memory*; it does not reduce the total number of sampled edges or their statistical variance.

Here is an action-sampling example. Suppose two claims, `A` and `B`, are
legal, but the cap allows only one. Each is chosen with probability `q=1/2`.
Imagine the true continuation value of `A` is `0.6`, and use no baseline.
On one traversal the estimate for `A` is `1.2` if selected and `0` if not:

```text
E[estimated value of A] = (1/2) × (0.6 / (1/2)) + (1/2) × 0 = 0.6.
```

The estimate is unbiased **in this simple conditional example**, but its
individual realizations are more extreme than the true value. The sampled
`1.2` can even exceed the `+1` terminal payoff range; it is an
inverse-probability estimator, not a realized payoff. At a deeper node that
is reached only if two earlier sampled actions were selected, each with
probability `1/2`, the path inclusion probability is `1/4` and the record
weight is `4`. This restores its contribution *in expectation* but increases
variance. The implementation also has an optional `call` baseline, which
replaces zero for unsampled claims with a known value and corrects only the
sampled difference from that baseline. A good baseline reduces variance;
the algebra does not guarantee that a particular baseline is good.

There are **three distinct uses of the word batch** here: the number of
independent root deals (`traversals_per_player`), the number processed
simultaneously (`traversal_batch_size`), and the optimizer minibatch size
(`batch_size`). Raising one does not automatically raise either of the
others. `traverser_action_sample_schedule=(16,)` limits *claim* alternatives
at each traverser decision; it does not cap the number of root deals. It
also leaves `CALL`, when legal, handled separately. Setting no claim cap
expands all the traverser's legal claims, but still samples opponent actions
and private deals. Neither a larger `traverser_action_chunk_size` nor a larger
`traversal_live_row_budget` creates more independent samples; they only
change how expansion is held and processed in memory.

The resulting examples are correlated. Expanding many actions for one sampled deal does not make them independent observations of hidden hands or opponent behavior. More records, more root deals, and more independent opponent continuations are different resources.

### B. Fitting the regret networks

The network outputs are used as **normalized positive cumulative regrets**. Divide the exact CFR+ update by iteration `t`:

```text
R⁺_t / t = max(0, [(t-1)/t] · [R⁺_(t-1)/(t-1)] + r_t/t).
```

This scaling preserves regret-matching action ratios when the vector is known exactly. The implemented target substitutes the *previous network prediction* for `R⁺_(t-1)/(t-1)` and a sampled action-value difference for `r_t`:

```text
target_t(I,a) = ReLU( (t-1)/t · ReLU(previous_net(I,a))
                        + 1/t · sampled_instant_regret(I,a) ).
```

For each traversing player, the trainer clears a **recent regret buffer**, fills it with this iteration's examples, and takes a fixed number of Adam steps on masked, positive-weighted mean squared error. The historical information is mostly in the *network weights*, not in that cleared buffer. At a reached infoset, current play takes `ReLU(network outputs)` on legal actions and normalizes them; if all are nonpositive, it plays uniformly among legal actions. See [regret matching and fitting](../../liars_poker/algo/deep_cfr_plus.py).

For a concrete calculation, let `t=2`. At one infoset yesterday's network
predicts scaled regrets `(0.4, 0)` for `(CALL, RAISE)`, so today's current
strategy plays `CALL`. Suppose a sampled continuation estimates action values
`(0.2, 0.8)`. The node value is `0.2`, and the instantaneous regret estimate
is `(0, 0.6)`. The target is

```text
old contribution          = (1/2) × (0.4, 0)   = (0.2, 0)
new sampled contribution  = (1/2) × (0, 0.6)   = (0, 0.3)
regression target         = ReLU((0.2, 0.3)) = (0.2, 0.3).
```

If yesterday's network was exact, this agrees with storing cumulative regrets
`(0.4, 0.6)` and dividing by two. If it predicted `(0.7, 0)` instead, the
target becomes `(0.35, 0.3)`. The traversal cannot tell that the `0.7` was
wrong: it is treated as historical regret. At later iterations, this
inherited term receives almost all the weight. **Adam's learning rate does
not change the `1/t` arithmetic**; it controls how effectively the network
fits the target that the traversal produced.

This is the most important difference to hold in mind: a neural training loss can be tiny while the regret estimate is strategically wrong. The target is mostly an old prediction when `t` is large. At `t = 10,000`, a sampled instantaneous regret of `1` contributes only `0.0001`; at `t = 30,000`, it contributes `0.0000333`. Model error, numerical scale, or optimizer drift of comparable size can overwhelm the new signal. Low MSE against *self-generated* targets is not an independent correctness check.

There is also a nonlinear sampling issue. Imagine the true expected
instantaneous regret for one action is zero, with a sample of `+1` or `-1`
equally likely and zero previous regret. Averaging samples first and then
applying CFR+ gives `ReLU(0) = 0`; clipping each sampled target first and then
fitting its mean gives `(ReLU(+1) + ReLU(-1))/2 = 0.5` before the common
`1/t` scale is applied. In the `t=2` example above, suppose two visits to
the *same* infoset instead estimate `RAISE` regret as `+0.6` and `-0.6`.
Their raw scaled updates are `+0.3` and `-0.3`:

| Target mode | Targets seen by the fit at the two visits | Mean fitted target for `RAISE` |
| --- | --- | ---: |
| `clip_each_record` | `0.3`, `0` | `0.15` |
| `aggregate_then_clip` | Mean raw update `0`, then clip; assign `0` to both visits | `0` |

`clip_each_record` is the historical default. The experimental
`aggregate_then_clip` mode is implemented for **CPU with the `gpu_native`
traversal backend**: it groups identical encoded infosets within the current
regret buffer, takes a path-weighted mean of their raw targets, clips once,
and puts that common target back on each visit before fitting. It does not
create an exact expectation from two samples, keep an exact regret ledger, or
change the network architecture. The CPU experiments found a substantial
exact-exploitability improvement from this change on the 18-claim game;
[the longer factorial](../experiments/18_claim/2026-09-28_13-29-00_18_claim_target_sampling_factorial.md)
is the most direct follow-up. The toy example explains the direction of the
bias, not the full measured effect. In particular, if the old scaled regret
is already positive and far from zero, clipping may not change either sample;
the effect is greatest near the zero boundary.

### C. Fitting the average strategy networks

During a traversal for one player, the other player's visited decisions generate records of that other player's *current strategy*. Sampling that player's own preceding actions naturally visits some histories in proportion to their own reach. When traverser action sampling removed paths, the record receives inverse-inclusion weighting; it also receives the CFR+ iteration weight `t`. A [reservoir buffer](../../liars_poker/algo/deep_cfr.py) keeps a limited, approximately uniform sample of historical records. The [strategy fitting loss](../../liars_poker/algo/deep_cfr_plus.py) trains two separate networks with cross-entropy to reproduce those distributions.

The resulting `average_policy()` is a *distilled approximation* to the reach-weighted average, not an average-regret network and not an exact historical sum. Its error can be separate from regret-learning error. The original [Deep CFR paper](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf) uses an average-strategy network; [Single Deep CFR](https://arxiv.org/pdf/1901.07621) discusses the additional approximation introduced by that network and an alternative based on saved iteration policies. Those papers are useful comparisons, but their algorithms and compute budgets are not identical to this repo's online neural CFR+.

### A concrete strategy-record example

Suppose an opponent decision visited during the traverser-0 pass has current
strategy `(CALL=0.7, RAISE=0.3)`. The trainer records the infoset features,
legal-action mask, and **both** probabilities, not just the action it samples
to continue. The sampled opponent action decides which descendants are
visited; it does not turn the recorded target into a one-hot label. At
iteration 10 with `strategy_weighting='linear'`, that record has nominal
weight `10`. If sampled traverser-action choices made this path only one
quarter as likely to appear as under full expansion, its importance weight
is `4`, giving an effective fitting weight of `40`. The reservoir keeps
historical examples across iterations, so the average network has data about
older policies as well as the latest one.

This is why the two replay capacities mean different things. A
`regret_buffer_capacity` of 500,000 limits examples from the **current
player update**, because that buffer is cleared before the next update for
that player. A `strategy_buffer_capacity` of two million limits a reservoir
drawn from **many iterations**. Neither number directly specifies the
number of distinct infosets represented, nor how much data each infoset gets.

## 4. Parameter map: which approximation or cost does a knob change?

The [trainer constructor](../../liars_poker/algo/deep_cfr_plus.py) is the
source of truth for accepted names and defaults. The old
[69-claim adaptive script](../../scripts/run_cfr_plus_69_claim_adaptive.py)
shows one historical configuration, not a known best setting. In particular,
it used `random` action selection, `none` baseline, streamed traversal, and
action caps 16 or 24. The CPU aggregate-then-clip experiment is a different
configuration; that mode is **not available on CUDA in this code**.

### Sampling and traversal

| Knob | What changes | What it does **not** do |
| --- | --- | --- |
| `traversals_per_player` | Number of independently drawn starting deals per player update. More can reduce sampling noise; it also costs time. | Does not increase Adam steps or the claim cap. |
| `traversal_batch_size` | Root deals processed together in `gpu_native`. Affects device utilisation and peak memory. | Does not change the *intended* number of deals in an iteration. |
| `traverser_action_sample_schedule` | Cap on sampled **claim** edges at successive traverser decisions, e.g. `(16,)` uses cap 16 at every depth; `CALL` is handled separately. | Does not change root-deal count. No cap still samples chance and opponent play. |
| `traverser_action_sample_count`, `traverser_action_sample_fraction`, `traverser_action_full_first` | Alternative ways to specify claim sampling or fully expand the first traverser decision. | Do not make later sampled decisions exact. |
| `traverser_action_sample_mode` | `random` draws new random scores; `hash` computes deterministic scores from history, action, decision depth, and iteration. The inclusion correction is the same formal idea, but correlations may differ. | Does not change the cap by itself. |
| `traverser_action_priority_count` | Reserves some sampled slots for high-predicted-regret claims before randomly filling the rest. | Does not make selected actions representative without the corresponding inclusion accounting. |
| `traverser_action_baseline` | `none` uses zero for unselected claims; `call` uses a call-derived value as a control variate. | Does not change the terminal payoff or guarantee lower variance in every state. |
| `traversal_backend` | `recursive` runs a direct Python traversal; `gpu_native` batches traversal using tensors, even on CPU. | Does not identify the physical device: `device='cpu'` with `gpu_native` is valid. |
| `traversal_streaming`, `traversal_live_row_budget`, `traverser_action_chunk_size`, `traversal_record_flush_size` | Bound the size of live edge work or record transfers and change throughput/memory tradeoffs. | Do not themselves change the chosen action cap or provide more independent sampling. |

Sampling **more root deals** reduces uncertainty about hidden deals and
opponent continuations. Expanding **more claims per reached decision** makes
action comparisons for those particular deals more complete. Those are
different statistical resources. For example, taking four times as many
claim edges on the same deal still leaves the hidden hand and sampled
opponent choices unchanged. Streaming can make a large expansion fit in
memory, but does not make that expansion cheap in compute time.

### Neural fitting and time

| Knob | What changes | Likely tradeoff or diagnostic |
| --- | --- | --- |
| `regret_hidden_sizes`, `strategy_hidden_sizes` | Capacity of the separate player-specific MLPs; e.g. `(2048, 2048)` means two hidden layers of width 2,048. | Larger nets may represent more states but slow every inference and fit step. |
| `regret_train_steps`, `strategy_train_steps` | Number of Adam minibatches for **each** corresponding player network per outer iteration. | More fitting may improve target fit while reducing the number of CFR+ iterations completed per hour. |
| `batch_size` | Examples per Adam fit step. | Independent of root traversal batch size. Larger batches may change fit dynamics as well as throughput. |
| `learning_rate` | Adam parameter-update scale for the networks. | Does **not** change the CFR+ `1/t` scaling of instantaneous regret or increase traversal data. |
| Optimizer reset in a run script | Recreates Adam momentum/second-moment state while retaining network weights and replay, if the script implements it that way. | Not equivalent to changing learning rate, or to resetting cumulative regret. Compare separately. |
| `regret_positive_weight` | Extra squared-error emphasis on positive *target entries*. The code uses factor `1 + weight` for a positive entry. | Does not alter the CFR+ target equation; it changes the regression objective. |
| `regret_target_mode` | Choose clip per sampled record or path-weighted aggregate of identical infoset targets before clipping. | Aggregate mode is currently restricted to CPU `gpu_native`; it does not correct inaccurate old network predictions. |
| `regret_buffer_capacity`, `strategy_buffer_capacity` | Max recent-regret and historical-strategy records respectively. | Bigger regret capacity does not turn the buffer into a cumulative-regret table. |
| `strategy_weighting` | `linear` assigns iteration `t` weight; `uniform` assigns weight 1. | Does not by itself reconstruct exact own-reach weighting if records are unrepresentative. |
| `validation_fraction`, `validation_buffer_capacity` | Reserve sampled records for fit diagnostics. | A small loss against self-generated regret targets does not validate true regrets or exploitability. |
| `device_replay`, `fused_optimizer`, `amp_dtype`, `compile_models` | Storage, optimizer, precision, and execution choices. | Primarily throughput/memory knobs, though changed precision can also change numerics. |

For a fair comparison, keep separate records of **training minutes,
iterations, root deals, sampled claim edges, regret records, fit steps,
and peak memory**. Comparing only iteration count favors expensive updates;
comparing only record count can hide correlation among records. The
[18-claim target/traversal experiment](../experiments/18_claim/2026-09-28_13-29-00_18_claim_target_sampling_factorial.md)
illustrates this: 4,096 traversals often produce stronger progress *per
iteration*, but complete far fewer iterations within the same time.

## 5. Current strategy, average strategy, and evaluation

There are now **two different ways to play**. `NeuralRegretMatchingPolicy` computes the *current* policy from the two regret networks. `NeuralPolicy` uses the two strategy networks to play the *learned average*. In two-player zero-sum CFR, the average is normally the policy of interest; a worse and noisier current policy is not automatically a failure. But a current policy that deteriorates can be an early warning that regret estimates are going astray.

On small games, we can compile a neural policy to a dense table and compute an **exact** best response. On 30- and 69-claim games, that dense enumeration is too expensive; instead, we train an approximate responder against the frozen policy and Monte Carlo-evaluate the responder. The repo's exploitability convention is `p_first + p_second - 1`, where each probability is a responder win rate in one seat. The approximate result is *discovered* exploitability: a weak responder can underestimate a policy's true exploitability. The confidence bound covers the Monte Carlo evaluation of the **found responder**, not the possibility that the responder missed a better strategy. Five-, ten-, twenty-, and sixty-minute responders are different measurements and must be labeled separately.

This matters for the old 69-claim trajectory. Under the same 20-minute BR procedure, the estimate improved to about `0.262` for the 400-minute policy, then was about `0.280` for the 720-minute policy. Under 60-minute BRs the corresponding selected-snapshot values were about `0.282` and `0.301`. That is evidence of worsening *discovered* exploitability under matched responder budgets, though it is not exact exploitability and is not a multi-seed conclusion. The 18-claim exact-evaluation experiments make a genuine learning problem more plausible.

### How to read one exploitability number

Suppose we freeze a policy `P`, train a responder that plays the first seat
against `P`, and estimate that responder's win rate as `p_first = 0.65`.
Train a separate responder for the second seat and find `p_second = 0.60`.
Under this repository's zero-sum convention, the **discovered**
exploitability is `0.65 + 0.60 - 1 = 0.25`. If both responses are truly
optimal and their evaluations are exact, `0.25` is the policy's exact
exploitability. If the trained responders missed better actions, `0.25`
understates it. A tight Monte Carlo confidence interval around `0.25`
reduces uncertainty about *those responders' win rates*; it cannot prove
that their decisions were optimal.

For the small 18-claim game, an exact best response removes this responder
training uncertainty. That makes it particularly useful when asking whether
a change to the neural CFR+ update helped. But even an exact evaluation of
each saved snapshot does not mean its plotted line is monotone: the trainer
uses sampling, finite network fits, and a changing current policy. For large
games, comparisons need the same responder algorithm, compute budget, role
setup, and preferably multiple responder seeds. A 2-minute BR estimate and
a 20-minute BR estimate of different snapshots are not interchangeable.

When reading learning curves, compare three different x-axes deliberately:

1. **Training time:** did the method produce a stronger policy for the
   compute budget we actually pay?
2. **CFR+ iteration:** did each outer strategy update accomplish more? A
   costly traversal or fit setting can win here while losing on time.
3. **Responder training time:** is the apparent exploitability still rising
   because the evaluator is getting stronger? If so, a short BR curve can
   make several policies look equally good when they are not.

## 6. A map from assumptions to failure modes

| Exact algorithm assumes | Approximation here | What can go wrong | An informative test |
| --- | --- | --- | --- |
| Correct terminal payoff and legal history | GPU tensors and packed history | A rule or history bug quietly changes the game | Compare GPU terminal and legal-action outputs with `Env` across random deals and late histories, including claims above index 63 |
| Correct expected action values | Sampled deals, opponent actions, and sometimes own claim actions | High variance or wrong inclusion probability | Freeze a small-game policy; compare sample means and variances with exact action values, by infoset and depth |
| CFR+ clips *cumulative* regrets | Historical default clips each noisy per-record target; CPU experiment aggregates before clipping | Sampling/clipping interaction can change the expected target | Compare `mean(clip(sampled update))` with `clip(mean(sampled update))` against a small-game exact oracle |
| Old cumulative regret is exact | Previous neural prediction stands in for old regret | Self-confirming error and vanishing `1/t` innovation | Compare predictions with independent exact regret targets and action rankings across iteration `t` |
| Every infoset can be represented and revisited | Fixed-size network, fresh regret records, finite optimizer steps | Rare critical infosets have poor fit while average MSE looks good | Stratify oracle errors by depth, legal-action count, reach, and BR-discovered weakness |
| Average is a correct own-reach-weighted sum | Sampled records, finite reservoir, second network | The output policy differs from the true historical average | On 18 claims, compute both exactly for the same iteration policies and compare exact exploitability |
| Exploitability is measured accurately | Finite-time learned responder | Apparent improvement or plateau reflects responder skill | Use matched budgets, several responder seeds, and longer curves on frozen snapshots |

### Why particular old checks did not settle the question

The streamed traversal validation established valuable **implementation equivalence** between two traversal paths on selected small-spec samples. It compared root values and record counts. Both paths can agree and still share the same sampled-target or fitting flaw. Likewise, held-out regret MSE is measured against targets generated by the old network, not an independent table of true counterfactual regrets. In the 69-claim adaptive run, the late regret validation had only dozens of records per player at a monitor point, despite very low reported MSE. These observations motivate stronger semantic tests; they do not invalidate the checks already performed.

## 7. A diagnostic sequence, including completed checks

Read these as ways to localize an error, **not** as seven prerequisites that
must all be rerun. The linked CPU experiments above have already tested the
root target, a small-game shadow ledger, clipping order, and aspects of
average-policy fitting. Those results support changing clipping order but
have not explained the entire gap to exact CFR+ or established a GPU-scale
solution. The steps below also specify what still needs checking in larger
or deeper game states.

1. **Check semantics before training.** On the 18-claim game, choose frozen tabular policies and exact infosets. Compare terminal truth, legal masks, backed-up values, and GPU packed histories with the exact CPU implementation. Repeat for random 69-claim terminal states to cover the two-word packed history. This catches mechanical mistakes without paying for a long run.
2. **Isolate sampling and clipping.** For those same frozen policies, collect many independent traversal samples and check whether *pre-clipping* action values converge to exact values. Measure variance and effective sample size at cap 16, cap 24, and full traverser expansion. Then measure the post-clipping gap above. If pre-clipping values are wrong, fix sampling; if they are right but post-clipping targets differ substantially, address variance or update design.
3. **Remove the neural net from the experiment.** Run a small-game tabular regret updater using the *same sampling rule* and, where possible, common random seeds; evaluate it exactly over time. Once the two updaters choose different policies, they cannot literally follow identical self-play trajectories, so compare learning curves across multiple seeds. If tabular sampling also plateaus or deteriorates, the sampler/CFR+ combination is the priority. If it succeeds while neural CFR+ fails, investigate representation, target bootstrapping, optimizer steps, and fallback behavior.
4. **Test the network against an independent oracle.** Instrument a short small-game run to retain a tabular “shadow” cumulative-regret ledger while it follows the neural policies. Exact enumeration can then compute each frozen policy's instantaneous counterfactual regrets and update that ledger independently of the network's predictions. Compare the network's regret *signs, action ordering, and resulting policy* against this oracle at important infosets, including ones selected by an exact BR. Record how often the policy falls back to uniform. A falling self-target MSE alongside worsening oracle error would directly demonstrate self-confirming training.
5. **Isolate averaging.** Instrument that 18-claim run to accumulate the true own-reach-weighted average of its neural current policies as a dense table. Compare its exact exploitability with the strategy network's exact exploitability. If the gap is small, leave average distillation alone for now. If large, adjust the reservoir or consider saved-policy averaging before changing regret learning.
6. **Only then spend on larger training.** Carry the winning, understood change to 30 claims with several seeds and properly trained BRs; then make a matched 69-claim continuation. Keep training time, iterations, number of independent deals, number of expanded action edges, and BR compute visible separately. Otherwise a faster implementation can look worse per iteration and better per hour, or vice versa.

There is one concrete replay audit worth adding early: [the device reservoir](../../liars_poker/algo/deep_cfr.py) draws replacement slots using default float32 `torch.rand` even when `seen` is in the billions. Its claimed exact uniformity deserves a high-counter statistical test and, if necessary, a float64 random draw. That is a correctness concern, **not yet an explanation** for the 69-claim curve.

## 8. How possible repairs relate to the evidence

If action estimates are too noisy, use a **low-variance action-value baseline** or more independent traversals where they matter; simply allowing a larger edge tensor does not fix variance per unit compute. [Variance-reduced MCCFR](https://arxiv.org/pdf/1809.03057) gives the underlying rationale. The CPU aggregate-then-clip result now gives a concrete reason to preserve raw sampled updates until after grouping by infoset; using it on a GPU would require a memory-bounded implementation and its own validation. If the bootstrapped network is still the issue after that change, compare against a conventional Deep CFR-style advantage memory and a regression target that does not mostly reproduce yesterday's network. If only the average network is weak, consider a reach-correct snapshot mixture or an SD-CFR-style policy construction. Each is a different repair for a different broken link; learning-rate decay alone cannot distinguish them.

The encouraging part is that the 18-claim exact results show the pipeline *can* learn strong policies. The unresolved question is which approximation ceases to track its exact counterpart as the history space, sampling variance, and iteration count grow. The sequence above is designed to locate that divergence rather than guess another six-hour schedule.

## 9. Check your understanding

Try each question before reading the answer. The numbers are deliberately
small and independent of a particular `GameSpec`.

1. **Exact regret update.** Yesterday's cumulative CFR+ regrets for two
   actions were `(0.4, 0)`. Today's counterfactual instantaneous regrets
   are `(-0.7, +0.2)`. What is the next regret-matched strategy?

   **Answer:** `R⁺ = (max(0, 0.4-0.7), max(0, 0+0.2)) = (0, 0.2)`, so the
   next current strategy chooses the second action. Adding only positive
   instantaneous regret would have given the wrong first entry.

2. **Sampled action.** A claim is included with probability `1/2`. Its
   sampled continuation value is `0.3` and its fixed baseline is `0.1`.
   What value does the inclusion-corrected estimate assign *when selected*?

   **Answer:** `0.1 + (0.3-0.1)/(1/2) = 0.5`. When unselected it assigns
   `0.1`. Their average is `0.3`. Neither individual estimate has to equal
   the true conditional action value.

3. **Scaled neural target.** At iteration `t=10,000`, an action's old
   scaled-regret prediction is `0.02` and its sampled instantaneous regret
   is `-1`. Before clipping, what target is produced?

   **Answer:** `(9,999/10,000)×0.02 - 1/10,000 = 0.019898`. The old
   prediction still dominates. Lowering Adam's learning rate would not
   change this target, though it would change the subsequent fit.

4. **Interpreting a graph.** Method A produces lower exploitability after
   2,000 CFR+ iterations, but each A iteration takes three times as long.
   Which method should you choose?

   **Answer:** The iteration graph alone cannot tell you. Compare exact
   exploitability at equal training time, and distinguish that from the
   quality of each update. If evaluation uses an approximate BR, match its
   training budget too.
