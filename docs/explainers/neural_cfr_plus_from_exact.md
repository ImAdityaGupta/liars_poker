# From exact CFR+ to this repository's neural CFR+

This guide is meant to answer a specific question: **what, exactly, did we replace when we moved from a tabular CFR+ solver to neural networks, and how could those replacements stop helping as training continues?** It describes the implemented algorithm, not a generic promise that all algorithms called “Deep CFR” behave alike.

Start with [the runnable orientation notebook](../../notebooks/sep_2026/repo_reorientation.ipynb) if the game rules or artifacts are unfamiliar. For an entry point into the implementation, see the [neural CFR+ code map](neural_cfr_plus_code_map.md). The most relevant implementation files are [the game and rules](../../liars_poker/env.py), [exact dense CFR+](../../liars_poker/algo/cfr_plus_dense.py), [the neural trainer](../../liars_poker/algo/deep_cfr_plus.py), and [GPU traversal](../../liars_poker/algo/neural_cfr_plus_gpu.py).

CPU experiments carried out after this guide are recorded in [sampled regret targets](../experiments/2026-09-27_23-28-05_cfr_plus_sampled_targets_cpu.md), [the exact shadow ledger](../experiments/2026-09-27_23-58-30_cfr_plus_shadow_neural_cpu.md), [the neural clip-order intervention](../experiments/2026-09-28_01-04-50_cfr_plus_neural_clip_order_cpu.md), [its longer run](../experiments/2026-09-28_01-17-33_cfr_plus_neural_clip_order_long_cpu.md), and [the depth target audit](../experiments/2026-09-28_01-47-35_cfr_plus_neural_depth_target_audit_cpu.md). Each note states the question, possible outcomes, measured results, and limits.

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

### A. Sampling a traversal

The trainer draws a batch of complete private deals. For one traversing player, it explores their available actions at a reached decision while sampling one action from the other player's current strategy. This is **external sampling**: “external” chance and opponent choices are sampled; the traverser's choices are expanded. Each leaf reached by a call has an exact payoff for that sampled deal. Backing up those payoffs gives noisy estimates of the action values at the visited infosets.

“Full action expansion” does **not** mean exact CFR: opponent actions and private deals are still sampled. In the GPU code, even traverser actions may be capped: with cap 16, at most 16 legal *claims* are sampled at a traverser decision; `CALL`, when legal, is handled exactly. Claims with fewer than 16 legal alternatives are fully expanded. To keep an action-value estimate unbiased *before later nonlinear operations*, a selected claim with inclusion probability `q` gets an inverse-`q` correction:

```text
estimated_value(a) = baseline(a)
                   + selected(a)/q(a) · [sampled_child_value(a) - baseline(a)].
```

With no baseline, an unselected claim contributes zero to that sample's action-value vector. At deeper traverser nodes, reaching the node also depended on earlier action selections; an inverse product of their inclusion probabilities weights the training record. [The sampler and correction](../../liars_poker/algo/neural_cfr_plus_gpu.py) implement these ideas. The streamed traversal breaks expanded edges into chunks to bound *live GPU memory*; it does not reduce the total number of sampled edges or their statistical variance.

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

This is the most important difference to hold in mind: a neural training loss can be tiny while the regret estimate is strategically wrong. The target is mostly an old prediction when `t` is large. At `t = 10,000`, a sampled instantaneous regret of `1` contributes only `0.0001`; at `t = 30,000`, it contributes `0.0000333`. Model error, numerical scale, or optimizer drift of comparable size can overwhelm the new signal. Low MSE against *self-generated* targets is not an independent correctness check.

There is also a nonlinear sampling issue. Imagine the true expected instantaneous regret for one action is zero, with a sample of `+1` or `-1` equally likely and zero previous regret. Averaging samples first and then applying CFR+ gives `ReLU(0) = 0`; clipping each sampled target first and then fitting its mean gives `(ReLU(+1) + ReLU(-1))/2 = 0.5`. The implementation uses the latter order. This does **not** by itself prove that its sampled process cannot converge, but it shows why an unbiased pre-clipping value estimator is insufficient to claim equivalence to the exact CFR+ update. This gap grows in relevance when action sampling makes individual estimates noisy.

### C. Fitting the average strategy networks

During a traversal for one player, the other player's visited decisions generate records of that other player's *current strategy*. Sampling that player's own preceding actions naturally visits some histories in proportion to their own reach. When traverser action sampling removed paths, the record receives inverse-inclusion weighting; it also receives the CFR+ iteration weight `t`. A [reservoir buffer](../../liars_poker/algo/deep_cfr.py) keeps a limited, approximately uniform sample of historical records. The [strategy fitting loss](../../liars_poker/algo/deep_cfr_plus.py) trains two separate networks with cross-entropy to reproduce those distributions.

The resulting `average_policy()` is a *distilled approximation* to the reach-weighted average, not an average-regret network and not an exact historical sum. Its error can be separate from regret-learning error. The original [Deep CFR paper](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf) uses an average-strategy network; [Single Deep CFR](https://arxiv.org/pdf/1901.07621) discusses the additional approximation introduced by that network and an alternative based on saved iteration policies. Those papers are useful comparisons, but their algorithms and compute budgets are not identical to this repo's online neural CFR+.

## 4. Current strategy, average strategy, and evaluation

There are now **two different ways to play**. `NeuralRegretMatchingPolicy` computes the *current* policy from the two regret networks. `NeuralPolicy` uses the two strategy networks to play the *learned average*. In two-player zero-sum CFR, the average is normally the policy of interest; a worse and noisier current policy is not automatically a failure. But a current policy that deteriorates can be an early warning that regret estimates are going astray.

On small games, we can compile a neural policy to a dense table and compute an **exact** best response. On 30- and 69-claim games, that dense enumeration is too expensive; instead, we train an approximate responder against the frozen policy and Monte Carlo-evaluate the responder. The repo's exploitability convention is `p_first + p_second - 1`, where each probability is a responder win rate in one seat. The approximate result is *discovered* exploitability: a weak responder can underestimate a policy's true exploitability. The confidence bound covers the Monte Carlo evaluation of the **found responder**, not the possibility that the responder missed a better strategy. Five-, ten-, twenty-, and sixty-minute responders are different measurements and must be labeled separately.

This matters for the old 69-claim trajectory. Under the same 20-minute BR procedure, the estimate improved to about `0.262` for the 400-minute policy, then was about `0.280` for the 720-minute policy. Under 60-minute BRs the corresponding selected-snapshot values were about `0.282` and `0.301`. That is evidence of worsening *discovered* exploitability under matched responder budgets, though it is not exact exploitability and is not a multi-seed conclusion. The 18-claim exact-evaluation experiments make a genuine learning problem more plausible.

## 5. A map from assumptions to failure modes

| Exact algorithm assumes | Approximation here | What can go wrong | An informative test |
| --- | --- | --- | --- |
| Correct terminal payoff and legal history | GPU tensors and packed history | A rule or history bug quietly changes the game | Compare GPU terminal and legal-action outputs with `Env` across random deals and late histories, including claims above index 63 |
| Correct expected action values | Sampled deals, opponent actions, and sometimes own claim actions | High variance or wrong inclusion probability | Freeze a small-game policy; compare sample means and variances with exact action values, by infoset and depth |
| CFR+ clips *cumulative* regrets | Clips each noisy per-record target before regression | Sampling/clipping interaction can change the expected target | Compare `mean(clip(sampled update))` with `clip(mean(sampled update))` against a small-game exact oracle |
| Old cumulative regret is exact | Previous neural prediction stands in for old regret | Self-confirming error and vanishing `1/t` innovation | Compare predictions with independent exact regret targets and action rankings across iteration `t` |
| Every infoset can be represented and revisited | Fixed-size network, fresh regret records, finite optimizer steps | Rare critical infosets have poor fit while average MSE looks good | Stratify oracle errors by depth, legal-action count, reach, and BR-discovered weakness |
| Average is a correct own-reach-weighted sum | Sampled records, finite reservoir, second network | The output policy differs from the true historical average | On 18 claims, compute both exactly for the same iteration policies and compare exact exploitability |
| Exploitability is measured accurately | Finite-time learned responder | Apparent improvement or plateau reflects responder skill | Use matched budgets, several responder seeds, and longer curves on frozen snapshots |

### Why particular old checks did not settle the question

The streamed traversal validation established valuable **implementation equivalence** between two traversal paths on selected small-spec samples. It compared root values and record counts. Both paths can agree and still share the same sampled-target or fitting flaw. Likewise, held-out regret MSE is measured against targets generated by the old network, not an independent table of true counterfactual regrets. In the 69-claim adaptive run, the late regret validation had only dozens of records per player at a monitor point, despite very low reported MSE. These observations motivate stronger semantic tests; they do not invalidate the checks already performed.

## 6. The diagnostic sequence I would actually run

1. **Check semantics before training.** On the 18-claim game, choose frozen tabular policies and exact infosets. Compare terminal truth, legal masks, backed-up values, and GPU packed histories with the exact CPU implementation. Repeat for random 69-claim terminal states to cover the two-word packed history. This catches mechanical mistakes without paying for a long run.
2. **Isolate sampling and clipping.** For those same frozen policies, collect many independent traversal samples and check whether *pre-clipping* action values converge to exact values. Measure variance and effective sample size at cap 16, cap 24, and full traverser expansion. Then measure the post-clipping gap above. If pre-clipping values are wrong, fix sampling; if they are right but post-clipping targets differ substantially, address variance or update design.
3. **Remove the neural net from the experiment.** Run a small-game tabular regret updater using the *same sampling rule* and, where possible, common random seeds; evaluate it exactly over time. Once the two updaters choose different policies, they cannot literally follow identical self-play trajectories, so compare learning curves across multiple seeds. If tabular sampling also plateaus or deteriorates, the sampler/CFR+ combination is the priority. If it succeeds while neural CFR+ fails, investigate representation, target bootstrapping, optimizer steps, and fallback behavior.
4. **Test the network against an independent oracle.** Instrument a short small-game run to retain a tabular “shadow” cumulative-regret ledger while it follows the neural policies. Exact enumeration can then compute each frozen policy's instantaneous counterfactual regrets and update that ledger independently of the network's predictions. Compare the network's regret *signs, action ordering, and resulting policy* against this oracle at important infosets, including ones selected by an exact BR. Record how often the policy falls back to uniform. A falling self-target MSE alongside worsening oracle error would directly demonstrate self-confirming training.
5. **Isolate averaging.** Instrument that 18-claim run to accumulate the true own-reach-weighted average of its neural current policies as a dense table. Compare its exact exploitability with the strategy network's exact exploitability. If the gap is small, leave average distillation alone for now. If large, adjust the reservoir or consider saved-policy averaging before changing regret learning.
6. **Only then spend on larger training.** Carry the winning, understood change to 30 claims with several seeds and properly trained BRs; then make a matched 69-claim continuation. Keep training time, iterations, number of independent deals, number of expanded action edges, and BR compute visible separately. Otherwise a faster implementation can look worse per iteration and better per hour, or vice versa.

There is one concrete replay audit worth adding early: [the device reservoir](../../liars_poker/algo/deep_cfr.py) draws replacement slots using default float32 `torch.rand` even when `seen` is in the billions. Its claimed exact uniformity deserves a high-counter statistical test and, if necessary, a float64 random draw. That is a correctness concern, **not yet an explanation** for the 69-claim curve.

## 7. What I would change only after the tests

If action estimates are too noisy, use a **low-variance action-value baseline** or more independent traversals where they matter; simply allowing a larger edge tensor does not fix variance per unit compute. [Variance-reduced MCCFR](https://arxiv.org/pdf/1809.03057) gives the underlying rationale. If clipping noisy per-record targets is the main issue, test a regret-update design that aggregates more independent samples before clipping, or a raw-advantage method with independently stored targets. If the bootstrapped network is the issue, compare against a conventional Deep CFR-style advantage memory and a regression target that does not mostly reproduce yesterday's network. If only the average network is weak, consider a reach-correct snapshot mixture or an SD-CFR-style policy construction. Each is a different repair for a different broken link; learning-rate decay alone cannot distinguish them.

The encouraging part is that the 18-claim exact results show the pipeline *can* learn strong policies. The unresolved question is which approximation ceases to track its exact counterpart as the history space, sampling variance, and iteration count grow. The sequence above is designed to locate that divergence rather than guess another six-hour schedule.
