# Learned history baselines: estimating the actions you did not take

This note explains one idea from [VR-DeepDCFR+](https://arxiv.org/abs/2511.08174) ([code](https://github.com/rpSebastian/DeepPDCFR)), which it inherits from [VR-MCCFR](https://arxiv.org/pdf/1809.03057) and [DREAM](https://arxiv.org/abs/2006.10410). The question it answers:

**if a sampled trajectory follows only one action at a node, how can it still produce a low-noise estimate of the value of *every* action there?**

The answer is a *baseline*: a guess of each action's value, used so that only the guess's *error* is sampled. Surprisingly, the guess is allowed to see information the player cannot see, such as the opponent's cards, without making the estimate biased or leaking that information into play.

For our own traversal, see the [neural CFR+ code map](neural_cfr_plus_code_map.md). For what we have measured about sampled targets, see [sampled regret targets](../experiments/18_claim/2026-09-27_23-28-05_cfr_plus_sampled_targets_cpu.md).

## 1. The problem: one sample per node

A regret update at an information set needs an advantage for every legal action:

```text
advantage(a) = value(a) - Σ_b σ(b) value(b).
```

Our trainer gets these values by **external sampling**. It samples the deal and the opponent's actions, but expands **every** action of the traversing player. Each traverser action then has an actual sampled continuation. That is affordable at 18 claims and becomes the dominant cost at 69 claims; the action caps (`traverser_action_sample_schedule`) exist to limit it.

**Outcome sampling**, used by VR-DeepDCFR+, is cheaper: one path from deal to terminal, one action per node. At a traverser node, only one action `a*` has a sampled child value. The standard unbiased estimate divides that value by the probability `p(a*)` of having sampled it, and gives every other action zero:

```text
q(a*) = v_child / p(a*),     q(a) = 0 for a ≠ a*.
```

On average this is right. Each action is sampled with probability `p(a)`, and when sampled it counts `1/p(a)` times. Individual samples, however, are terrible: the untried actions look worthless, and the tried one is inflated.

## 2. The trick: sample only the error of a guess

Suppose we have any guess `b(a)` of each action's value. Use it for every action, and correct only the sampled one:

```text
q(a) = b(a) + 1[a = a*] · (v_child - b(a)) / p(a).
```

**This is unbiased for any `b`.** Taking the expectation over which action is sampled:

```text
E[q(a)] = b(a) + p(a) · (E[v_child | a] - b(a)) / p(a) = E[v_child | a].
```

The guess cancels. It can be stale, crude or wrong without biasing the estimate. What it changes is the **variance**. For a deterministic child value `v`, the estimate is `b + (v-b)/p` with probability `p` and `b` otherwise, so

```text
Var[q(a)] = (v - b)² · (1 - p) / p.
```

Without a baseline, `b = 0` and the variance is `v²(1-p)/p`. With a good baseline, `v - b` is small and the variance nearly disappears. A baseline cannot make an estimate *worse on average*. A bad one can make it noisier, when `|v - b| > |v|`.

## 3. A worked example

A traverser node has actions A, B and C, with current strategy `σ = (0.5, 0.3, 0.2)`. The sampler chose B, with probability `p(B) = 0.5`, and the subtree below B returned `+1`. Suppose the true action values are `(0.4, 1.0, -0.2)`, so the true advantages are `(-0.06, +0.54, -0.66)`.

| | Estimated action values | Node value `Σσq` | Estimated advantages |
| --- | --- | ---: | --- |
| No baseline | `(0, 1/0.5, 0) = (0, 2, 0)` | 0.60 | `(-0.60, +1.40, -0.60)` |
| Baseline `(0.4, 0.7, -0.2)` | `(0.4, 0.7 + 0.3/0.5, -0.2) = (0.4, 1.3, -0.2)` | 0.55 | `(-0.15, +0.75, -0.75)` |
| True values | `(0.4, 1.0, -0.2)` | 0.46 | `(-0.06, +0.54, -0.66)` |

Both rows are unbiased over repeated samples. The baseline row is far closer on *this* sample. Its B estimate would be 1.3 when B is sampled and 0.7 when it is not: a spread of 0.6 around the true value 1.0. The no-baseline estimate would be 2 or 0. By the formula above, the variances are `0.3² × 1 = 0.09` and `1² × 1 = 1`: an elevenfold reduction.

## 4. Why the baseline may look at the opponent's cards

Unbiasedness in section 2 required only that the guess `b(a)` be fixed before the action is sampled. It does **not** require the guess to use only the acting player's information. The baseline is part of the *estimator*, not part of the *policy*: regret matching never reads it, so it cannot leak hidden cards into play.

This matters because values are much easier to predict from the full history, with both hands known, than from an information set. An information-set value averages over every opponent hand consistent with the claims. A history value does not. VR-DeepDCFR+ therefore trains `Q(h, a)` on the full history: in OpenSpiel terms, the concatenation of both players' information-state tensors. In our game, once both hands are known, whether any claim is true is **deterministic**. The value of `CALL` at a history is then just the terminal payoff, with nothing to learn. Only the values of further claims depend on future play.

## 5. Applying it along a whole trajectory

The trick is applied at **every** decision node on the sampled path, from the terminal back to the root. The corrected node value `v = Σ_a σ(a) q(a)` is what gets passed up to the parent as `v_child`:

- **At opponent nodes**, the sampled action comes from the opponent's own strategy, so `p(a) = σ_opp(a)`. The correction reduces noise in the value passed up. It does not change any unbiased expectation.
- **At traverser nodes**, the sampled action comes from an exploration mixture: `0.6 × uniform + 0.4 × σ` in the paper's configuration. The full vector `q - v` is recorded as that information set's sampled advantage. Actions not taken get their advantage from the baseline, and the taken one from the corrected estimate.

This is the variance-reduced MCCFR estimator of Schmid et al.: one sample per node and an unbiased value for every action. Its variance depends on how good the baselines are, not on how many actions were skipped. The paper's implementation samples chance events without a correction. In our game chance acts only at the deal.

## 6. How VR-DeepDCFR+ learns Q

- **Data.** Every transition of every sampled episode goes into a 1,000,000-entry circular buffer that persists across iterations: full history, action, next history, next information set, legal mask, reward and whether it is terminal.
- **Target.** An expected-SARSA backup under the strategy about to be played:

  ```text
  Q(h, a) ← reward + Σ_a' σ_{t+1}(a' | I') · Q_target(h', a').
  ```

  Here `σ_{t+1}` is computed from the **just-updated** regret networks, so the baseline tracks the next iteration's play. Because the target reweights by the current strategy, old transitions from other strategies can be reused: the learning is off-policy.
- **Schedule.** After each player's regret update, Q is **reinitialised and retrained** for 1,000 steps. The target network syncs every 50 steps, and the lowest-loss snapshot is kept.
- **Symmetry.** Q predicts Player 1's value; Player 2 uses its negation, as the game is zero-sum.

None of this affects correctness. A poorly fitted Q gives noisier advantages, not biased ones. The practical risk is noisy estimates, from an off-policy target that lags a changing policy or from under-training.

## 7. What it buys and what it costs

| Buys | Costs |
| --- | --- |
| Episode cost roughly proportional to game length rather than to the number of expanded traverser actions | One Q inference per visited node, and a third network to train |
| Advantages for all actions from one path | Exploration of the traverser's own actions is still needed, because sampling probabilities enter the correction |
| Lower variance, which also lowers the bias from clipping a noisy estimate at zero | The variance reduction is only as good as Q; early in training Q is poor |

The last "buys" item links to our [clip-order findings](../experiments/18_claim/2026-09-28_13-29-00_18_claim_target_sampling_factorial.md). Clipping a noisy quantity at zero biases it upwards by roughly the noise scale. Reducing variance at the source reduces that bias even where no aggregation is possible, such as information sets visited once per iteration.

## 8. Where this fits in our code

**At 18 claims** with full traverser expansion, every traverser action already gets a genuine sampled value, so traverser nodes need no baseline. The remaining noise comes from the sampled deal and the opponent's sampled actions. A baseline at **opponent** nodes could reduce it.

**At 69 claims**, where traverser claims are capped, the GPU traverser already contains a primitive version of this idea. `traverser_action_baseline="call"` in [`neural_cfr_plus_gpu.py`](../../liars_poker/algo/neural_cfr_plus_gpu.py) sets each **unselected** claim's baseline to the value of calling at the current history, computed from both sampled hands. It then corrects only the selected claims by their inverse inclusion probabilities. That is a **full-information baseline**, legitimate for the reason in section 4. It is exact for `CALL`, but a crude guess for claims, whose values depend on how play continues. A learned `Q(h, a)` replaces that crude guess with a real prediction. It could make aggressive claim caps, or full outcome sampling, workable at 69 claims.

**Encoding.** A history input for our game is easy to build: both hands' rank counts plus the public claim bits. That is 2 × 6 + 69 = 81 features at 69 claims. The terminal value of `CALL` needs no network at all.

## 9. Check your understanding

1. **Unbiasedness.** An action is sampled with probability 0.25. Its baseline is 0.2 and its sampled child value is 0.6. What is its estimated value if sampled, and if not? What is the expectation, if 0.6 is the true value?

   **Answer:** Sampled: `0.2 + (0.6 - 0.2)/0.25 = 1.8`. Not sampled: `0.2`. Expectation: `0.25 × 1.8 + 0.75 × 0.2 = 0.6`.

2. **Variance.** For the same action, compare the variance with baseline 0.2 and with no baseline.

   **Answer:** `(1-p)/p = 3`. With baseline: `0.4² × 3 = 0.48`. Without: `0.6² × 3 = 1.08`.

3. **Hidden information.** Why is it acceptable for `Q` to use the opponent's cards, when the regret network must not?

   **Answer:** The regret network's output chooses actions, so it must depend only on the information set. The baseline only enters an estimator whose expectation does not depend on it. It is never used to choose an action.

4. **A bad baseline.** Can a very wrong baseline bias the regret targets?

   **Answer:** No. It can increase their variance when `|v - b|` exceeds `|v|`, but the expected estimate is unchanged. Our clipped targets are the exception: clipping is nonlinear, so extra variance *does* raise the expected clipped value.
