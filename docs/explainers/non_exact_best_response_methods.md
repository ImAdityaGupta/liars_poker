# Best responses without exact evaluation

**Status: proposal for review (1 October 2026).** Nothing here is implemented yet. This note explains the families of best-response (BR) method we could use once exact evaluation is no longer possible. It also explains why this game makes several of them unusually cheap, and how to calibrate them on the 18-claim game before relying on them at 30 claims and above.

## Why this matters, and what precision we need

Exact exploitability is affordable up to 18 claims. There, the best training runs approach 10⁻³. Larger games will not get close to that: for the full game, a true exploitability near **0.02** would be an excellent outcome. The long 18-claim runs are not about reaching 10⁻³ at scale. They are a test bench for **training-process lessons** that should scale: cumulative rather than `/t` regret targets, better average-policy refits, root-count schedules and regret-network fitting.

The evaluation target at 30 claims is therefore different from at 18:

- Expected exploitability is roughly **0.01–0.05**. An evaluator that recovers most of the true value, say within 10–20%, is useful. One accurate to 10⁻³ is not required.
- **Ranking fidelity matters most.** We mainly need to know whether recipe X beats recipe Y, and whether a run is still improving.
- Calibration on 18 claims should cover policies across **about 0.004–0.05**, using older snapshots as well as the newest. Policies near 0.004 alone would be unrepresentative.

Our current approximate evaluator, the action-conditioned fitted-return responder, recovered about 94–97% of exact exploitability after 15 minutes on the 18-claim game. Its absolute miss was about 0.0024–0.005. That miss is comparable to today's best 18-claim policies, but only a modest fraction of the values expected at 30 claims.

## The structural advantage: beliefs are exactly computable

Against a **fixed** policy σ, the only hidden information is the opponent's hand. With suit symmetry there are few hand types: 10 at 18 claims, 35 at 30 claims, 126 at 69 claims. There are more card-level combinations, because of blockers, but the dense evaluator already handles that.

For a responder holding `h` who has seen public history `H`, the posterior over the opponent's hand `o` is exact:

```text
b(o | h, H) ∝ P(o | h) × Π over the opponent's past decisions σ(action | o, history prefix)
```

- `P(o | h)` is the deal probability given card blockers.
- Only the **opponent's** actions update the belief. The responder's own claims are public, but σ does not see the responder's hand.
- Updating costs one batched policy evaluation, over all opponent hands, per opponent decision. That is cheap on CPU.

So the BR problem is a **single-agent planning problem with a known, tractable belief state**:

- opponent moves come from σ, which we can query exactly;
- a `CALL` is resolved deterministically once both hands are known;
- claims strictly increase, so the game tree is finite and bounded by the number of claims.

The dense exact evaluator computes the same beliefs through its likelihood tables. The methods below do so lazily along played or searched lines, which keeps them affordable when `2^claims` histories are too many to enumerate.

## The exact recursion every method approximates

At a responder decision with history `H`, where the last claim `L` was made by the opponent:

```text
V(CALL) = Σ_o b(o) · (+1 if L is false given (h, o), else −1)

V(c)    = Σ_o b(o) [ σ(CALL | o, H+c) · (+1 if c is true given (h, o), else −1)
                     + Σ_c' σ(c' | o, H+c) · V*(H+c+c', b') ]
```

Here `b'` is the belief updated by the opponent's raise `c'`, and `V*` is the responder's optimal value at its next decision. Applying the recursion all the way down is the **exact best response** (expectimax over the responder's choices, expectation over the opponent's hand and actions).

Every method below replaces `V*` with something cheaper. Each one still produces an actual playable responder strategy, so **its measured win rate against σ is a valid lower bound on exploitability**. Exploitability in this repository is the first-seat BR win probability plus the second-seat BR win probability, minus one. Each seat is bounded separately.

## Method 1: local best response (LBR)

Replace `V*` at the next decision with a simple continuation, typically **"call at our next turn"**. Its value is exact under the updated belief.

At each of its decisions, the responder computes:

- `V(CALL)` exactly;
- for each legal claim `c`: σ either calls (resolved exactly) or raises to `c'`, after which the responder calls `c'` (also exact).

It plays the best of these, then recomputes from the new belief at its next decision.

- **Cost:** one batched policy evaluation per candidate claim. At 30 claims that is about 30 × 35 ≈ 1,000 network rows per decision: milliseconds on one CPU core.
- **Strengths:** no training; exact beliefs; the decisive `CALL` choices are handled perfectly. Policies that mishandle calls are exposed immediately.
- **Weakness:** it is **myopic**. It cannot find lines that need two or more steps, for example "bluff now, because σ over-calls after a second raise".

This is the standard cheap evaluator in poker research ([Lisý and Bowling, 2017](https://arxiv.org/abs/1612.07547)), adapted here with exact beliefs.

## Method 2: depth-limited expectimax

Extend the recursion to `d` levels of the responder's own decisions. At the leaves, use the call-next value of Method 1 or a learned value (Method 4).

- **It is exact once `d` covers the rest of the game**, so `d` is a dial from LBR (`d = 1`) to the exact BR.
- **Opponent pruning:** skip opponent responses whose total probability `Σ_o b(o) σ(c' | o, ·)` is below `ε`. Values lie in [−1, 1], so the error at that node is at most about twice the skipped mass. That is an accountable loss. Trained policies are concentrated, so pruning cuts the opponent's effective branching to a handful of responses.
- **Responder branching** is all legal claims plus `CALL`. It shrinks deeper in the game, since only higher claims stay legal.
- **Cost:** roughly (responder branching × pruned opponent branching)^d network rows per decision. At 30 claims:
  - `d = 2` is about 30 × 30 continuations of 35 hands, roughly 30,000 rows: a few milliseconds per decision;
  - `d = 3` is roughly 30 times more, about a million rows: perhaps 0.1–1 s per decision on one core.

  Games have a handful of responder decisions, so this is thousands of games per core-hour, and it parallelises perfectly across cores.

### Removing Monte Carlo noise: evaluate the search policy exactly

Given its search settings, the responder is **deterministic**. Instead of estimating its win rate from random games, enumerate:

- every responder hand, weighted by deal probability;
- every opponent hand, weighted by the belief;
- every opponent response above `ε`, weighted by σ.

Follow only the responder's **chosen** action at its own nodes. This gives the search policy's **exact value** up to the pruning error, with no Monte Carlo noise. The enumeration tree is small because the responder contributes a single branch at each of its nodes.

For comparison, sampling gives a standard error of about `sqrt(0.25 / n)` per seat: `n = 100,000` games is needed for about ±0.0016 per seat. Exact enumeration removes this cost and the noise.

## Method 3: sampled tree search over the responder's information states

This is Monte Carlo tree search (MCTS) on the same belief problem.

- **Tree nodes** are the responder's information states: its own hand (fixed at the root) and the public history.
- **Each simulation** samples an opponent hand from the belief, then opponent actions from σ given that hand.
- **At responder nodes**, an upper-confidence rule (UCT) chooses which claim or `CALL` to explore.
- **At leaves**, use a rollout (call-next, or a simple heuristic policy) or a learned value.

It converges to the exact expectimax value as simulations increase. Compared with fixed depth, it **spends compute where it matters**: promising multi-step lines are followed deeper, and hopeless claims are abandoned early.

The cost is simulations × policy evaluations. Cache σ by (public history, opponent hand): the same public histories recur constantly across simulations and games, so the cache hit rate should be high.

## Method 4: a trained responder with search at play time

Train the existing [fitted-return responder](../../liars_poker/algo/br_fitted_return_action_conditioned.py) for a short budget on the GPU. Then use its value estimates `Q(h, H, a)`:

- at the leaves of Method 2, or
- as priors, rollout policy or leaf values in Method 3.

If the leaf values were the exact values of following the trained responder afterwards, a one-step lookahead on top of it would be **at least as good** as the trained responder itself (the policy-improvement theorem). With approximate values this holds approximately.

In practice, training gets the long-range strategy roughly right, and search with exact beliefs corrects local mistakes, especially `CALL` decisions, which it evaluates exactly. The network is trained once per target policy; search is applied whenever it plays. This is the natural candidate for 69 claims, where pure depth-limited search runs out of depth.

## Method 5: extrapolating to the converged value, safely

Methods 2–4 each have a **dial that provably converges to the exact BR**: depth `d`, or simulations per move `n`. Measure discovered exploitability `e(c)` at several settings `c` and fit:

```text
e(c) = e_∞ − A · c^(−α)
```

`e_∞` is the constant that makes `log(e_∞ − e(c))` a straight line against `log c`. It estimates the exploitability we would discover with unlimited compute, which for these methods is the true exploitability.

This is safer than extrapolating a **learned responder's training curve**. In June, power-law extrapolation of the DQN responder predicted that responder's own plateau, not the true BR value: a training curve carries no guarantee of approaching the exact answer. The search dial does.

Caveats:

- Depth is discrete and its cost grows exponentially, so simulations per move is the smoother dial.
- Report `e_∞` as an **estimate**. Only the measured points are **bounds**.
- The functional form must be validated at 18 claims: fit on cheap settings, predict expensive ones, and compare with exact exploitability.

## Method 6: take the maximum, with a held-out check

Every method gives a lower bound **for each seat**. Taking each seat's maximum across methods, then summing the two seats, is still a lower bound, and usually a tighter one. Different seats may be won by different methods.

Selecting the maximum of many noisy estimates overstates the result. So choose the best method or seed on one evaluation, then confirm it on fresh deals. Better still, confirm it by exact enumeration of the search policy (Method 2), which has no sampling noise at all.

## Expected strengths and costs

| Method | Training needed | Main cost | Tightness | Best use |
| --- | --- | --- | --- | --- |
| 1. LBR | None | Milliseconds per decision, CPU | Lower; misses multi-step lines | Fast screening; exposing poor `CALL` play |
| 2. Depth-limited expectimax (`d = 2–3`) | None | Up to about a second per decision, CPU | Good; exact as `d` grows | Main evaluator at 30 claims, likely |
| 3. Sampled tree search | None | Simulations × policy evaluations, CPU | Good; converges with simulations | Long or uneven lines; extrapolation dial |
| 4. Trained responder + search | Short GPU training per target | Training plus search at play time | Strongest | 69 claims; anything search alone cannot reach |
| 5. Extrapolation | As for its method | Several settings of the dial | An estimate, not a bound | Seeing how close discovered values are to converged |
| 6. Maximum across methods | As for its components | Sum of components | Tightest bound available | Final reported numbers |

Every method except the learned responder's training runs on CPU and parallelises across games, responder hands and cores. CPU time on the VM, or on cheap extra rentals, scales them directly.

## Calibration plan at 18 claims

Exact answers are available at 18 claims, so every method can be calibrated before it is trusted elsewhere.

1. **Choose about eight snapshots spanning 0.004–0.05** exact exploitability. Include current O4 policies, older online-average policies and early-iteration policies.
2. **For each method and compute setting**, record:
   - the fraction of exact exploitability recovered, per seat and total;
   - the cost in CPU core-hours (and GPU minutes for Method 4);
   - for the search methods, the exact-enumeration value as well as a sampled estimate, to confirm they agree.
3. **Ranking fidelity:** for pairs of snapshots and pairs of recipes, does the method order them the same way as exact evaluation? This decides whether a method is usable for choosing between training recipes.
4. **Extrapolation:** fit `e_∞` from the cheaper settings and compare it with exact.
5. **Baseline:** include the fitted-return responder at its usual budgets (about 5, 15 and 60 minutes) for comparison.

Then move to 30 claims. The belief over 35 hand types is still cheap there. Run the calibrated winners alongside the fitted-return responder. Agreement between independent methods is the best available check where exact evaluation is not.

## Open questions

- How far can depth-limited search go at 30 and 69 claims before cost explodes? Typical games under good policies are short, but the responder must still consider long lines.
- How concentrated are trained policies in practice, and so how much does opponent pruning save, and at what accounted loss?
- Is σ evaluation, batched on CPU, the dominant cost? If so, caching by public history and opponent hand is the key optimisation.
- Can a cheap upper bound on exploitability be found for any useful subclass of policies? Every method here gives lower bounds only.

## Related notes

- [Learned history baselines](learned_history_baselines.md): the same exact-belief idea used for variance reduction during training.
- [GPU training schedules](gpu_training_schedules.md).
- [Neural CFR+ from exact CFR+](neural_cfr_plus_from_exact.md): the existing exact and approximate evaluation conventions.
