# 18-claim: bootstrapped versus teacher-forced regret networks (N/T)

**Status: running on the VM (2 October 2026).** N and T run concurrently on the RTX 4060 Ti to 24,000 iterations. Exact exploitability runs asynchronously on CPU.

## Summary

- **Why neural CFR+ trails the table.** At K=4,096, neural regrets are about 1.6× worse than a table once O4's averaging cost is removed. Both neural runs also get **worse** late: K=4,096 after about 15k iterations, K=1,024 after about 23k. Averaging has been ruled out as the cause of the late rise. O4 is still 1.24× the exact average on a 31.5k-iteration table run ([long-run check](2026-10-02_12-29-46_18_claim_average_fit_long_run_check.md)).
- **Hypothesis.** The network builds each target from its own previous output (**bootstrapping**). Each fit's errors are therefore never corrected, and they pile up. The true regrets stay bounded, so the accumulated error grows relative to the signal until the policies it produces get worse.
- **Test.** Two neural runs with identical settings and an exact average:
  - **N (bootstrapped)**, as now;
  - **T (teacher-forced)**, where the network is refitted each iteration to the true cumulative regret kept in a table.

  Both also keep that table as a **shadow**, so the network's accumulated error can be measured directly over time.

  Both arms run concurrently on the GPU to 24,000 iterations, with snapshots every fixed number of iterations (at least 5 minutes apart).
- **What it decides.**
  - If T matches the table and doesn't rise while N does, bootstrapping is the cause, and the 30-claim method needs a regret target that doesn't build on its own output.
  - If T also lags or rises, the network is the problem as a reader of regrets, and the fix lies in fitting (for example, a learning rate that decays over the run).

## Background: what bootstrapping does

### How a table and the network update regrets

At every iteration, both see the same sampled conditional advantage ĝ at the information sets visited that iteration. For player 1 that is about 4,000 of roughly 400,000 reachable sets.

- **Table.** R(I) ← max(0, R(I) + ĝ(I)) at visited sets. Every other row is left exactly as it was.
- **Network (bootstrapped).** The target at visited sets is max(0, R̂(I) + ĝ(I)), where R̂ is **the network's own current output**. The fit is 24 Adam steps on batches of 1,024 rows drawn from this iteration's visits. The previous targets are discarded. The network is the only record of the cumulative regret.

Each fit is imperfect in two ways:

1. **Partial fits at visited sets.** Twenty-four steps at batch 1,024 is about half a pass over player 1's roughly 52,000 visit rows. An information set's share of the gradient is proportional to how often it was visited. Rarely visited sets receive only part of their increment, or none.
2. **Changes at unvisited sets.** The weights are shared, so fitting the visited sets also moves the outputs at the roughly 99% of sets that were not in the loss. The Part C audits measured this: at unvisited sets the network moves the policy by a third to a half of a true step. For player 2, those moves are unrelated to the correct update.

Because the next target starts from the network's output, **neither error is ever corrected.** Nothing in the method remembers the true sum.

### Why this could get worse over time

- **The signal stays bounded.** Regret-matching+ bounds the positive cumulative regret at an information set by about √T. With exact updates it typically settles or oscillates: dominated actions are clipped to zero, and the best action stops gaining once the policy stops playing worse actions. With sampled updates, the overall scale of an information set's regrets can drift. The policy depends only on ratios, so nothing pulls the scale back, and noise can make it grow roughly like √t. Either way, the signal grows no faster than about √t.
- **The accumulated error keeps growing.** Each fit adds an error that stays in the network. Random per-fit errors add up like a random walk, about √t. Systematic ones, such as the same rarely visited sets always under-updated, add up like t.
- **So the ratio of error to signal does not shrink, and may grow.** A table has none of this error. Part B showed that the average responds within a few hundred iterations when iterate quality falls ([step-early](2026-10-01_10-25-57_18_claim_average_fit_traversal_schedule_regret_noise.md#order-matters-the-average-forgets-old-phases-quickly)). Growing error would therefore make the average turn upward instead of levelling off.
- **It also explains the order of the turns.** Accumulated network error overtakes sampling noise sooner when sampling noise is small. K=4,096 turned at about 15k iterations, K=1,024 at about 23k.

This is a hypothesis. The Part C audits measured single updates at about 2k and 6k iterations, which cannot show accumulation.

## Design

### Arms

All arms use seed 17 and the 18-claim game (`ranks=4, suits=4, hand_size=2`, claims `RankHigh`, `Pair`, `TwoPair`, `Trips`, suit symmetry). They use the recipe of `neural_o4_k4096`:

- K=4,096 roots per player;
- cumulative conditional regrets, aggregate then clip, plain masked MSE (`regret_positive_weight=0`);
- 512×512 regret networks, learning rate `1e-3`, batch 1,024, 24 regret steps per update, 4,000,000-row regret buffer;
- batched `gpu_native` traversal with traversal batch 512.

| Arm | Regret store | Network is fitted to | Who plays | Averaging |
| --- | --- | --- | --- | --- |
| `exact4096` (exists) | table | n/a | table | exact linear average |
| **N: bootstrapped** | network, **plus a shadow table** | max(0, own output + ĝ), as now | network | exact linear average of the network's policy |
| **T: teacher-forced** | network, **plus a shadow table** | **the shadow table's updated value** at each visited set | network | exact linear average of the network's policy |

**The shadow table.** It accumulates the same ĝ as the network sees, using the table update above: per-set mean over the iteration's visits, then clip, exactly as in `TabularRegretFork._train_regret`.
- In both N and T it never affects play. The network plays and generates the data.
- In T it supplies the fit targets.
- In N it is used only for measurement: it is the true CFR+ regret of the sequence of policies N actually played. The gap between N's network and its shadow table is N's **accumulated error**, measured directly.

N and T differ in one thing only: **the target** the network is fitted to. They have the same fit budget, the same per-visit rows, the same partial fits and the same interference at unvisited sets. The difference is whether a later fit can overwrite an earlier error with the true value.

**Averaging.** Both arms use the exact own-reach-weighted linear average of the policy the network actually played, which is the same averaging as `exact4096`. This makes all three arms directly comparable, without O4's roughly 1.2× and its seed noise. No online average network, strategy reservoir or O4 refit is needed.

### Hardware and coordination

**Both arms run on the GPU, at the same time, in separate processes.** This follows Part C, where four CUDA arms shared the RTX 4060 Ti 16 GB in separate tmux sessions ([`launch_cfr_plus_18_regret_noise.sh`](../../../scripts/launch_cfr_plus_18_regret_noise.sh)).

- **On the GPU, per arm:**
  - traversal (the Part C CUDA path);
  - the regret fit;
  - the shadow table (2¹⁸ histories × 10 rank-count hands × 19 actions, about 200 MB);
  - the exact average (below).
- **Memory.** Each arm needs about 2–3 GB: the 4M-row regret buffer, the table, the float64 average sum (about 400 MB), the dense current policy and the CUDA context. Two arms fit comfortably in 16 GB.
- **On the CPU:** only exact exploitability evaluation. Snapshots are written to disk and evaluated by separate CPU worker processes, as in Part B and Part C. Two evaluator workers should keep up; the cadence below gives each arm two evaluations per snapshot.
- **Sharing the card.** Two processes slow each other, and not necessarily equally. That doesn't bias the comparison, because everything is compared **by iteration**, and the stopping rule below is in iterations. The [distillation experiment](2026-10-02_13-54-40_18_claim_regret_table_distillation.md) can share the card too, for the same reason. Expect it to slow N and T while it runs.

**Making the exact average fast on the GPU.** On CPU in `exact4096`, the exact average took about 1.05 s per iteration, more than everything else combined. Most of that is `DenseTabularPolicy.recompute_likelihoods`, a Python loop over all 262,144 histories. Each iteration needs:

1. the network's dense current policy: a forward pass over 2.6M (history, hand) rows on the GPU, a fraction of a second;
2. own reach for both players: replace the loop with a GPU computation that processes histories **one depth (claim count) at a time**, 18 vectorised steps;
3. the float64 accumulation, also on the GPU.

The implementation uses a batched GPU recurrence for own reach. In the smoke test, its reach arrays matched `DenseTabularPolicy.recompute_likelihoods` exactly (maximum absolute error 0). Resumed smoke runs reached iteration 300 for both arms and wrote diagnostics. The initial concurrent 50-iteration benchmark took about one measured minute per arm, roughly 1.2 seconds per iteration; 24,000 iterations is therefore estimated at about eight hours of measured work per arm before snapshot/evaluation overhead.

### Stopping rule and cadence

**For this experiment, stop at a fixed iteration count: 24,000 iterations per arm.** The rise starts at about 15k iterations, so this leaves room past it. With a shared GPU, measured time differs between the arms for reasons unrelated to the method, and everything is compared by iteration.

**Snapshots every 250 iterations**, the same for both arms, based on the concurrent smoke benchmark. At the observed rate this is about five minutes of measured training between snapshots. The runners record actual timing so this can be checked during the run. The rolling checkpoint is atomically replaced at each snapshot; policy snapshots are deleted after their exact evaluation completes.

At every snapshot, each arm:

- writes its **exact average** and its **current policy** (the network's regret-matched policy) for CPU evaluation of exact exploitability;
- computes the **accumulation diagnostics** below on the GPU, where everything they need is already in memory, and appends them to its metrics file.

A rolling checkpoint every 10 snapshots is enough; checkpoints are only for resuming.

### Measurements

**Exact exploitability** of the exact average and of the current policy, at every snapshot.

**Accumulation diagnostics**, at every snapshot, for each player:

| Quantity | Definition | What it shows |
| --- | --- | --- |
| Regret scale | Reach-weighted mean over sets of Σₐ R⁺_table(I, a) | Whether true regrets are flat or grow like √t |
| Accumulated error | Reach-weighted mean over sets of Σₐ \|R̂_net(I, a) − R⁺_table(I, a)\| | How far the network has drifted from the true sum |
| Relative error | Accumulated error ÷ regret scale | **The quantity the hypothesis is about** |
| Policy gap | Reach-weighted total variation between the network's and the table's regret-matched policies | The same error, in policy terms |

Reach means the current policy's reach times chance. Report each quantity by expected-visit bin (<0.1, 0.1–1, 1–10, ≥10 visits per iteration), as in the Part C audits. In T, the policy gap is how well the network reads a table it is trained on. In N, it is that reading error plus everything that has accumulated.

**Timing** per iteration: traversal, fit, shadow-table update, averaging and diagnostics. These tell us what the 30-claim version would cost.

## How to read the results

| Observation | Interpretation | Next step |
| --- | --- | --- |
| T ≈ `exact4096` and keeps falling; N trails and turns up; N's relative error grows over time and T's does not | **Bootstrapping is the cause.** The network can read regrets well but cannot accumulate them. | Design a target that does not build on the network's own output, such as fitting to a replay of increments or periodically re-anchoring. The [discounting rerun](2026-09-30_00-55-09_18_claim_tabular_discounting.md#interpretation) shows plain CFR fails on our increments, so a replay method needs reach-weighted increments or a DCFR-style rule. |
| T and N are similar, both trailing the table; relative error is flat | The cost is **the network as a reader** (capacity or per-update fit), not accumulation. | Use the [distillation experiment](2026-10-02_13-54-40_18_claim_regret_table_distillation.md) to separate capacity from fitting; try per-update fixes (one row per set, more steps, extra player-2 roots). |
| Both turn up late, and N's relative error grows | Error growth that teacher forcing does not remove: per-fit error itself grows as training goes on. | Decay the regret learning rate over the run, or rescale targets to a constant size. |
| Neither turns up | The earlier rise needed something these arms lack: the O4/online-average path, or seed noise. | Re-check with O4 on N's snapshots before acting. |
| T is between the table and N | Both contribute. | N ÷ T is the share that a non-bootstrapped target could recover. |

**Expected levels at matched iterations.** From the earlier runs, N should be about 1.6× `exact4096` from 4k to 12k iterations (neural with O4 at 1.85×, divided by O4's ~1.15×). That is about the level of a table with K≈1,500 ([yardstick](2026-10-01_10-25-57_18_claim_average_fit_traversal_schedule_regret_noise.md#a-yardstick-for-the-regret-network)). If N lands well below that, the exact average itself interacts with the neural trajectory, and that should be understood first.

## Implementation notes for Codex

**Reuse existing code; one trainer with an option.**
- `TabularRegretFork` already owns a compact 18-claim regret table, its key function and the clipped per-set mean update.
- `DeepCFRPlusTrainer` owns the regret networks and their fit.
- `ExactAverageTabularDiscountTrainer.accumulate_exact_average` is the observer. It currently compiles the table's policy; N and T must compile the **network's** policy instead (`DeepCFRPlusTrainer.current_policy_dense`, on the GPU).

A single option such as `regret_target_source={"bootstrap", "table"}` covers both arms.

**Where the target's old value comes from.** The traversal reads regrets once (`regret_values_tensor`). It uses them both for the policy and as the "old" value in each stored target, old + ĝ. Here the network sets the policy, so stored targets are network-old + ĝ. Each iteration, before fitting:

1. Recover ĝ per visit row as stored target − network output at that row. The network is frozen during traversal, so this is exact.
2. Update the shadow table: per-set mean of (table-old + ĝ) over the visits, then clip.
3. **N:** fit to the stored targets, as now. **T:** replace each row's target with the updated table value at its set, then fit with the same steps, batch and rows.

**GPU path.**
- Start from the Part C CUDA arms in [`run_cfr_plus_18_neural_o4_cpu.py`](../../../scripts/run_cfr_plus_18_neural_o4_cpu.py), with an option for the exact average and the shadow table.
- `TabularRegretFork` currently insists on CPU. The shadow table needs a CUDA variant of its key function and update, which are plain indexing.

**Implementation.** `scripts/run_cfr_plus_18_regret_bootstrap_teacher_forced.py` subclasses `DeepCFRPlusTrainer`. It stores the shadow table and visit counts on the GPU, keeps an exact own-reach linear average, writes a rolling checkpoint every snapshot, and emits training and binned network/table diagnostics. Current and average policies are saved temporarily for exact CPU evaluation, then removed so disk use stays bounded. The focused dashboard is served on VM port 8770; the existing overview on 8765 discovers both arms from their logs.

**Launch order.**
1. **Benchmark:** completed concurrently on the GPU; the initial 50 iterations established the snapshot cadence.
2. **Smoke test:** completed for both arms through iteration 300, including exact evaluation, diagnostics and resume. The reach parity check passed exactly.
3. **Full launch:** N and T were started in separate tmux sessions on 2 October 2026. Exact evaluators are spawned separately. The focused dashboard is at VM port 8770, and the VM overview on port 8765 discovers both runs. At initial verification, both arms were progressing, each rolling checkpoint was about 613 MiB, GPU use was about 13 GiB of 16 GiB, and the VM had about 13 GiB disk free. Monitor disk and GPU use as snapshots/evaluations begin.

**Checks before launch.**
- The shadow-table update must match `exact4096`. If the policy is read from the shadow table instead of the network, the trainer should reproduce `exact4096`'s first few iterations.
- The GPU reach computation must match `recompute_likelihoods` exactly on a few random policies.
- Checkpoints must include the shadow table and the exact average sum, so runs can resume. Keep one rolling checkpoint per arm; the table and average add about 600 MB each.

**Leave alone.** No online strategy network, reservoir or O4 refits in these arms. Evaluation runs on CPU, outside the training processes.
