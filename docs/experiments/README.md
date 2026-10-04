# Experiment index (September–October 2026)

One place to see every experiment in this folder: what it tested, what it found, and whether it still matters. Each entry links to its full report. Background notes and proposals are indexed separately in the [explainers index](../explainers/README.md). **Numbers are current as of 2 October 2026.** Where a report lags its run, the number here came from the VM and is marked *(VM)*.

Reports live in two folders: [`18_claim/`](18_claim/), which includes the five earliest six-claim toy-game reports, and [`30_claim/`](30_claim/).

## How to read the numbers

- **Exploitability** is exact (exact best response) unless marked otherwise. It is the first-seat best-response win probability plus the second-seat one, minus one. Lower is better; 0 is an equilibrium.
- **Almost every run is one seed.** Single snapshots scatter by about ±15%. Trust consistent trends across several snapshots, not single points.
- **Important caveat for older neural results.** Until 1 October, every *neural-average* number was computed by the **online average network**. That network turned out to be about **5× more exploitable** than the exact average of the same training trajectory (see Phase 5). So those older numbers mostly measure the averager, not regret learning, and **comparisons between regret recipes made through it are weak**. They are marked "online average" below.
- "Table" or "tabular" means regrets stored exactly per information set (no regret network). "Exact average" means the true own-reach-weighted linear average, accumulated exactly.
- The game is the **18-claim** spec (`r4_s4_h2`, RankHigh/Pair/TwoPair/Trips, suit symmetry) unless stated. The first five experiments use a six-claim toy game.
- **Reference point:** exact full-tree CFR+ reaches 0.00197 at 10,350 iterations (January 2026 benchmark). The [exact reference comparison](../explainers/claim_exact_reference_comparison.md) puts it on the same axes as our best sampled and neural runs.

## What we currently believe

| # | Finding | Confidence | Key evidence |
| --- | --- | --- | --- |
| 1 | **Cumulative regret targets beat normalized `/t` targets** for neural regrets. | High | Cumulative regret scale; table forks |
| 2 | **Average sampled targets before clipping** (aggregate-then-clip) rather than clipping each sample. The 1.5× positive-entry loss weight makes no difference. | High at 18 claims | Clip-order runs; target factorial; clip-on-read and plain-MSE study |
| 3 | **Don't put a reach factor (N/K) in neural targets.** In tables, visit-frequency weighting helps modestly. Sampled conditional updates with 4,096 roots and an exact average **beat exact full-tree CFR+ per iteration early** (about 2× at 1k iterations, 0.88× by 10k) but have a shallower late slope. | High | Tabular bridge; cumulative regret scale; 2×2 controls; exact reference comparison |
| 4 | **The online average network was the largest error** (about 5×). An annealed, large-batch refit at snapshot time (**O4**) removes about 97–99% of that excess. A **fresh** 40k-step refit is as good only in short runs: at 31.5k iterations it was 2.30× exact, while O4 stayed at 1.24×, because the warm start carries information the 2M reservoir has lost. The rest (about 1.1–1.2×) is in the **reservoir data** (sampling noise or coverage), not the network: distilling the exact average reaches 1.01–1.04×. | High | Offline average fitting; optimizer & objective; Part A |
| 5 | **With a table, total roots set the result, not how they are split into iterations.** Constant K from 256 to 16,384 agree within about 15% at matched roots, falling as roughly roots^−0.55. Higher K wins at equal time because of the fixed per-iteration cost of averaging. **The average reflects mostly recent iterates:** lowering K late hurts within a few hundred iterations, and a 512→7,680 ramp beats every constant K per root by 1.2–1.4×. | High for tables; one seed | Part B root schedules |
| 6 | **The regret network costs about 2×** versus a table at matched iterations (about 1.6–1.75× after allowing for O4), roughly a table with K≈1,500 instead of 4,096. **Neural runs also get worse late** (K=4,096 after about 15k iterations, K=1,024 after about 23k). This is not the averaging: O4 is still 1.24× exact at 31.5k iterations. Hypothesis: bootstrapped fit errors accumulate (tested by N/T). | Medium; rise is high | Neural O4 CPU and continuation; long-run O4 check |
| 7 | **Regret-fit optimizer noise is not the bottleneck.** Larger batch or per-update annealing gave differences within noise to about 9k iterations; a lower learning rate was worse. The audits show the network's per-update error is mostly at rarely visited and unvisited information sets. | Medium | Part C |
| 8 | **Discounting and quadratic averaging don't matter here; keep CFR+ with linear averaging.** All clipped or discounted variants are within about 9% of linear CFR+ at matched iterations. Vanilla CFR diverges on our conditional-mean increments. | Medium | Tabular discounting, exact-average rerun |
| 9 | **Clip on read is worse than aggregate-then-clip at 18 claims**, where most rows are repeat visits; so is the hybrid that aggregates but leaves the mean signed. It may matter at 30+ claims, where single visits dominate. | Medium | Clip-on-read and plain-MSE study |

## Phase 1: diagnosing the update on a six-claim toy game (27–28 September)

**[Sampled regret targets](18_claim/2026-09-27_23-28-05_cfr_plus_sampled_targets_cpu.md)**
- *What:* checked whether the traverser estimates action values correctly, and whether clipping each noisy sample biases regret targets.
- *Found:* pre-clip root values were unbiased. Clipping each sample inflated targets (mean 0.347 against 0.144 exact). In a tabular model, aggregating before clipping halved exploitability (0.016 against 0.032 at 200 iterations).

**[Exact shadow ledger](18_claim/2026-09-27_23-58-30_cfr_plus_shadow_neural_cpu.md)**
- *What:* ran an independent exact ledger alongside a neural run, to separate average-network error from regret error.
- *Found:* the learned average was about 11% worse than the exact average of the policies actually played, by 300 iterations. Root regret targets were biased upwards.

**[Neural clip order](18_claim/2026-09-28_01-04-50_cfr_plus_neural_clip_order_cpu.md)** and **[its longer run](18_claim/2026-09-28_01-17-33_cfr_plus_neural_clip_order_long_cpu.md)**
- *What:* put aggregate-then-clip into the real neural trainer.
- *Found:* about 44–48% less exploitable, persisting to 800 iterations (0.018 against 0.034).

**[Depth target audit](18_claim/2026-09-28_01-47-35_cfr_plus_neural_depth_target_audit_cpu.md)**
- *What:* compared neural targets with exact targets at every depth, not just the root.
- *Found:* aggregation reduced target error at most depths, though not uniformly. Deep information sets had too little data to judge.

## Phase 2: aggregation and traversal count at 18 claims (28 September; online average)

**[18-claim target order](18_claim/2026-09-28_02-34-03_cfr_plus_18_claim_target_order_cpu.md)**
- *What:* aggregate-then-clip against clip-each on the 18-claim game, 75 minutes each, then aggregate alone extended to 105 minutes.
- *Found:* aggregate was better at every snapshot (0.035 against 0.042 at 75 minutes), then flattened near 0.035.

**[Target × traversal factorial](18_claim/2026-09-28_13-29-00_18_claim_target_sampling_factorial.md)**
- *What:* {clip-each, aggregate} × {1,024, 4,096 roots} × 2 seeds, 330 minutes each, normalized units.
- *Found:* aggregate won 86 of 88 matched comparisons, about 20% lower at the end. No stable time-matched winner between root counts. Late progress stalled around 0.02–0.03 (online average).

## Phase 3: reach, units and where the neural loop loses ground (29 September)

**[Tabular bridge](18_claim/2026-09-29_01-42-50_18_claim_tabular_bridge.md)**
- *What:* six tabular arms with an exact average, varying how reach (`q`) and the conditional advantage (`g`) are estimated. No networks.
- *Found:* at 300 minutes, exact 0.0114, sampled reach 0.0117, **ignoring reach 0.0367 (3.2× worse)**, sampled value 0.0135, sampled both 0.0075, conditional only 0.0108. Sampling is not what creates the neural plateau; reach matters.

**[Low-root tabular bridge](18_claim/2026-09-29_12-57-52_18_claim_low_root_tabular_bridge.md)**
- *What:* the bridge's sample-both and conditional arms at 128, 256 and 512 roots.
- *Found:* fewer roots were clearly worse and barely faster. Iteration time is dominated by full-table sweeps, not sampling.

**[Cumulative regret and sampled reach](18_claim/2026-09-29_11-14-31_18_claim_cumulative_regret_scale.md)**
- *What:* removed `/t` (cumulative units), with and without an N/K reach multiplier.
- *Found:* **N/K failed badly** (0.46–0.56). **Cumulative conditional beat normalized** (0.0086 at 240 minutes against about 0.027). It was extended to 1,200 minutes, about 0.0065 at the end (online average).

**[Visit-count multiplier (N)](18_claim/2026-09-29_19-54-06_18_claim_cumulative_visit_count.md)**
- *What:* cumulative targets multiplied by visit count N instead of N/K.
- *Found:* learned, but worse than conditional (0.0125 at 600 minutes). Reach-style multipliers don't help neural targets.

**[Late-checkpoint update audit](18_claim/2026-09-29_13-47-16_18_claim_late_checkpoint_update_audit.md)**
- *What:* one regret update at two 330-minute checkpoints, comparing the old, exact-target, sampled-target and fitted-network policies.
- *Found:* fitting error exceeded sampling error, but inconsistently between seeds. Not conclusive alone.

**[O/E/S/N audit and tabular regret fork](18_claim/2026-09-29_14-30-00_18_claim_oens_longitudinal_and_regret_table_fork.md)**
- *What:* a seed-31 normalized run with repeated one-step audits; an exact-`g` rescue; then a fork replacing the regret network with a table at 300 minutes.
- *Found:* the normalized run plateaued at 0.02–0.026, and audit distances didn't track the plateau. Exact-`g` was too slow to judge. **The table fork improved 0.0222 → best 0.0024** (final 0.0044 at 70.8k iterations), implicating the neural regret loop.

## Phase 4: fitting, clipping and discounting (30 September)

**[Regret fit-steps sweep](18_claim/2026-09-30_00-55-10_18_claim_regret_fit_steps_sweep.md)**
- *What:* GPU forks from a 645-minute cumulative checkpoint with 24, 96 and 384 regret fit steps, plus a tabular fork from the same checkpoint.
- *Found:* no gain from more steps up to 23.7k–36.5k iterations (online average). The **same-source table reached 0.0035 at 72k**, but its gains came after about 40k iterations, so **the sweep was too short to conclude**. The GPU was about 8× faster than CPU per iteration.

**[Clip on read, hybrid and aggregate-then-clip with plain MSE](18_claim/2026-09-30_10-37-00_18_claim_clip_on_read_and_aggregation_mse.md)**
- *What:* three regret-target constructions with plain MSE against the 1.5×-weighted aggregate-then-clip reference, 4,096 roots: clip on read (per-visit signed targets, 330 minutes), the hybrid (group mean left signed, clip on read, 600 minutes) and aggregate-then-clip (600 minutes).
- *Found:* **plain-MSE aggregate-then-clip matches the weighted run** within about 10% per iteration, so the positive weight is unnecessary. The hybrid is 1.3–1.7× worse throughout; clip on read is about 1.2× worse and stalls after about 6k iterations. All averaged with the online average network.

**[Tabular discounting](18_claim/2026-09-30_00-55-09_18_claim_tabular_discounting.md)**
- *What:* tabular regrets with a neural average, 4,096 roots: CFR, CFR+ (linear and quadratic average), DCFR+, DCFR exact and visited-only.
- *First screen:* at about 45-50k iterations, the online averages were E 0.0039, A 0.0043, B 0.0050, C 0.0051, D 0.0056 (VM). Vanilla CFR reached about 0.22. These values did not distinguish the rules because the online neural average was too noisy.
- *Exact-average rerun:* all five new arms completed 540 minutes. E (visited-only DCFR) finished at 0.001303, D at 0.001374, C at 0.001433, B at 0.001471; V worsened to 0.1775. Existing control A finished at 0.001264 and was better by equal time, while D and E were about 4-9% better than A's interpolated curve at matched iterations. These are one-seed leads, not a settled DCFR ranking.

**[Offline average fitting](18_claim/2026-09-30_21-47-00_18_claim_offline_average_fitting.md)**
- *What:* on `exact4096` checkpoints, compared the exact average, the online neural average and refits of the online network.
- *Found:* **the exact average was 4.6–5.7× less exploitable** than the online average from the same trajectory. More steps of the online recipe did not help.

**[Batched bridge controls](18_claim/2026-09-30_00-55-09_18_claim_tabular_discounting.md)** (reported in the discounting note)
- *What:* the missing cells of an {exact, neural average} × {1,024, 4,096 roots} grid, all with tabular regrets.
- *Found:* **exact average, 4,096 roots: 0.0013 at 17.9k iterations**, better than exact CFR+ per iteration early but with a shallower late slope (see the [exact reference comparison](../explainers/claim_exact_reference_comparison.md)). This run is arm A of the discounting rerun. Exact average at 1,024 roots was about 3× worse. Under the neural average, the root counts were indistinguishable (`neural1024`: 0.0047 at 163k).

## Phase 5: fixing the average and testing the regret network (1 October)

**[Average fit: optimizer and objective](18_claim/2026-10-01_00-58-40_18_claim_average_fit_optimizer_and_objective.md)**
- *What:* refit recipes on three frozen checkpoints: learning-rate schedule, batch size, cross-entropy vs MSE, warm vs fresh.
- *Found:* **O4 (cosine 1e-3 → 1e-5, batch 16,384, 5k steps, warm)** reached 0.0058 / 0.0046 / 0.0032 against exact 0.0052 / 0.0044 / 0.0028, removing about 97–99% of the excess. MSE was not better. Fresh 20k was close.

**[Neural CFR+ with O4 refits (CPU)](18_claim/2026-10-01_07-56-40_18_claim_neural_o4_refit_cpu.md)**
- *What:* full neural runs (cumulative, aggregate, plain MSE) at 1,024 and 4,096 roots, with an O4 refit every 15 minutes.
- *Found (600 minutes):* **4,096 roots: refit 0.0032 at 11.9k iterations** (online 0.0098). **1,024 roots: refit 0.0046 at 22.6k, rising to 0.0060 by 31k.** About 2× tabular plus exact at matched iterations. The report frames the four runs as a regret-storage × roots 2×2.
- *Continuation to 1,140 minutes:* the 4,096-root refit bottomed at **0.0029 (15.2k iterations)** and rose to **0.0047 by 20.9k**, while the table kept improving. An O4 refit of `exact4096`'s final checkpoint (31.5k iterations) is still 1.24× exact, so the rise is in the regret network's iterates, not the averaging. A fresh 40k fit on the same checkpoint was 2.30×.

**[Part A / B / C: average-fit schedules, root schedules, regret-fit noise](18_claim/2026-10-01_10-25-57_18_claim_average_fit_traversal_schedule_regret_noise.md)**
- **A (finished, written up):** ten refit recipes on the three `exact4096` checkpoints.
  - **Distilling the exact average table reaches 1.01–1.04× exact;** every reservoir fit ends at about 1.1–1.2×. The residual gap is the reservoir's data, not the fit.
  - O4, warm with fresh Adam, fresh 40k and fresh 80k are tied within seed noise. Shorter warm anneals are slightly worse at the earliest checkpoint only. Keep O4.
  - The stale-Adam explanation was rejected: a reset optimizer still degraded at a constant learning rate of 1e-3.
- **B (complete, 2 October):** ten network-free tabular runs, each for 540 measured minutes, with exact averages.
  - At matched total roots, all constant K agree within about 15%. Higher K wins at equal time because averaging has a fixed per-iteration cost.
  - Dropping K from 7,680 to 512 took the average from 0.00168 to 0.00301 in 505 iterations, and it then rejoined the constant-512 curve. The 512→7,680 ramp was best per root.
  - **Continuation to 1,140 minutes:** no floor. Every constant arm still falls at about −0.5 (K=256 reached 0.0036). Constant arms still collapse by total roots. The ramp, continued from 7,680 to 32,768, is best by every measure: **0.00071**, 1.15–1.3× better per root than any constant K.
  - **Follow-up with O4 averaging (complete, 1,200 min):**
    - The ramp wins clearly (1.4–2.2× per root over any constant K).
    - **An 8M reservoir beats 2M in 39 of 41 late snapshots (median 0.79×).** The 8M ramp reached **0.000345**, the best 18-claim result so far.
    - With O4, constant K no longer orders by time, and very small or very large K are worse per root.
    - O4 is about 1.2× exact with 2M, and roughly matches exact with 8M.
- **C (crashed at 150 minutes, about 9k iterations):** the four arms crashed when the disk filled, and have not been resumed.
  - Regret-fit batch 8,192 and per-update annealing were within noise of the baseline (refit 0.0039–0.0048 against 0.0043–0.0044 at 5–7k iterations).
  - A lower learning rate (3e-4) was worse throughout.
  - All eight audits finished. At heavily visited sets the network adds about 30% to each update's error on top of sampling noise. At unvisited sets it moves the policy by a third to a half of a true step; for player 2 these moves are unrelated to the correct update.

**[Long-run averaging check](18_claim/2026-10-02_12-29-46_18_claim_average_fit_long_run_check.md)**
- *What:* O4 (three seeds) and a fresh 40k fit on `exact4096`'s final checkpoint (31.5k iterations).
- *Found:* **O4 is 1.17–1.32× exact (mean 1.24)**, the same as at 1–4k iterations, so the neural late rise is not the averaging. **A fresh fit is 2.30×** (1.09–1.22× in Part A): in long runs the 2M reservoir alone is not enough, and O4 relies on its warm start from the online network. Chaining each refit from the previous one instead was no better: 1.55× at 31.5k iterations.

## Phase 6: why the regret network trails the table (planned, 2 October)

**[Bootstrapped versus teacher-forced regret networks (N/T)](18_claim/2026-10-02_13-54-39_18_claim_regret_bootstrap_vs_teacher_forced.md)**
- *What:* two neural runs with exact averages, identical except for the fit target: the network's own output plus the increment (N, as now), or the true cumulative regret from a shadow table (T). Both keep the shadow table to measure accumulated error over time. Both run concurrently on the GPU to 24,000 iterations, with exact evaluations every fixed number of iterations (at least 5 minutes apart).
- *Decides:* whether bootstrapping (errors that are never corrected) causes the 1.6× gap and the late rise.

**[Regret-table distillation](18_claim/2026-10-02_13-54-40_18_claim_regret_table_distillation.md)**
- *What:* fit fresh regret networks offline to `exact4096`'s regret tables at 908, 1,424, 3,988 and 31,538 iterations. Variants: raw regrets or policy targets; visit-weighted or mixed sampling; 512×512 or 1,024×3.
- *Decides:* whether the network can hold the table at all, including late in training, and how large it must be at 30 claims.
- *Found:* capacity is fine. With policy targets (P-visit), every set visited at least 0.1 times per iteration is matched exactly, even late. Raw-regret MSE was a large part of the failure (late: 31× and 16.5× against 4.65×). Rarely visited sets stay poorly fitted under every objective. Concluded; normalised regret targets are the first fix to try if the regret network limits 30 claims.

## Phase 7: towards 30 claims (planned, 3 October)

**[Approximate best-response calibration](18_claim/2026-10-03_03-16-56_18_claim_approximate_best_response_calibration.md)**
- *What:* LBR, depth-limited expectimax with exact beliefs and exact enumeration, MCTS, and the existing fitted-return responder. Each is run on 12 saved 18-claim policies with known exact exploitability (0.001–0.03).
- *Decides:* which evaluator to use at 30 claims, by how much of the exact value it finds, how well it ranks policies, and what it costs. Mostly CPU.
- *Found (first pass):* **depth-limited expectimax with exact beliefs is essentially exact at 18 claims.** d=3, ε=10⁻⁴ recovers 0.99–1.00 on all 12 policies, ranks every pair correctly including ties, and costs about 11 CPU-seconds per policy. d=2, ε=10⁻⁴ recovers 0.92–1.00. LBR finds 64–89% and nothing near equilibrium. Next: lazy opponent queries (costs here relied on dense tables), then a trial at 30 claims.

**[30-claim port and benchmark](30_claim/2026-10-03_03-16-57_30_claim_port_and_benchmark.md)**
- *What:* port the settled recipe to a 30-claim spec on the GPU. Measure per-iteration cost, records per iteration and memory across K and network width, and run a 60-minute smoke run.
- *Decides:* the configuration of the first real 30-claim run. Superseded by the plan below, which includes its benchmarks.

**[30-claim first run](30_claim/2026-10-03_05-12-43_30_claim_first_run.md)** (running)
- *What:* the settled recipe on `r5_s4_h3_hp2ptq_ss` for 24 measured hours, on the GPU. Two concurrent arms differ only in regret network width (512×512 against 2048×2048). K ramps from 1,024 up to 16k or 32k, with hourly O4 snapshots. It starts with a benchmark, an 18-claim regression run with exact evaluation, and a smoke run.
- *Evaluated by:* depth-limited expectimax (lazy queries) on every snapshot, deeper on some, plus June's fitted-return responder on the final snapshots and the same expectimax screen on June's policies.
- *Results (complete, 1,440 min; precise depth-2 screen ±0.005):*
  - **Both arms clearly beat June** (0.034 and 0.040 against 0.057–0.069 at 60 min). Part of that may be O4 averaging.
  - **W512 stalls** at about 0.025 after about 12 hours.
  - **W2048 (regret width) keeps improving** to about 0.019 and is significantly better from 1,140 min, with 65% of the roots. Its late gain is all in the second-seat strategy.
  - **The final fitted-return BR found nothing** (estimates below zero), so it is no longer a usable gauge.

**[Precise 30-claim checkpoint BR screen](30_claim/2026-10-03_17-30-00_30_claim_precise_checkpoint_br.md)** (running)
- *Why:* the initial 2,000-game depth-two estimates had roughly ±0.044 uncertainty and could not distinguish policies.
- *What:* independent CPU shards re-evaluate saved policies with exact responder-hand weighting, terminal opponent-hand averaging, common random inputs, and paired confidence intervals. Routine depth-three/four screens have stopped.

## Open questions and next steps

**Where we stand for 30 claims.** The recipe is mostly settled:
- cumulative units and aggregate-then-clip;
- CFR+ with linear averaging;
- GPU training;
- O4 refits warm-started from the online averager;
- a rising root schedule.

The open items are listed below.

1. **Evaluation at 30 claims.** The 18-claim calibration chose depth-limited expectimax with exact beliefs (Phase 7). Next steps, from the [calibration doc](18_claim/2026-10-03_03-16-56_18_claim_approximate_best_response_calibration.md#next-steps):
   - **Lazy opponent queries:** batched network queries with a cache, instead of compiling the opponent to a dense table. Re-run on the same 12 policies: values must match, and the run measures queries and time per policy.
   - **30-claim trial:** d=2, 3 (and 4 if affordable) on policies from the June 30-claim screen. Check that values converge with depth and compare with the June fitted-return estimates.
2. **First 30-claim run** ([plan](30_claim/2026-10-03_05-12-43_30_claim_first_run.md)): W512 against W2048, 24 measured hours each, starting with benchmarks and an 18-claim regression check.
3. **Root schedule with O4 averaging** (Part B follow-up, **running since 3 October**): confirms the ramp, and the reservoir size, under the averaging available at 30 claims.
4. **The regret network's cost** (about 2–2.6× the table, with no late collapse in the clean N run). Accepted for now; no further 18-claim diagnosis planned.
