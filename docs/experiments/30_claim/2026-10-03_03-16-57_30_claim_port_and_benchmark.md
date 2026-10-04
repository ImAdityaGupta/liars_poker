# 30 claims: port the recipe and benchmark it

**Status: superseded (3 October 2026)** by the [first 30-claim run](2026-10-03_05-12-43_30_claim_first_run.md), which includes these benchmarks as its step 1. This note is kept for the code audit and the background.

## Summary

- **Goal:** get the 18-claim recipe running on the GPU at 30 claims, and measure what it costs. The first real 30-claim run can then be sized from data rather than guesses.
- **Deliverables:**
  1. a working 30-claim training path (resumable, with snapshots and O4 refits);
  2. a cost table: per-iteration time, memory and records per iteration, across K and network width;
  3. a proposed configuration for the first real run.
- **Out of scope:** comparing recipes and judging policy quality. That waits for the calibrated evaluator.

## The recipe to port

Everything here was settled at 18 claims; see the [experiment index](../README.md).

| Component | Setting |
| --- | --- |
| Algorithm | CFR+ with linear averaging |
| Regret targets | Cumulative conditional regret, aggregate then clip, plain masked MSE |
| Regret network | Bootstrapped (target = clipped own output + new increment). 24 Adam steps per update, batch 1,024, learning rate `1e-3`. Width to be measured (below). |
| Average | Online average network (six steps per iteration) with a reservoir; an **O4 refit** (warm, 5,000 steps, batch 16,384, cosine `1e-3` → `1e-5`) at every snapshot |
| Sampling | Full traverser-action expansion; K roots per player, **ramped up over training and never decreased** |
| Hardware | GPU traversal and fitting (the Part C CUDA path); O4 on the GPU; evaluation on CPU |

## Decisions to confirm before starting

1. **The 30-claim spec: `ranks=5, suits=4, hand_size=3`, with RankHigh, Pair, TwoPair, Trips, Quads, and suit symmetry** (`r5_s4_h3_hp2ptq_ss`: 5 + 5 + 10 + 5 + 5 = 30 claims, 35 hand types). This is the spec of the earlier 30-claim learning-rate screen (June 2026, local `artifacts/cfr_plus_30_claim_lr_schedule_screen/`). That run also gives a rough baseline: fitted-return best-response estimates of about 0.08, 0.05 and 0.03 after 10, 20 and 30 training minutes (5-minute responders, 200,000 games per seat).
2. **Regret network width:** 512×512 as at 18 claims, or 1,024×1,024. Decide from the benchmark's fit cost.
3. **Reservoir size:** the largest that fits next to everything else. Part B with O4 averaging will show whether 8M beats 2M.

## Code audit: what is specific to 18 claims

| Item | Where | What to do |
| --- | --- | --- |
| CUDA aggregate-then-clip is only allowed for the 18-claim spec | `DeepCFRPlusTrainer`, the `cuda_aggregate_supported` guard ([`deep_cfr_plus.py`](../../../liars_poker/algo/deep_cfr_plus.py)) | Find out why the guard exists, validate the CUDA path on the 30-claim spec, then relax the guard |
| Runner constants `SPEC`, `H=1<<18`, hand and action counts | Part C runner via [`run_cfr_plus_18_target_order_cpu_overnight.py`](../../../scripts/run_cfr_plus_18_target_order_cpu_overnight.py); [N/T runner](../../../scripts/run_cfr_plus_18_regret_bootstrap_teacher_forced.py) | New 30-claim runner, or a spec option in the Part C runner |
| O4 harness takes its reference from an exact average | [`run_cfr_plus_18_average_fit_optimizer_experiment.py`](../../../scripts/run_cfr_plus_18_average_fit_optimizer_experiment.py) (`fit_one_arm`) | Use the Part C runner's GPU O4 worker, which needs no exact reference |
| Exact dense evaluation and exact average | `DenseTabularPolicy`, `ExactAverageTabularDiscountTrainer` | Not possible at 2³⁰ histories; remove from the 30-claim path |
| The table, `TabularRegretFork` | 18-claim only | Not needed |

The encoder (`InfosetEncoder`: input = ranks + claims, actions = claims + 1) and regret targets are already generic.

## Benchmarks

Run on the GPU with nothing else on the card, so timings are clean. Schedule this before Part B with O4 averaging starts, or between its O4 refits. Allow about 1–2 hours.

**1. Per-iteration cost.** At K = 1,024, 4,096, 16,384 and 32,768, record seconds per iteration, split into:
- traversal;
- regret fit (24 steps);
- online average steps (6);
- reservoir insertion.

Do this for regret width 512 and 1,024.

**2. Data per iteration.** At the same K:
- regret records and strategy records per iteration;
- visited information sets per player;
- rows per visited information set (it was 1.6 at K=1,024 and 2.4 at K=4,096 in an earlier estimate).

The regret buffer must hold a whole iteration for aggregate-then-clip, so this sets its size. It was 4M at 18 claims.

**3. Memory:** peak GPU memory at each K, and the largest reservoir that fits alongside (try 2M, 8M and 20M records).

**4. O4 refit time:** for a 2M and an 8M reservoir.

**5. Smoke run.** 60 measured minutes:
- ramp K from 1,024 upwards;
- snapshot every 15 minutes, with an O4 refit at each;
- stop and resume once;
- evaluate each snapshot with the existing fitted-return responder (5 GPU minutes, 50,000 games) as a placeholder until the calibrated evaluator exists.

This checks the pipeline end to end, not policy quality.

## Output: the first-run configuration

From the benchmarks, propose:

| Setting | How to choose it |
| --- | --- |
| K ramp start and end | Start where one iteration costs about the regret fit. End at the largest K whose iteration takes no more than about 10–20 s, so a snapshot interval holds enough iterations. Ramp linearly in measured time, as in Part B. |
| Time budget | 24 measured hours for the first run |
| Regret network width | 1,024 if its fit costs less than about 30% of an iteration at the ramp's end K; otherwise 512 |
| Reservoir | The largest that fits, up to 20M |
| Snapshot cadence | Every 30–60 measured minutes. Each snapshot gets an O4 refit and an approximate BR, so the BR's cost sets this. |

## Not needed yet

- Sampling traverser actions. Full expansion is affordable at 30 claims; sampling is for 69.
- Seeds, discounting, alternative targets.
