# 30-claim checkpoint best-response screen with less sampling noise

**Status: complete for the first run (4 October 2026).** All 48 hourly O4 snapshots and the four June policies were scored; results are in the [first-run doc](2026-10-03_05-12-43_30_claim_first_run.md). A batched, GPU-assisted solver (below) now makes the screen about 12× faster, and a paired depth-3 audit is running.


## Why redo the screen?

The initial depth-two evaluation played 2,000 full games per seat. Around a discovered exploitability of 0.03–0.09, its two-seat 95% interval was roughly ±0.044. That cannot resolve changes between hourly snapshots, the two regret-network widths, or the June baselines. Routine depth-three and depth-four screens were also too costly on this spec. The old evaluation workers were stopped; their result files remain for reference.

## Method

The responder still chooses actions with depth-two search and exact beliefs over the opponent's hand. Only **scoring that fixed responder** changes:

1. Every sweep plays one game for each responder hand type. The estimates are combined with the exact probability of each hand. This removes chance variation from which responder hand was dealt.
2. Opponent actions still come from a sampled hidden hand. At the terminal call, the payoff is averaged over **all** opponent hands, weighted by the exact posterior after the observed claims and, when the opponent calls, the probability of that call. This removes the remaining hidden-hand payoff draw without changing the mean.
3. Each policy uses the same random inputs indexed by seat, shard, sweep, and responder hand. Consequently, snapshots and widths share opening deals and action uniforms. Differences are computed on the same sweeps, with their own paired confidence intervals.

Each independent CPU shard makes 58 sweeps. With 35 responder hand types at 30 claims, that is 2,030 games **per seat**. Thirty shards aim for 60,900 games per seat per policy. Up to 30 shards run at once on CPU; results are saved atomically and completed shards are reused after a restart. The queue follows newly saved hourly O4 policies from W512 and W2048 and also evaluates the four June policies. Output lives under `artifacts/cfr_plus_30_claim_first_run/main_20261003/precise_evaluations/` on the VM.

For each seat, the estimate is the mean over independent hand-weighted sweeps. Its standard error is the sweep standard deviation divided by the square root of the number of sweeps. The two seat variances are added for the exploitability interval. For a comparison, scores from the same shard and sweep are subtracted first; the reported interval is based on these paired differences. This is the uncertainty in the **value of the depth-two responder**, not an error bound on its gap to the true best response.

## Calibration before the queue

On a saved 18-claim policy with an exact depth-two value of **0.022612**, the new scorer gave **0.022643 ± 0.004307** (95% interval) from 50,000 games per seat. A separate two-sweep 30-claim checkpoint run completed. A ten-sweep timing pilot took 156 seconds for 350 games per seat, including cold search-cache work. These checks support the scoring formula and show that parallel CPU shards are practical; they do not validate the 30-claim responder's search depth.

## How to read the output

- `summary.json`: per-policy estimate, two-seat 95% half-width, nonnegative lower confidence bound, number of games, and shards completed out of 30.
- `paired.json`: W512 minus W2048 at each common snapshot, consecutive snapshots within an arm, and each new 60-minute policy against each June baseline. A positive difference means the left policy is more exploitable. If the interval crosses zero, the comparison remains unresolved.
- `shards/<policy>/<id>.npz`: sweep-level scores for both seats, retained so paired comparisons can be recomputed later.
- The 30-claim dashboard at VM port 8774 / local port 18774 shows completed estimates with intervals and a paired-width table. It retains the original low-sample results in a separate table.

The queue can lag training. A partially evaluated policy appears in the table with its shard count; the graph waits for all 30 shards. The queue uses CPU and leaves the GPU trainers and their checkpoint format alone. CPU load can still affect training speed, so iteration timings and host memory are monitored.

After a VM restart, resume the evaluation queue with `bash scripts/launch_cfr_plus_30_precise_eval.sh` from the VM checkout. It skips completed shards. The training controller has its own resume procedure.

## Limits

Lower Monte Carlo variance cannot repair a weak depth-two responder. A deeper search audit remains useful on a few selected policies, but its planning cost must be measured separately; adding more rollout workers alone does not make depth three cheap. June's five-minute fitted-return BR is an independent lower bound and may be much weaker than the search responder.

Implementation: [`evaluate_cfr_plus_30_precise.py`](../../../scripts/evaluate_cfr_plus_30_precise.py).

## Batched opponent queries (4 October)

**The bottleneck.** `LimitedBestResponse` asks the opponent network about one public history at a time: one call of 35 rows, one per opponent hand type. At depth 2, about 69% of the time went into hundreds of thousands of these tiny calls.

**The change.** [`BatchedLimitedBestResponse`](../../../liars_poker/algo/br_limited_batched.py) is a subclass; the original is unchanged.
1. Before each responder decision, it walks the planning tree level by level, applying the same ε-pruning, and collects every opponent history the search will reach.
2. It evaluates them in a few large batches, optionally on the GPU.
3. It vectorises the opponent step and the last planning level, using one matrix product per node instead of one dot product per legal reply.

The recursive search, pruning rule and scoring are unchanged.

**Validation.**
- **Local, June 30-claim policy:** depth 1 and depth 2 values are identical to the original. At depth 2, the opponent-network calls fell from 69,835 to 71.
- **VM, full shard:** one complete depth-2 shard (58 sweeps, W2048 at 1,440 minutes) matches the stored precise-screen shard to within 2×10⁻⁸ per sweep, with no decision changes.

**Speed** (VM, one CPU core per worker, opponent queries on the GPU):

| Workload | Original | Batched + GPU | Speed-up |
| --- | ---: | ---: | ---: |
| Depth 2, 10 sweeps (350 games per seat) | 111 s | 11.8 s | 9.4× |
| Depth 2, one full shard (2,030 games per seat) | 290 s | 24 s | 12× |
| Depth 3, first 2 sweeps from cold caches | not finished in 20 min for 1 sweep | 290 s | – |

**Memory, for planning parallel runs.**
- **Depth-3 workers are heavy:** about 13–15 GB of RAM each with a 1M-history cache, plus about 1.2 GB of GPU memory each.
- **Ten at once exhausted both RAM and GPU memory.** Some depth-2 shards failed with CUDA out-of-memory and were re-run. Five depth-3 workers with a 400k cache, alongside six depth-2 workers, fit.

Driver for paired re-scoring, deeper search and other policy kinds: [`evaluate_cfr_plus_30_depth_gap.py`](../../../scripts/evaluate_cfr_plus_30_depth_gap.py). Profiling: [`profile_cfr_plus_30_limited_br.py`](../../../scripts/profile_cfr_plus_30_limited_br.py).

## Online policies (4 October)

The online average policies of both arms at 60 and 1,440 minutes were scored with the batched solver. These are the same 30 shards and the same random inputs as the O4 and June scores, so all comparisons are paired. Results and the split of the gap to June are in the [first-run doc](2026-10-03_05-12-43_30_claim_first_run.md).

## Depth-3 audit (4 October)

**What it measures:** how much depth 2 misses at 30 claims. The final O4 snapshots of both arms (1,440 minutes) were re-scored at depth 3 (ε=10⁻⁴) on 10 of the 30 shards: 580 sweeps, or 20,300 games per seat. The random inputs are identical, so each depth-3 sweep pairs with its depth-2 sweep.

| Policy | Depth 2 | Depth 3 | Depth 3 − depth 2 (95% interval) | First-seat responder gain | Second-seat responder gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| W2048, 1,440 min | 0.0192 | 0.0206 | +0.0014 [−0.0051, +0.0079] | +0.0022 [−0.0036, +0.0079] | −0.0008 [−0.0036, +0.0021] |
| W512, 1,440 min | 0.0241 | 0.0244 | +0.0003 [−0.0040, +0.0046] | +0.0004 [−0.0031, +0.0039] | −0.0001 [−0.0025, +0.0022] |

- **Depth 3 finds no more than depth 2 within about ±0.005,** for either policy or either seat. Under the current leaf rule (call the opponent's next raise), the search has effectively converged in depth.
- **This does not show the numbers are close to true exploitability.** Depths 2 and 3 share the same leaf rule, so any blind spot it creates is shared too. Better leaf values are the natural next test.
- **Pairing helped less than hoped.** Depth 3 often chooses differently from depth 2, so the paired games decorrelate. The intervals on the difference are about ±0.004–0.006, similar to the per-policy intervals.
- **Cost:** about 18 minutes per depth-3 shard (58 sweeps) with GPU queries, against about 24 s per depth-2 shard.

Output: `artifacts/cfr_plus_30_br_depth_gap/main_20261004/` on the VM (`summary_d3.json`, shard files and logs).
