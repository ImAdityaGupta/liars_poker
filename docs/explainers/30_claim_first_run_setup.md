# How the first 30-claim run was prepared

This note records the checks behind the first long run on `r5_s4_h3_hp2ptq_ss` (30 claims). It explains why the run uses two regret-network widths, a capped traversal ramp, separate snapshot fitting, and sampled evaluation. It is a setup rationale, not a report on the eventual 24-hour results; see the [experiment record](../experiments/30_claim/2026-10-03_05-12-43_30_claim_first_run.md) for those.

## What we needed to establish

The recipe had worked on 18 claims, but several assumptions could fail at 30:

- the CUDA aggregate-then-clip path had only been guarded and exercised on the smaller spec;
- the larger network, traversal batch, and replay buffers might not fit together with the O4 average-policy fitter;
- a checkpoint might fit on disk but cause a second memory spike when restored;
- exact full-game evaluation and the older approximate responder might be too slow to use for frequent policy comparisons.

We checked these separately so that a bad training result would be less likely to hide a basic porting or resource failure.

## Training path and numerical checks

The trainer uses the established 18-claim recipe: cumulative conditional regrets, aggregation before clipping, plain masked MSE, a bootstrapped regret target, full expansion of the traverser's legal actions, and sampled opponent actions. The 30-claim encoder has 35 hand-type features and 31 actions. Exact dense policies and exact exploitability are not practical at this size.

Before the long run, the CUDA smoke compared aggregate-then-clip on the same synthetic rows on CPU and CUDA, checked that the regret buffer raises rather than silently overwriting a partial iteration, ran short CUDA traversal/training iterations, and saved and restored a checkpoint. The point was to test the algorithmic path and resume mechanics on the new spec, not to infer policy quality from a tiny run.

We also ran a 1,000-iteration 18-claim regression through the new runner, where exact evaluation is still feasible. The online average measured 0.02195 exact exploitability; O4 refitting it measured 0.01073. That is close to the earlier O4 result around 0.0105, so the new runner's average-policy path behaved as expected on a game with a known reference.

## Hardware measurements determined the run configuration

The GPU pilot allocated both regret networks together while the existing O4 worker was using the card. It varied K and timed iterations, counted regret records, and measured memory. The observed constraints led to these choices:

| Observation | Decision | Reason |
| --- | --- | --- |
| Traversal batch 1,024 ran out of memory with the 2,048-wide model; 256 worked. | Set traversal batch to 256 for both arms. | The goal was to run both widths beside O4 without relying on a configuration that had already OOMed. |
| At K=32,768 the wide model took about 25–27 seconds per iteration, above the planned 10-second ceiling. | Ramp K from 1,024 to 16,384, not 32,768. | Very long iterations reduce snapshot resolution and make a late-run slowdown expensive. |
| The largest measured per-player regret batch at K=16,384 was about 0.91 million rows. | Use a 2-million-row regret buffer. | This is roughly 2.2 times the observed batch, exceeding the planned 1.5-times margin and leaving room for variation. |
| Both arms fit with a 4-million-record strategy reservoir per player; 8 million left too little GPU headroom. | Use 4 million for each arm. | The two experiments need comparable averaging capacity, and must share the GPU with periodic O4 fits. |
| W512 matches the settled 18-claim width; W2048 is the earlier 30-claim width. | Run those two widths with other settings held fixed. | This makes network capacity the main planned comparison. |

The measured-time K ramp rises monotonically from 1,024 to 16,384 during each arm's 24-hour budget. The run is meant to compare policies both by elapsed training time and cumulative traversal roots: the wider network has a different iteration cost, so iteration number alone is not a fair comparison.

## Resume and snapshot checks changed the O4 handoff

The first long-smoke attempt exposed two memory peaks that only occur around snapshot work:

1. A persistent controller retained cached CUDA allocations after O4 fitting.
2. Checkpoint restore allocated a second device replay reservoir before replacing the reservoir already owned by the trainer.

The runner was changed so O4 fitting runs in a short-lived subprocess, which releases its CUDA context when done. Restore now copies checkpoint rows into the existing reservoir in chunks. A 20-minute smoke then completed four five-minute snapshot, O4-fit, and resume cycles for both arms. The temporary full checkpoints were removed afterward to recover disk space; the smoke policies and logs were retained.

These checks justify the mechanics and observed memory envelope. They do not prove that a 24-hour run cannot fail for unrelated reasons, which is why each arm keeps a rolling checkpoint and detailed logs.

## Evaluation needed a different tradeoff

Exact exploitability at 30 claims is out of reach. A first attempt to score a depth-two responder by enumerating all full-game outcomes spent over 20 CPU minutes on one policy without finishing. We therefore kept exact opponent beliefs for choosing actions, but estimated the responder's final value from sampled complete games.

On an 18-claim policy, the sampled estimate from 100,000 games was 0.01886, versus 0.02261 from exact enumeration. Their difference, 0.00375, was inside the Monte Carlo 95% half-width of 0.00618. This check supports using sampled rollouts as a *screen*, not as a precise final answer. The 30-claim screen uses 2,000 games per seat at depth two, 500 at depth three, and 100 at depth four, with uncertainty bounds. Low sample counts make the deeper screens noisy; promising policies need longer independent evaluation. A five-minute fitted-return best response remains a separate check at the end.

The screen is run by CPU workers outside the trainer's hot loop. Its results can lag behind training, but evaluation does not consume the GPU time needed for regret fitting. The initial 2,000-game screen proved too noisy for policy comparisons. The [precise checkpoint follow-up](../experiments/30_claim/2026-10-03_17-30-00_30_claim_precise_checkpoint_br.md) now uses hand stratification, terminal belief averaging, and about 60,900 games per seat per policy.

## Dashboard and operational boundary

The dashboard is a read-only process on the VM at port 8774. It reads training, snapshot, checkpoint, and evaluation files; it does not run evaluations or alter trainer state. The Windows SSH tunnel exposes it at `http://127.0.0.1:18774`. If that local URL stops responding while the remote monitor session remains alive, rerun `scripts/tunnel_vm_dashboards.ps1`; the tunnel is separate from both the dashboard server and the training processes.

## What these choices do not establish

- The smoke checks do not show that the 30-claim policy converges or that either network width is better.
- The K ceiling is a throughput decision from this GPU and this concurrent workload, not a general optimum.
- The sampled expectimax values are approximate and have different uncertainty at each depth. Compare policies at matched depth and sample count, then spend more games on interesting candidates.
- The run compares widths at one seed. A width difference would be suggestive, not a seed-robust result.

## Source files

- [30-claim run plan and results](../experiments/30_claim/2026-10-03_05-12-43_30_claim_first_run.md)
- [Earlier port and benchmark plan](../experiments/30_claim/2026-10-03_03-16-57_30_claim_port_and_benchmark.md)
- `scripts/smoke_cfr_plus_30_cuda_aggregate.py`
- `scripts/benchmark_cfr_plus_30_first_run.py`
- `scripts/smoke_cfr_plus_18_regression_for_30.py`
- `scripts/run_cfr_plus_30_claim_first_run.py`
- `scripts/evaluate_cfr_plus_30_claim_first_run.py`
