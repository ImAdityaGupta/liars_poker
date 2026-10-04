# Explainers

Background notes, specifications and proposals for the neural CFR+ work. For experiment reports and current conclusions, see the [experiment index](../experiments/README.md). **Index updated 2 October 2026.**

Each entry gives the note's purpose and how current it is. Notes written before a result came in are marked where a later experiment changed the picture.

## Start here

| Note | What it covers | Currency |
| --- | --- | --- |
| [From exact CFR+ to neural CFR+](neural_cfr_plus_from_exact.md) | What the neural trainer replaces in tabular CFR+ (sampled targets, regret network, average network) and how each replacement can stop helping. | Background; still accurate. |
| [Neural CFR+ code map](neural_cfr_plus_code_map.md) | Reading guide to one `run_iteration()` call through the trainer, traversal and target code. | Describes the code after commit `a66eefc`. |
| [Exact reference comparison](claim_exact_reference_comparison.md) | Exact full-tree CFR+ against sampled tables with an exact average, neural regrets with O4 refits, and the online-average run, by iteration. Also how to read `p_first` and `p_second`. | **Current (2 October).** Sampled tables beat exact CFR+ early (about 2×) but flatten later; neural regrets with O4 are about 2× worse than the table. |
| [Preparing the first 30-claim run](30_claim_first_run_setup.md) | Why the smoke checks, hardware limits, resume changes, and approximate evaluation led to the final launch configuration. | Setup rationale; see the linked experiment record for live results. |

## Reach, units and regret targets

| Note | What it covers | Currency |
| --- | --- | --- |
| [From exact CFR+ to sampled regret updates](neural_cfr_plus_regret_units_and_bridge.md) | The tabular bridge: how reach `q` and conditional advantage `g` are estimated when sampling, with regrets and average kept exact. | Background for the [tabular bridge](../experiments/18_claim/2026-09-29_01-42-50_18_claim_tabular_bridge.md). |
| [Where reach goes in exact, sampled and neural CFR+](cfr_plus_reach_sampling_and_neural_targets.md) | Why a sampled table keeps reach weighting through visits, while a neural target fitted on visits does not scale its increment by reach. | Background; the experiments since found that adding `N/K` or `N` multipliers to neural targets hurts. |
| [Clip on read](clip_on_read_regret_targets.md) | Specification for fitting signed regret targets and clipping on read. | Implemented. Its status line is out of date: the 18-claim result is in, and clip on read was worse ([results](../experiments/18_claim/2026-09-30_10-37-00_18_claim_clip_on_read_and_aggregation_mse.md)). |
| [Aggregate-then-clip, clip on read and the hybrid](aggregation_and_clip_on_read.md) | Worked examples of the three target constructions and when each should win. | **Partly superseded.** It argued the hybrid should combine both strengths; at 18 claims the hybrid was 1.3–1.7× worse than aggregate-then-clip ([results](../experiments/18_claim/2026-09-30_10-37-00_18_claim_clip_on_read_and_aggregation_mse.md)). The single-visit argument for larger games is untested. |

## Why training stalls, and how to train at scale

| Note | What it covers | Currency |
| --- | --- | --- |
| [Why more compute stops helping](why_more_compute_does_not_help.md) | Hypothesis (28 September) for why exploitability falls, levels out and rises with more iterations. | **Largely superseded.** Most of the plateau turned out to be the online average network, about 5× worse than the exact average ([offline average fitting](../experiments/18_claim/2026-09-30_21-47-00_18_claim_offline_average_fitting.md)), plus a regret-network cost of about 2× ([neural O4 run](../experiments/18_claim/2026-10-01_07-56-40_18_claim_neural_o4_refit_cpu.md)). |
| [GPU training schedules](gpu_training_schedules.md) | Open questions about fitting steps, learning rate, batch size and averaging on the GPU, with proposed diagnostics. | Plan from 30 September. The averaging questions are answered by Part A of the [schedule experiment](../experiments/18_claim/2026-10-01_10-25-57_18_claim_average_fit_traversal_schedule_regret_noise.md) (use O4); the regret-fit questions remain open (Part C). |
| [Learned history baselines](learned_history_baselines.md) | The baseline idea from VR-MCCFR, DREAM and VR-DeepDCFR+: estimate untaken actions' values with low variance. | Idea only; not implemented. Relevant once traverser actions are sampled rather than fully expanded. |
| [Neural CFR+ refactor plan](neural_cfr_plus_refactor_plan.md) | What the small refactor did, which paper features the code must accommodate, and a staged plan. | Plan; the structural refactor is deliberately delayed until a feature needs it. |

## Evaluation beyond exact

| Note | What it covers | Currency |
| --- | --- | --- |
| [Best responses without exact evaluation](non_exact_best_response_methods.md) | Approximate best-response families (LBR, depth-limited search, MCTS, learned responders) and how to calibrate them at 18 claims before 30. | Proposal (1 October); nothing implemented. |
