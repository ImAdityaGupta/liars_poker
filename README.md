Liar’s Poker — Research Skeleton

This repository now exposes a layered stack for two-player Liar’s Poker:

1. **Engine** – `core.py`, `infoset.py`, `env.py`
   - `GameSpec` no longer carries a `starter`; P1 always opens.
   - `InfoSet` is a named tuple (`pid`, `last_idx`, `hand`, `history`) so callers can clearly identify whose turn it is.
2. **Policies** – `policies/`
   - Base `Policy` ABC with `action_probs`, `sample`, and `prob_dist_at_infoset`.
   - `RandomPolicy`, `TabularPolicy`, and a posterior-aware `CommitOnceMixture` (the per-decision mixer has been removed).
3. **Algorithms** – `algo/br_mc.py` for Monte-Carlo best responses (returns a `TabularPolicy` annotated with values/visits).
4. **Training** – `training/` contains `FSPConfig`, scheduling helpers, and `FSPTrainer`.
5. **IO** – `io/` bundles JSON helpers, manifests, policy (de)serialization, and the new `RunManager` (policies stored as `.json.gz`).
6. **Evaluation** – `eval/match.py` runs matches and seat-swapped evaluations.

Because seating randomisation is now handled outside the environment and infosets are named, this release is **not backwards compatible** with earlier APIs.

## Quickstart

```python
from liars_poker import (
    GameSpec,
    Env,
    RandomPolicy,
    best_response_mc,
    RunManager,
    FSPConfig,
    FSPTrainer,
    eval_both_seats,
)

spec = GameSpec(ranks=13, suits=1, hand_size=2)
env = Env(spec, seed=123)
obs = env.reset()
assert env.current_player() == "P1"

# Initialise a run directory with a random average policy (A0)
run = RunManager(spec, save_root="artifacts", seed=7)
a0 = run.log_policy(RandomPolicy(), role="average", parents=[], mixing=None, seed=7, train={"algo": "init_random"})

# Train one FSP iteration
config = FSPConfig(episodes0=5000, epsilon=0.1, min_visits=1, max_iters=1, seed=7)
trainer = FSPTrainer(run, config)
br_policy, avg_policy, br_id, avg_id, metrics = trainer.step(run.current_policy(), iter_index=0)
print("Logged:", br_id, avg_id, metrics)

# Play a quick match
p1 = avg_policy
p2 = best_response_mc(spec, RandomPolicy(), episodes=200, epsilon=0.0, min_visits_per_action=1)
result = eval_both_seats(spec, p1, p2, episodes=200)
print(result)
```

## CLI

- `scripts/lp-train` – minimal command-line entry-point wrapping `RunManager` + `FSPTrainer`.
- `scripts/play_human.py` – simple text UI to play as P1 against a random opponent.

## Tests

The suite focuses on the new invariants:

- `tests/test_p1_always_starts.py`
- `tests/test_infoset_basics.py`
- `tests/test_commit_once_posterior.py`
- `tests/test_br_annotations.py`
- `tests/test_run_manager_logging.py`
