Liar’s Poker — Research Skeleton

The repository focuses on a compact core for two-player Liar’s Poker:

1. **Engine** – `core.py`, `infoset.py`, `env.py`
   - `GameSpec` defines the deck / claim structure (P1 always opens).
   - `InfoSet` is a frozen dataclass describing the player’s perspective (`pid`, `hand`, `history`).
2. **Policies** – `policies/`
   - `Policy` base class with `action_probs`, `prob_dist_at_infoset`, sampling, and efficient persistence hooks.
   - Implementations: `RandomPolicy`, `TabularPolicy`, and a posterior-aware `CommitOnceMixture`.
3. **Algorithms** – `algo/br_mc.py` for Monte-Carlo best responses.
4. **Evaluation** – `eval/match.py` provides simple match/evaluation helpers.

## Quickstart

```python
from liars_poker import GameSpec, Env, RandomPolicy, best_response_mc, eval_both_seats

spec = GameSpec(ranks=13, suits=1, hand_size=2)
env = Env(spec, seed=123)
obs = env.reset()
assert env.current_player() == "P1"

candidate = RandomPolicy()
candidate.bind_rules(env.rules)

opponent = best_response_mc(spec, candidate, episodes=1000, epsilon=0.1)
results = eval_both_seats(spec, candidate, opponent, episodes=500)
print(results)
```

Policies can persist themselves using the efficient binary helpers:

```python
directory = "artifacts/random_policy"
candidate.store_efficiently(directory)
from liars_poker.policies.base import Policy
loaded_policy, loaded_spec = Policy.load_policy(directory)
```

## Tests

Key regression tests live under `tests/`:

- `tests/test_p1_always_starts.py`
- `tests/test_infoset_basics.py`
- `tests/test_commit_once_posterior.py`
- `tests/test_br_annotations.py`
- `tests/test_policy_storage.py`
