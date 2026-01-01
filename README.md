Liar's Poker – Core, Policies, and Best Responses
=================================================

This repo is a two-player Liar's Poker playground with:
- A minimal engine (`core.py`, `infoset.py`, `env.py`)
- Multiple policy representations (tabular, dense, mixtures)
- Exact and approximate best responses (tabular and dense)
- Simple evaluation and FSP-style training utilities

Core
----------------
- **Engine**: `GameSpec` describes ranks/suits/hand_size/claim_kinds. `Rules` drives legality and call resolution. `Env` steps games (P1 always opens).
- **Policies**:
  - `TabularPolicy`: sparse dict over `InfoSet -> action dist`, persisted via JSON + NPZ blobs.
  - `DenseTabularPolicy`: canonical `(hid, hand, action)` tensor with likelihood tables for posterior-aware mixing; serializable; fast to query.
  - `RandomPolicy`, `CommitOnceMixture` (posterior-aware mixture).
- **Best responses**:
  - `algo/br_exact.py`: exact BR for tabular opponents.
  - `algo/br_exact_dense_to_dense.py`: exact BR for dense opponents (no prob_dist calls).
  - `algo/br_mc.py`: Monte Carlo BR against a black-box opponent.
- **Evaluation**: `eval/match.py` (`play_match`, `eval_seats_split`, exact eval for tabular).
- **Training**:
  - `training/fsp.py`: tabular FSP loop (BR function injected).
  - `training/dense_fsp.py`: dense FSP loop for `DenseTabularPolicy`.
- **Serialization**: `serialization.py` registers all policies and saves/loads to `metadata.json` + `blobs.npz`. Works for tabular, dense, mixtures.

Quickstart
----------
Sample a policy, compute a BR, evaluate both seats, and save:
```python
from liars_poker import GameSpec, Env
from liars_poker.policies import RandomPolicy
from liars_poker.eval.match import eval_seats_split
from liars_poker.algo.br_mc import best_response_mc
from liars_poker.serialization import save_policy, load_policy
from liars_poker.infoset import InfoSet
from liars_poker.core import possible_starting_hands

spec = GameSpec(ranks=6, suits=1, hand_size=1, claim_kinds=("RankHigh",))
env = Env(spec, seed=123)
base = RandomPolicy(); base.bind_rules(env.rules)

br, info = best_response_mc(spec, base, episodes=2000, epsilon=0.1)
seat_split = eval_seats_split(spec, base, br, episodes=500)
print("Seat-wise win rates vs BR:", seat_split)

# Save/load
save_policy(base, "artifacts/demo_random")
loaded, loaded_spec = load_policy("artifacts/demo_random")
opening = InfoSet(pid=0, hand=possible_starting_hands(loaded_spec)[0], history=())
print("Loaded opening dist:", loaded.action_probs(opening))
```

Dense policy workflow
---------------------
- Build a `DenseTabularPolicy` for fast querying/mixing.
- Exact dense BR: `best_response_dense(spec, dense_policy)` (returns a dense BR + metadata with `computer.exploitability()`).
- Dense FSP loop: `training.dense_fsp.dense_fsp_loop` (logs exploitability, uses `eval_seats_split`).
- Plot/save runs: `training.fsp_utils.plot_exploitability_series`, `training.fsp_utils.save_fsp_run`.

Notebooks
---------
Under `notebooks/` you'll find correctness/benchmark harnesses (e.g., BR correctness, serialization tests, small-spec benchmarks). These mirror the current dense/tabular endpoints.
