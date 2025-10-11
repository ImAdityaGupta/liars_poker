Liar’s Poker — Minimal Research Skeleton

This repository provides a lightweight, extensible skeleton for two-player Liar’s Poker.
It focuses on:
- Flexible environment config (ranks/suits/hand size/claim kinds)
- Minimal Env mechanics (deal, legal actions, apply, infosets)
- Policy interfaces, Random/Tabular policies, and two mixers:
  - PerDecisionMixture
  - CommitOnceMixture
- Logging manifests for lineage and reproducibility
- FSP meta-loop scaffold (placeholders for learners/BR)

Quickstart

from liars_poker.core import GameSpec
from liars_poker.env import Env
from liars_poker.policy import RandomPolicy, PerDecisionMixture, CommitOnceMixture
from liars_poker.fsp import train_fsp

spec = GameSpec(ranks=13, suits=1, hand_size=2, starter="random", claim_kinds=("RankHigh","Pair"))

env = Env(spec, seed=123)
obs = env.reset()
assert env.current_player() in ("P1","P2")
assert -1 in obs["legal_actions"] or len(obs["legal_actions"]) > 0

pi = RandomPolicy()
beta = RandomPolicy()
mixA = PerDecisionMixture(pi, beta, w=0.1)
mixB = CommitOnceMixture(pi, beta, w=0.1)

run_info = train_fsp(spec, eta_schedule="harmonic", mix="commit_once", max_iters=1, seed=7)
print("Run dir:", run_info["run_dir"])
print("Average policy id:", run_info["average_policy_id"])

# Simple API helpers
from liars_poker.simple_api import start_run, mix_policies, play_vs_bot

run = start_run(spec, seed=123)
pid = run.log_policy(RandomPolicy(), role="average", seed=123)
mix = mix_policies(run.current_policy(), RandomPolicy(), {"impl": "commit_once", "w": 0.05})
run.log_policy(
    mix,
    role="average",
    parents=[{"id": pid, "role": "avg", "weight": 0.95}],
    mixing={"impl": "commit_once", "eta_k": 0.05},
)
play_vs_bot(spec, run.current_policy(), my_cards=[1, 3], bot_cards="random", start="me")

Scripts
- scripts/run_fsp.py: Parse configs/exp/fsp_k2_demo.yaml and run the FSP scaffold.
- scripts/play_human.py: Play a text-based match (human vs random/mix policy).

Tests
- tests/test_rules.py: Turn rules and raising order.
- tests/test_infoset.py: Determinism and infoset identity.
- tests/test_mixing.py: Mixer behavior and shapes.
- tests/test_repro.py: Seeded reproducibility.
