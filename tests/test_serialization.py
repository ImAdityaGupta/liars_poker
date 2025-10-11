import os

from liars_poker.core import GameSpec
from liars_poker.logging import read_strategy_manifest
from liars_poker.policy import (
    CommitOnceMixture,
    PerDecisionMixture,
    RandomPolicy,
    TabularPolicy,
    policy_from_json,
)
from liars_poker.simple_api import mix_policies, start_run


def test_policy_serialization_roundtrip():
    base = RandomPolicy()
    data = base.to_json()
    loaded = policy_from_json(data)
    assert isinstance(loaded, RandomPolicy)

    tab = TabularPolicy()
    key = (0, -2, (1,), ())
    tab.set(key, {1: 0.6, 2: 0.4})
    tab_loaded = policy_from_json(tab.to_json())
    probs = tab_loaded.action_probs(key, [1, 2])
    assert abs(probs[1] - 0.6) < 1e-9

    mix = PerDecisionMixture(RandomPolicy(), RandomPolicy(), 0.3)
    mix_loaded = policy_from_json(mix.to_json())
    assert isinstance(mix_loaded, PerDecisionMixture)
    assert abs(mix_loaded.w - 0.3) < 1e-9

    commit = CommitOnceMixture(RandomPolicy(), RandomPolicy(), 0.1)
    commit_loaded = policy_from_json(commit.to_json())
    assert isinstance(commit_loaded, CommitOnceMixture)
    assert abs(commit_loaded.w - 0.1) < 1e-9


def test_simple_api_run_logging(tmp_path):
    spec = GameSpec(ranks=5, suits=1, hand_size=1, starter="random", claim_kinds=("RankHigh",))
    run = start_run(spec, save_root=str(tmp_path), seed=99)
    pid0 = run.log_policy(RandomPolicy(), role="average", seed=11)
    assert pid0 == "A0"
    assert run.current_policy_id == pid0
    current = run.current_policy()
    assert isinstance(current, RandomPolicy)

    mix = mix_policies(current, RandomPolicy(), {"impl": "commit_once", "w": 0.2})
    pid1 = run.log_policy(
        mix,
        role="average",
        parents=[{"id": pid0, "role": "avg", "weight": 0.8}],
        mixing={"impl": "commit_once", "eta_k": 0.2},
    )
    assert pid1 == "A1"
    manifest = read_strategy_manifest(os.path.join(run.manifests_dir, "A1.json"))
    assert manifest.parents[0]["id"] == "A0"
    assert manifest.artifacts["policy"] == "policies/A1.json"
