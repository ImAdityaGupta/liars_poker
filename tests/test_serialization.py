import os

from liars_poker.core import GameSpec
from liars_poker.env import rules_for_spec
from liars_poker.logging import read_strategy_manifest
from liars_poker.policy import (
    CommitOnceMixture,
    PerDecisionMixture,
    RandomPolicy,
    TabularPolicy,
    policy_from_json,
)
from liars_poker.simple_api import mix_policies, start_run, load_policy


def test_policy_serialization_roundtrip():
    spec = GameSpec(ranks=5, suits=1, hand_size=1, starter="random", claim_kinds=("RankHigh",))
    rules = rules_for_spec(spec)
    base = RandomPolicy()
    base.bind_rules(rules)
    data = base.to_json()
    loaded = policy_from_json(data)
    assert isinstance(loaded, RandomPolicy)
    loaded.bind_rules(rules)

    tab = TabularPolicy()
    key = (0, -2, (1,), ())
    tab.set(key, {1: 0.6, 2: 0.4})
    tab_loaded = policy_from_json(tab.to_json())
    tab_loaded.bind_rules(rules)
    probs = tab_loaded.action_probs(key)
    assert abs(probs[1] - 0.6) < 1e-9

    rand_a = RandomPolicy()
    rand_b = RandomPolicy()
    rand_c = RandomPolicy()

    mix = PerDecisionMixture(rand_a, rand_b, 0.3)
    mix_loaded = policy_from_json(mix.to_json())
    assert isinstance(mix_loaded, PerDecisionMixture)
    assert abs(mix_loaded.w - 0.3) < 1e-9

    commit = CommitOnceMixture([rand_a, rand_b, rand_c], [0.7, 0.2, 0.1])
    commit_loaded = policy_from_json(commit.to_json())
    assert isinstance(commit_loaded, CommitOnceMixture)
    assert len(commit_loaded.weights) == 3
    assert abs(sum(commit_loaded.weights) - 1.0) < 1e-9


def test_simple_api_run_logging(tmp_path):
    spec = GameSpec(ranks=5, suits=1, hand_size=1, starter="random", claim_kinds=("RankHigh",))
    run = start_run(spec, save_root=str(tmp_path), seed=99)
    pid0 = run.log_policy(RandomPolicy(), role="average", seed=11)
    assert pid0 == "A0"
    assert run.current_policy_id == pid0
    current = run.current_policy()
    assert isinstance(current, RandomPolicy)
    loaded = load_policy(run.run_dir, pid0)
    assert isinstance(loaded, RandomPolicy)

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
