import random

import pytest

from liars_poker.core import GameSpec
from liars_poker.env import Env, rules_for_spec
from liars_poker.policies.base import Policy
from liars_poker.policies.commit_once import CommitOnceMixture
from liars_poker.policies.random import RandomPolicy
from liars_poker.policies.tabular import TabularPolicy


def test_random_policy_store_load(tmp_path):
    spec = GameSpec(ranks=3, suits=2, hand_size=1)
    rules = rules_for_spec(spec)
    policy = RandomPolicy()
    policy.bind_rules(rules)
    directory = tmp_path / "random"
    policy.store_efficiently(str(directory))

    loaded_policy, loaded_spec = Policy.load_policy(str(directory))
    assert loaded_spec == spec
    env = Env(spec)
    env.reset(seed=0)
    infoset = env.infoset_key("P1")
    dist = loaded_policy.prob_dist_at_infoset(infoset)
    assert pytest.approx(sum(dist.values()), rel=1e-9) == 1.0


def test_tabular_policy_store_load(tmp_path):
    spec = GameSpec(ranks=3, suits=2, hand_size=1)
    rules = rules_for_spec(spec)
    policy = TabularPolicy()
    policy.bind_rules(rules)
    env = Env(spec)
    env.reset(seed=1)
    infoset = env.infoset_key("P1")
    legal = env.legal_actions()
    policy.set(infoset, {legal[0]: 0.7, legal[1]: 0.3})
    policy.set_annotations(values={infoset: 0.42}, visits={infoset: 9})

    directory = tmp_path / "tabular"
    policy.store_efficiently(str(directory))

    loaded_policy, loaded_spec = Policy.load_policy(str(directory))
    assert loaded_spec == spec
    loaded_env = Env(spec)
    loaded_env.reset(seed=1)
    loaded_infoset = loaded_env.infoset_key("P1")
    loaded_legal = loaded_env.legal_actions()
    loaded_dist = loaded_policy.action_probs(loaded_infoset)
    assert pytest.approx(loaded_dist[loaded_legal[0]], rel=1e-9) == 0.7
    assert pytest.approx(loaded_dist[loaded_legal[1]], rel=1e-9) == 0.3
    assert loaded_policy.get_value(loaded_infoset) == 0.42
    assert loaded_policy.get_visits(loaded_infoset) == 9


def test_commit_once_store_load(tmp_path):
    spec = GameSpec(ranks=3, suits=2, hand_size=1)
    rules = rules_for_spec(spec)
    rng = random.Random(3)

    base_random = RandomPolicy()
    base_random.bind_rules(rules)

    tabular = TabularPolicy()
    tabular.bind_rules(rules)
    env = Env(spec)
    env.reset(seed=2)
    infoset = env.infoset_key("P1")
    legal = env.legal_actions()
    tabular.set(infoset, {legal[0]: 1.0})

    mixture = CommitOnceMixture([base_random, tabular], [0.5, 0.5], rng=rng)
    mixture.bind_rules(rules)

    directory = tmp_path / "mixture"
    mixture.store_efficiently(str(directory))

    loaded_mixture, loaded_spec = Policy.load_policy(str(directory))
    assert loaded_spec == spec
    assert len(loaded_mixture.policies) == 2
    assert loaded_mixture.weights == pytest.approx([0.5, 0.5], rel=1e-9)

    env2 = Env(spec)
    env2.reset(seed=4)
    iset2 = env2.infoset_key("P1")
    dist = loaded_mixture.prob_dist_at_infoset(iset2)
    assert pytest.approx(sum(dist.values()), rel=1e-9) == 1.0
