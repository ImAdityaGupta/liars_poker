import numpy as np

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.algo.br_limited_dense import LimitedBestResponse
from liars_poker.core import GameSpec
from liars_poker.policies.neural import NeuralPolicy, compile_neural_to_dense
from liars_poker.policies.tabular_dense import DenseTabularPolicy
import torch


def test_full_depth_matches_exact_for_nonuniform_opponent():
    spec = GameSpec(ranks=3, suits=2, hand_size=1,
                    claim_kinds=('RankHigh', 'Pair'), suit_symmetry=True)
    opponent = DenseTabularPolicy(spec)
    rng = np.random.default_rng(42)
    opponent.S[:] = rng.uniform(.01, 1, opponent.S.shape) * opponent.legal_mask[:, None, :]
    opponent.S[:] /= opponent.S.sum(axis=2, keepdims=True)
    opponent.recompute_likelihoods()

    _, exact = best_response_dense(spec, opponent, store_state_values=False)
    target = exact['computer'].exploitability()
    full = LimitedBestResponse(opponent, depth=opponent.k, epsilon=0)
    observed = [full.evaluate_seat(seat)['p'] for seat in (0, 1)]
    np.testing.assert_allclose(observed, target, atol=1e-6)

    shallow = LimitedBestResponse(opponent, depth=2, epsilon=1e-3)
    discovered = [shallow.evaluate_seat(seat)['p'] for seat in (0, 1)]
    assert all(found <= best + 1e-6 for found, best in zip(discovered, target))


def test_lazy_neural_matches_compiled_dense():
    torch.manual_seed(12)
    spec = GameSpec(ranks=3, suits=2, hand_size=1,
                    claim_kinds=('RankHigh', 'Pair'), suit_symmetry=True)
    policy = NeuralPolicy(spec, hidden_sizes=(16, 16))
    with torch.no_grad():
        for model in (policy.model_p1, policy.model_p2):
            torch.nn.init.normal_(model.net[-1].weight, std=.1)
            torch.nn.init.normal_(model.net[-1].bias, std=.1)
    dense = compile_neural_to_dense(policy)
    lazy = LimitedBestResponse(policy, depth=3, epsilon=1e-4)
    compiled = LimitedBestResponse(dense, depth=3, epsilon=1e-4)
    for hid in (0, 1, 3, 17, 33):
        np.testing.assert_allclose(lazy._probs(hid), dense.S[hid], atol=1e-7)
        for seat in (0, 1):
            np.testing.assert_allclose(lazy._reach(hid, seat), compiled._reach(hid, seat), atol=1e-7)
    for seat in (0, 1):
        np.testing.assert_allclose(lazy.evaluate_seat(seat)['p'],
                                   compiled.evaluate_seat(seat)['p'], atol=1e-6)
    assert lazy.network_queries > 0
    assert lazy._probs.cache_info().hits > 0
