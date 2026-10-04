#!/usr/bin/env python3
"""Check 30-claim aggregate targets, CUDA traversal, and checkpoint restore."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer, DeviceRecentBuffer
from liars_poker.core import GameSpec

SPEC = GameSpec(ranks=5, suits=4, hand_size=3,
                claim_kinds=('RankHigh', 'Pair', 'TwoPair', 'Trips', 'Quads'),
                suit_symmetry=True)


def main() -> None:
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    generator = torch.Generator().manual_seed(17)
    n, input_dim, action_dim = 3072, 35, 31
    x = torch.randint(0, 3, (256, input_dim), generator=generator).float()
    features = x[torch.randint(256, (n,), generator=generator)]
    targets = torch.randn(n, action_dim, generator=generator)
    masks = torch.rand(n, action_dim, generator=generator) > .4
    weights = torch.rand(n, generator=generator) + .1
    buffers = []
    for device in ('cpu', 'cuda'):
        b = DeviceRecentBuffer(n, input_dim, action_dim, device)
        b.require_no_overwrite = True
        b.add_many(features, targets, masks, weights)
        DeepCFRPlusTrainer._aggregate_regret_targets(
            b, accumulation_mode='cumulative', reach_mode='none')
        buffers.append(b)
    diff = float((buffers[0].targets - buffers[1].targets.cpu()).abs().max())
    print(json.dumps({'event': 'aggregation_parity', 'max_abs_diff': diff}), flush=True)
    if diff > 1e-5:
        raise AssertionError('CPU and CUDA aggregation disagree')
    try:
        buffers[0].add_many(features[:1], targets[:1], masks[:1], weights[:1])
    except OverflowError:
        pass
    else:
        raise AssertionError('Overflow guard did not raise')
    del buffers
    trainer = DeepCFRPlusTrainer(
        SPEC, device='cuda', seed=17, regret_hidden_sizes=(512, 512),
        strategy_hidden_sizes=(512, 512), learning_rate=1e-3,
        batch_size=1024, regret_batch_size=1024,
        regret_train_steps=2, strategy_train_steps=2,
        regret_buffer_capacity=100_000, strategy_buffer_capacity=100_000,
        regret_target_mode='aggregate_then_clip', regret_accumulation_mode='cumulative',
        regret_increment_reach_mode='none', regret_positive_weight=0.,
        strategy_weighting='linear', traversal_backend='gpu_native',
        traversal_batch_size=64, device_replay=True, fused_optimizer=False,
        validation_fraction=0.)
    for _ in range(2):
        start = time.perf_counter()
        row = trainer.run_iteration(traversals_per_player=64)
        print(json.dumps({'event': 'iteration', 'iteration': trainer.iteration,
                          'elapsed_s': time.perf_counter() - start,
                          'regret_rows': row['new_regret_records'],
                          'timing': row['timing']}), flush=True)
    with tempfile.TemporaryDirectory(prefix='cfr30_smoke_', dir='/dev/shm') as tmp:
        path = Path(tmp) / 'checkpoint.pt'
        trainer.save_checkpoint(path)
        restored = DeepCFRPlusTrainer.load_checkpoint(path, device='cuda')
        assert restored.iteration == trainer.iteration
        assert restored.spec == SPEC
        assert all(b.require_no_overwrite for b in restored.regret_buffers)
        print(json.dumps({'event': 'checkpoint_restored',
                          'iteration': restored.iteration}), flush=True)


if __name__ == '__main__':
    main()
