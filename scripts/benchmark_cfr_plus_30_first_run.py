#!/usr/bin/env python3
"""Pilot 30-claim GPU memory, regret-row counts, and iteration costs."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer
from scripts.smoke_cfr_plus_30_cuda_aggregate import SPEC


def trainer(width: int, regret_cap: int, strategy_cap: int, traversal_batch: int) -> DeepCFRPlusTrainer:
    return DeepCFRPlusTrainer(
        SPEC, device='cuda', seed=17, regret_hidden_sizes=(width, width),
        strategy_hidden_sizes=(512, 512), learning_rate=1e-3,
        batch_size=1024, regret_batch_size=1024,
        regret_train_steps=24, strategy_train_steps=6,
        regret_buffer_capacity=regret_cap,
        strategy_buffer_capacity=strategy_cap,
        regret_target_mode='aggregate_then_clip',
        regret_increment_reach_mode='none', regret_accumulation_mode='cumulative',
        regret_positive_weight=0., strategy_weighting='linear',
        traversal_backend='gpu_native', traversal_batch_size=traversal_batch,
        device_replay=True, fused_optimizer=False, validation_fraction=0.)


def memory() -> dict:
    free, total = torch.cuda.mem_get_info()
    return {'free_gib': round(free / 1024**3, 3),
            'allocated_gib': round(torch.cuda.memory_allocated() / 1024**3, 3),
            'peak_gib': round(torch.cuda.max_memory_allocated() / 1024**3, 3)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--regret-cap', type=int, default=2_000_000)
    p.add_argument('--strategy-cap', type=int, default=4_000_000)
    p.add_argument('--iterations-per-k', type=int, default=5)
    p.add_argument('--traversal-batch', type=int, default=256)
    p.add_argument('--ks', type=int, nargs='+', default=[1024, 4096, 16384, 32768])
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    with args.output.open('a', encoding='utf-8') as out:
        models = {}
        for width in (512, 2048):
            models[width] = trainer(width, args.regret_cap, args.strategy_cap,
                                    args.traversal_batch)
            row = {'event': 'allocated', 'width': width, 'memory': memory()}
            out.write(json.dumps(row) + '\n'); out.flush()
            print(row, flush=True)
        if memory()['free_gib'] < 2:
            raise MemoryError('Less than 2 GiB free after allocating both arms')
        for k in args.ks:
            for i in range(args.iterations_per_k):
                for width, model in models.items():
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    result = model.run_iteration(traversals_per_player=k)
                    elapsed = time.perf_counter() - start
                    row = {'event': 'iteration', 'width': width, 'k': k,
                           'iteration': model.iteration, 'elapsed_s': elapsed,
                           'timing': result['timing'],
                           'regret_rows': result['new_regret_records'],
                           'strategy_rows': result['new_strategy_records'],
                           'memory': memory()}
                    out.write(json.dumps(row) + '\n'); out.flush()
                    print(f'width={width} k={k} i={i+1} time={elapsed:.2f}s '
                          f'rows={row["regret_rows"]} mem={row["memory"]}', flush=True)
        del models
        gc.collect()


if __name__ == '__main__':
    main()
