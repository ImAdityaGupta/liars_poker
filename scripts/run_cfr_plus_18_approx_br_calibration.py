#!/usr/bin/env python3
"""Restartable CPU calibration of local and depth-limited BR on 18 claims."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.algo.br_limited_dense import LimitedBestResponse
from liars_poker.policies.neural import compile_neural_to_dense
from liars_poker.policies.tabular_dense import DenseTabularPolicy
from liars_poker.serialization import load_policy


ARTIFACTS = Path('/root/liars_poker/artifacts')
NEURAL = ARTIFACTS / 'cfr_plus_18_neural_o4_cpu/main_20261001'
EXACT = ARTIFACTS / 'cfr_plus_18_batched_bridge_controls/main_20260930/exact4096'
FIT = ARTIFACTS / 'cfr_plus_18_average_fit_long_run_check/main_20261002/1080m/O4/warm_05000'


def targets() -> list[tuple[str, Path]]:
    def neural(k: int, minute: int, kind: str) -> Path:
        return NEURAL / f'neural_o4_k{k}/policy_snapshots/{minute:04d}m/{kind}_policy'
    return [
        ('01_k4096_o4_0015', neural(4096, 15, 'average')),
        ('02_k4096_online_0015', neural(4096, 15, 'online')),
        ('03_k4096_o4_0060', neural(4096, 60, 'average')),
        ('04_k4096_o4_0180', neural(4096, 180, 'average')),
        ('05_k4096_online_0180', neural(4096, 180, 'online')),
        ('06_k4096_o4_0450', neural(4096, 450, 'average')),
        ('07_k4096_o4_0795', neural(4096, 795, 'average')),
        ('08_k4096_o4_1140', neural(4096, 1140, 'average')),
        ('09_k4096_online_1140', neural(4096, 1140, 'online')),
        ('10_k1024_o4_0600', neural(1024, 600, 'average')),
        ('11_exact4096_o4_1080', FIT),
        ('12_exact4096_table_1080', EXACT / 'policy_snapshots/1080m/average_policy'),
    ]


SETTINGS = [(1, 0.0), (2, 1e-3), (2, 1e-4), (3, 1e-3), (3, 1e-4)]


def atomic_json(path: Path, row: dict) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(row, indent=2), encoding='utf-8')
    tmp.replace(path)


def run_one(label: str, path: str, out: str, only: str | None) -> str:
    torch.set_num_threads(1)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    out_dir = Path(out)
    policy, spec = load_policy(path)
    dense = policy if isinstance(policy, DenseTabularPolicy) else compile_neural_to_dense(policy, batch_size=65536)
    del policy
    exact_file = out_dir / f'{label}_exact.json'
    if not exact_file.exists():
        start = time.perf_counter()
        _, meta = best_response_dense(spec, dense, store_state_values=False)
        p0, p1 = meta['computer'].exploitability()
        atomic_json(exact_file, {'label': label, 'policy_dir': path,
                                  'p_first': p0, 'p_second': p1,
                                  'exploitability': p0 + p1 - 1,
                                  'elapsed_s': time.perf_counter() - start})
        del meta
    for depth, epsilon in SETTINGS:
        name = f'{label}_d{depth}_e{epsilon:g}.json'
        if only is not None and name != only:
            continue
        dest = out_dir / name
        if dest.exists():
            continue
        responder = LimitedBestResponse(dense, depth, epsilon)
        rows = []
        for seat in (0, 1):
            rows.append(responder.evaluate_seat(seat))
        row = {'label': label, 'policy_dir': path, 'method': 'M1' if depth == 1 else 'M2',
               'depth': depth, 'epsilon': epsilon, 'p_first': rows[0]['p'],
               'p_second': rows[1]['p'],
               'discovered_exploitability': rows[0]['p'] + rows[1]['p'] - 1,
               'elapsed_s': sum(r['elapsed_s'] for r in rows), 'seats': rows}
        atomic_json(dest, row)
        print(f'{label} d={depth} eps={epsilon:g} value={row["discovered_exploitability"]:.6f} '
              f's={row["elapsed_s"]:.1f}', flush=True)
        del responder
    return label


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--one', help='Run just one policy label for a smoke check')
    p.add_argument('--setting', help='Run just one output filename for a smoke check')
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [(name, str(path), str(args.output_root), args.setting) for name, path in targets()
            if args.one is None or args.one == name]
    missing = [path for _, path, _, _ in jobs if not Path(path, 'metadata.json').exists()]
    if missing:
        raise FileNotFoundError(f'Missing policy files: {missing}')
    print(f'{len(jobs)} policies, {len(SETTINGS)} settings each, {args.workers} workers', flush=True)
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, *job): job[0] for job in jobs}
        for future in as_completed(futures):
            try:
                print('finished:', future.result(), flush=True)
            except Exception:
                failures.append(futures[future])
                traceback.print_exc()
    if failures:
        raise RuntimeError(f'Failed policies: {failures}')


if __name__ == '__main__':
    main()
