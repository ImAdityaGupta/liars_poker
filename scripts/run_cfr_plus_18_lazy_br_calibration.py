#!/usr/bin/env python3
"""Compare lazy neural-opponent BR with the saved dense 18-claim BR results."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import torch

from liars_poker.algo.br_limited_dense import LimitedBestResponse
from liars_poker.policies.tabular_dense import DenseTabularPolicy
from liars_poker.serialization import load_policy
from run_cfr_plus_18_approx_br_calibration import targets

OLD = Path('/root/liars_poker/artifacts/cfr_plus_18_approx_br_calibration/main_20261003')
SETTINGS = ((2, 1e-4), (3, 1e-4))


def save(path: Path, row: dict) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(row, indent=2), encoding='utf-8')
    tmp.replace(path)


def run_policy(label: str, path: str, root: str) -> str:
    torch.set_num_threads(1)
    out = Path(root)
    begin = time.perf_counter()
    policy, spec = load_policy(path)
    load_s = time.perf_counter() - begin
    if isinstance(policy, DenseTabularPolicy):
        # The twelfth reference policy is already a table. It still runs through
        # the shared search but has no neural-query cost to measure.
        kind = 'dense_reference'
    else:
        kind = 'lazy_neural'
    for depth, epsilon in SETTINGS:
        dest = out / f'{label}_d{depth}.json'
        if dest.exists():
            continue
        responder = LimitedBestResponse(policy, depth=depth, epsilon=epsilon)
        start = time.perf_counter()
        seats = [responder.evaluate_seat(seat) for seat in (0, 1)]
        elapsed = time.perf_counter() - start
        old = json.loads((OLD / f'{label}_d{depth}_e{epsilon:g}.json').read_text())
        seat_diffs = [seats[0]['p'] - old['p_first'], seats[1]['p'] - old['p_second']]
        row = {'label': label, 'policy_dir': path, 'kind': kind,
               'depth': depth, 'epsilon': epsilon,
               'p_first': seats[0]['p'], 'p_second': seats[1]['p'],
               'discovered_exploitability': seats[0]['p'] + seats[1]['p'] - 1,
               'dense_p_first': old['p_first'], 'dense_p_second': old['p_second'],
               'seat_differences': seat_diffs, 'matches_dense': max(map(abs, seat_diffs)) < 1e-5,
               'load_s': load_s, 'elapsed_s': elapsed,
               'network_queries': responder.network_queries,
               'network_rows': responder.network_rows,
               'network_query_s': responder.network_query_s,
               'cache': responder._probs.cache_info()._asdict(),
               'reach_cache': responder._reach.cache_info()._asdict(),
               'seats': seats}
        save(dest, row)
        print(f'{label} depth={depth} value={row["discovered_exploitability"]:.6f} '
              f'diff={max(map(abs, seat_diffs)):.2g} time={elapsed:.1f}s '
              f'queries={row["network_queries"]}', flush=True)
        if not row['matches_dense']:
            raise AssertionError(f'Lazy/dense mismatch for {label}, depth {depth}: {seat_diffs}')
        del responder
        gc.collect()
    return label


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--one', help='Run one policy label as a smoke check')
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [(name, str(path), str(args.output_root)) for name, path in targets()
            if args.one is None or args.one == name]
    for name, path, _ in jobs:
        if not (Path(path) / 'metadata.json').exists():
            raise FileNotFoundError(path)
        for depth, epsilon in SETTINGS:
            if not (OLD / f'{name}_d{depth}_e{epsilon:g}.json').exists():
                raise FileNotFoundError(f'Missing dense reference for {name}, depth {depth}')
    print(f'{len(jobs)} policies, {len(SETTINGS)} settings, {args.workers} CPU workers', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_policy, *job): job[0] for job in jobs}
        for future in as_completed(futures):
            print('finished:', future.result(), flush=True)


if __name__ == '__main__':
    main()
