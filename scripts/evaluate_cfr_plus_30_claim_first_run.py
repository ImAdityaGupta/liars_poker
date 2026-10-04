#!/usr/bin/env python3
"""Evaluate saved 30-claim policies with the lazy depth-limited responder."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from liars_poker.algo.br_limited_dense import LimitedBestResponse
from liars_poker.infoset import CALL
from liars_poker.serialization import load_policy


def targets(root: Path, final_minute: int):
    for arm in ('w512', 'w2048'):
        for snap in sorted((root / arm / 'snapshots').glob('*')):
            if not (snap / 'READY.json').exists():
                continue
            minute = int(snap.name.removesuffix('m'))
            yield arm, snap, 'o4', 2, snap / 'average_policy'
            if minute % 240 == 0 or minute == final_minute:
                yield arm, snap, 'o4', 3, snap / 'average_policy'
            if minute == final_minute:
                yield arm, snap, 'o4', 4, snap / 'average_policy'
            if minute in (360, 720, final_minute):
                yield arm, snap, 'online', 2, snap / 'online_policy'
                yield arm, snap, 'online', 3, snap / 'online_policy'
    for baseline in sorted((root / 'baseline_june').glob('*')):
        if not (baseline / 'metadata.json').exists():
            continue
        for depth in (2, 3):
            yield f'june_{baseline.name}', Path('0060m'), 'june', depth, baseline


def run_one(args: argparse.Namespace) -> None:
    import torch
    torch.set_num_threads(1)
    begin = time.perf_counter()
    policy, spec = load_policy(str(args.policy))
    solver = LimitedBestResponse(policy, depth=args.depth, epsilon=1e-4)
    if args.exact:
        seats = [solver.evaluate_seat(seat) for seat in (0, 1)]
    else:
        seats = [evaluate_seat_mc(solver, seat, args.episodes, 17_000 + 101 * seat)
                 for seat in (0, 1)]
    p_first, p_second = seats[0]['p'], seats[1]['p']
    uncertainty = sum(s.get('half_width', 0.) for s in seats)
    row = {
        'arm': args.arm, 'snapshot': args.snapshot, 'policy_kind': args.kind,
        'policy_dir': str(args.policy), 'depth': args.depth, 'epsilon': 1e-4,
        'p_first': p_first, 'p_second': p_second,
        'discovered_exploitability': p_first + p_second - 1,
        'lower_confidence_bound': p_first + p_second - 1 - uncertainty,
        'evaluation_mode': 'exact_tree' if args.exact else 'mc_full_games',
        'episodes_per_seat': None if args.exact else args.episodes,
        'elapsed_s': time.perf_counter() - begin,
        'network_queries': solver.network_queries,
        'network_rows': solver.network_rows, 'network_query_s': solver.network_query_s,
        'seats': seats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(row, indent=2), encoding='utf-8')
    os.replace(temporary, args.output)
    print(f'{args.arm} {args.snapshot} {args.kind} d{args.depth} '
          f'{row["discovered_exploitability"]:.6f} {row["elapsed_s"]:.1f}s', flush=True)


def evaluate_seat_mc(solver: LimitedBestResponse, seat: int, episodes: int,
                     seed: int) -> dict:
    """Sample complete legal games; responder decisions retain exact beliefs."""
    rng = np.random.default_rng(seed)
    joint = solver.hand_weights[:, None] * solver.blockers
    joint /= joint.sum()
    pairs = rng.choice(solver.n * solver.n, size=episodes, p=joint.ravel())
    wins = 0
    started = time.perf_counter()
    for episode, pair in enumerate(pairs, 1):
        my_hand, opp_hand = divmod(int(pair), solver.n)
        hid = 0
        while True:
            turn = hid.bit_count() & 1
            if turn == seat:
                action = solver._choice(hid, my_hand, seat)
            else:
                legal = solver._actions(hid)
                row = solver._probs(hid)[opp_hand]
                probs = np.array([row[0 if a == CALL else a + 1] for a in legal],
                                 dtype=np.float64)
                probs /= probs.sum()
                action = legal[min(int(np.searchsorted(np.cumsum(probs), rng.random())),
                                   len(legal) - 1)]
            if action == CALL:
                truth = bool(solver.truth[solver._last_claim(hid)][my_hand, opp_hand])
                wins += int((not truth) if turn == seat else truth)
                break
            hid |= 1 << action
        if episode % 1000 == 0:
            print(f'[mc] seat={seat} episodes={episode}/{episodes} '
                  f'elapsed_s={time.perf_counter()-started:.1f} '
                  f'queries={solver.network_queries}', flush=True)
    p = wins / episodes
    half_width = 1.96 * np.sqrt(p * (1. - p) / episodes)
    return {'seat': seat, 'p': float(p), 'half_width': float(half_width),
            'episodes': episodes, 'elapsed_s': time.perf_counter()-started,
            'plan_nodes': solver.plan_nodes, 'network_queries': solver.network_queries}


def worker(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    while not (root / 'STOP_EVALUATOR').exists():
        found = False
        for arm, snap, kind, depth, policy in targets(root, int(args.minutes)):
            if args.arm_filter and arm != args.arm_filter:
                continue
            if args.depth_filter and depth != args.depth_filter:
                continue
            if args.june_only and not arm.startswith('june_'):
                continue
            if not args.june_only and arm.startswith('june_'):
                continue
            output = root / 'evaluations' / f'{arm}_{snap.name}_{kind}_d{depth}.json'
            failure = root / 'eval_failures' / output.name
            if output.exists() or failure.exists() or (root / f'SKIP_{output.stem}').exists():
                continue
            found = True
            episodes = args.episodes if depth == 2 else min(args.episodes, 500 if depth == 3 else 100)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'one',
                       '--arm', arm, '--snapshot', snap.name, '--kind', kind,
                       '--depth', str(depth), '--policy', str(policy), '--output', str(output),
                       '--episodes', str(episodes)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            try:
                subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=args.timeout_s)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                print(f'[eval error] {output.name}: {exc}', flush=True)
                failure.parent.mkdir(parents=True, exist_ok=True)
                failure.write_text(json.dumps({'policy': str(policy),
                                               'depth': depth, 'error': str(exc)}))
        if args.once:
            return
        if not found:
            time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('one', 'worker'))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--minutes', type=float, default=1440)
    parser.add_argument('--timeout-s', type=int, default=3600)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--exact', action='store_true',
                        help='Enumerate the entire evaluation tree (small games only)')
    parser.add_argument('--episodes', type=int, default=2_000)
    parser.add_argument('--arm-filter')
    parser.add_argument('--depth-filter', type=int)
    parser.add_argument('--june-only', action='store_true')
    parser.add_argument('--arm')
    parser.add_argument('--snapshot')
    parser.add_argument('--kind')
    parser.add_argument('--depth', type=int)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'one':
        run_one(args)
    else:
        worker(args)


if __name__ == '__main__':
    main()
