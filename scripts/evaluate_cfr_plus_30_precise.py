#!/usr/bin/env python3
"""Stratified, belief-averaged depth-two BR scores for saved 30-claim policies.

Each shard scores the same own-hand sweeps and random inputs for every policy.
Independent shards can run on separate CPU cores. This never loads a training
checkpoint or uses CUDA.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from liars_poker.algo.br_limited_dense import LimitedBestResponse
from liars_poker.infoset import CALL
from liars_poker.serialization import load_policy


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + '.tmp')
    staged.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(staged, path)


def targets(root: Path) -> list[dict]:
    found = []
    for arm in ('w512', 'w2048'):
        for snap in sorted((root / arm / 'snapshots').glob('*')):
            policy = snap / 'average_policy'
            if (snap / 'READY.json').exists() and (policy / 'metadata.json').exists():
                found.append({'id': f'{arm}_{snap.name}_o4', 'arm': arm,
                              'snapshot': snap.name, 'kind': 'o4', 'policy': str(policy)})
    for policy in sorted((root / 'baseline_june').glob('*')):
        if (policy / 'metadata.json').exists():
            found.append({'id': f'june_{policy.name}_0060m',
                          'arm': f'june_{policy.name}', 'snapshot': '0060m',
                          'kind': 'june', 'policy': str(policy)})
    # Finish paired early snapshots before spending CPU on later ones.
    return sorted(found, key=lambda t: (int(t['snapshot'][:-1]),
                                        0 if t['arm'] == 'w512' else
                                        1 if t['arm'] == 'w2048' else 2,
                                        t['arm']))


def terminal_expectation(solver: LimitedBestResponse, hid: int, hand: int,
                         seat: int, caller: int) -> float:
    """Expected win given every public action, including an opponent CALL."""
    mass = solver.blockers[hand] * solver._reach(hid, seat)
    if caller != seat:
        mass = mass * solver._probs(hid)[:, 0]
    total = float(mass.sum())
    if total <= 0:
        raise ArithmeticError('Sampled terminal history has zero belief mass')
    truth = solver.truth[solver._last_claim(hid)][hand]
    return float(np.dot(mass, truth if caller != seat else ~truth) / total)


def evaluate_seat(solver: LimitedBestResponse, seat: int, shard: int,
                  sweeps: int) -> np.ndarray:
    """One weighted sweep covers every responder hand type exactly once."""
    rng = np.random.default_rng(np.random.SeedSequence([30_2026, seat, shard]))
    own_prob = solver.hand_weights / solver.hand_weights.sum()
    conditional = solver.blockers / solver.blockers.sum(axis=1, keepdims=True)
    opp_cdf = np.cumsum(conditional, axis=1)
    # Generate inputs before playing, so divergent histories cannot shift the
    # random stream of later episodes. Shard IDs are shared across policies.
    opp_u = rng.random((sweeps, solver.n))
    action_u = rng.random((sweeps, solver.n, solver.k + 1))
    values = np.empty(sweeps, dtype=np.float64)
    for sweep in range(sweeps):
        weighted = 0.0
        for hand in range(solver.n):
            opp = min(int(np.searchsorted(opp_cdf[hand], opp_u[sweep, hand],
                                          side='right')), solver.n - 1)
            hid = 0
            while True:
                turn = hid.bit_count() & 1
                if turn == seat:
                    action = solver._choice(hid, hand, seat)
                else:
                    legal = solver._actions(hid)
                    row = solver._probs(hid)[opp]
                    probs = np.asarray([row[0 if a == CALL else a + 1]
                                        for a in legal], dtype=np.float64)
                    probs /= probs.sum()
                    u = action_u[sweep, hand, hid.bit_count()]
                    col = min(int(np.searchsorted(np.cumsum(probs), u,
                                                  side='right')), len(legal) - 1)
                    action = legal[col]
                if action == CALL:
                    weighted += own_prob[hand] * terminal_expectation(
                        solver, hid, hand, seat, turn)
                    break
                hid |= 1 << action
        values[sweep] = weighted
    return values


def one(args: argparse.Namespace) -> None:
    import torch
    torch.set_num_threads(1)
    if args.output.exists():
        return
    started = time.perf_counter()
    policy, _spec = load_policy(str(args.policy))
    solver = LimitedBestResponse(policy, depth=2, epsilon=1e-4)
    seat0 = evaluate_seat(solver, 0, args.shard, args.sweeps)
    seat1 = evaluate_seat(solver, 1, args.shard, args.sweeps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged = args.output.with_name(args.output.name + '.tmp')
    try:
        with staged.open('wb') as out:
            np.savez_compressed(out, seat0=seat0, seat1=seat1,
                                n_hands=np.int64(solver.n), sweeps=np.int64(args.sweeps),
                                shard=np.int64(args.shard))
        os.replace(staged, args.output)
    finally:
        staged.unlink(missing_ok=True)
    print(f'{args.policy} shard={args.shard} games/seat={solver.n*args.sweeps} '
          f'e={seat0.mean()+seat1.mean()-1:.6f} '
          f'elapsed={time.perf_counter()-started:.1f}s', flush=True)


def stats(scores: np.ndarray) -> tuple[float, float]:
    mean = float(scores.mean())
    se = float(scores.std(ddof=1) / math.sqrt(len(scores))) if len(scores) > 1 else float('nan')
    return mean, se


def load_shards(root: Path, target_id: str, count: int) -> dict[int, tuple[np.ndarray, np.ndarray, int]]:
    rows = {}
    for shard in range(count):
        path = root / 'shards' / target_id / f'{shard:03d}.npz'
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as arr:
            rows[shard] = (arr['seat0'].astype(np.float64),
                           arr['seat1'].astype(np.float64), int(arr['n_hands']))
    return rows


def report(root: Path, found: list[dict], shards: int) -> None:
    data = {t['id']: load_shards(root, t['id'], shards) for t in found}
    summaries = []
    for t in found:
        rows = data[t['id']]
        if not rows:
            continue
        a = np.concatenate([rows[k][0] for k in sorted(rows)])
        b = np.concatenate([rows[k][1] for k in sorted(rows)])
        # Independent RNG streams for the two seats; pair only across policies.
        p0, se0 = stats(a)
        p1, se1 = stats(b)
        se = math.hypot(se0, se1)
        estimate = p0 + p1 - 1
        summaries.append({**{k: t[k] for k in ('id', 'arm', 'snapshot', 'kind', 'policy')},
                          'depth': 2, 'evaluation_mode': 'stratified_terminal_belief',
                          'p_first': p0, 'p_second': p1,
                          'discovered_exploitability': estimate,
                          'standard_error': se, 'half_width_95': 1.96 * se,
                          'lower_confidence_bound': max(0.0, estimate - 1.96 * se),
                          'games_per_seat': len(a) * next(iter(rows.values()))[2],
                          'completed_shards': len(rows), 'target_shards': shards,
                          'sweeps': len(a)})
    comparisons = []
    t_by_id = {t['id']: t for t in found}
    pairs = []
    for t in found:
        if t['arm'] == 'w512':
            other = f'w2048_{t["snapshot"]}_o4'
            if other in t_by_id:
                pairs.append((t['id'], other))
        if t['arm'] in ('w512', 'w2048') and t['kind'] == 'o4':
            minute = int(t['snapshot'][:-1])
            prev = f'{t["arm"]}_{minute-60:04d}m_o4'
            if prev in t_by_id:
                pairs.append((t['id'], prev))
            if minute == 60:
                pairs.extend((t['id'], baseline['id']) for baseline in found
                             if baseline['kind'] == 'june')
    for left, right in pairs:
        a, b = data[left], data[right]
        common = sorted(a.keys() & b.keys())
        if not common:
            continue
        differences = []
        n_hands = a[common[0]][2]
        for shard in common:
            if a[shard][2] != b[shard][2] or len(a[shard][0]) != len(b[shard][0]):
                raise ValueError(f'Pairing mismatch: {left} vs {right}, shard {shard}')
            differences.append((a[shard][0] - b[shard][0]) +
                               (a[shard][1] - b[shard][1]))
        delta = np.concatenate(differences)
        mean, se = stats(delta)
        comparisons.append({'left': left, 'right': right,
                            'difference_left_minus_right': mean,
                            'standard_error': se, 'half_width_95': 1.96 * se,
                            'ci95': [mean - 1.96 * se, mean + 1.96 * se],
                            'paired_sweeps': len(delta),
                            'paired_games_per_seat': len(delta) * n_hands})
    atomic_json(root / 'summary.json', summaries)
    atomic_json(root / 'paired.json', comparisons)


def queue(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    out = root / 'precise_evaluations'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = {}
        while True:
            found = targets(root)
            for proc, (target_id, shard, started, handle) in list(active.items()):
                code = proc.poll()
                if code is None and time.monotonic() - started <= args.timeout_s:
                    continue
                if code is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    code = -1
                handle.close()
                if code != 0:
                    atomic_json(out / 'failures' / target_id / f'{shard:03d}.json',
                                {'target': target_id, 'shard': shard, 'exit_code': code})
                    print(f'[failure] {target_id} shard={shard} exit={code}', flush=True)
                del active[proc]
            running = {(t, s) for t, s, _, _ in active.values()}
            pending = [(t, shard) for t in found for shard in range(args.shards)
                       if not (out / 'shards' / t['id'] / f'{shard:03d}.npz').exists()
                       and not (out / 'failures' / t['id'] / f'{shard:03d}.json').exists()
                       and (t['id'], shard) not in running]
            while pending and len(active) < args.workers:
                target, shard = pending.pop(0)
                dst = out / 'shards' / target['id'] / f'{shard:03d}.npz'
                log = out / 'logs' / target['id'] / f'{shard:03d}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open('a', encoding='utf-8')
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'one',
                       '--policy', target['policy'], '--output', str(dst),
                       '--shard', str(shard), '--sweeps', str(args.sweeps)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           NUMEXPR_NUM_THREADS='1')
                proc = subprocess.Popen(cmd, cwd=REPO, env=env,
                                        stdout=handle, stderr=subprocess.STDOUT)
                active[proc] = (target['id'], shard, time.monotonic(), handle)
                running.add((target['id'], shard))
            report(out, found, args.shards)
            print(f'[queue] policies={len(found)} active={len(active)} '
                  f'pending={len(pending)}', flush=True)
            if (root / 'ALL_DONE').exists() and not active and not pending:
                return
            time.sleep(15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('one', 'queue', 'report'))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--sweeps', type=int, default=58)
    parser.add_argument('--shards', type=int, default=30)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--timeout-s', type=int, default=1800)
    args = parser.parse_args()
    if args.command == 'one':
        if args.policy is None or args.output is None:
            parser.error('one requires --policy and --output')
        one(args)
    else:
        if args.output_root is None:
            parser.error(f'{args.command} requires --output-root')
        if args.command == 'queue':
            queue(args)
        else:
            report(args.output_root.resolve() / 'precise_evaluations',
                   targets(args.output_root.resolve()), args.shards)


if __name__ == '__main__':
    main()
