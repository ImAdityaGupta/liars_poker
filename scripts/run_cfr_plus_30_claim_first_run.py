#!/usr/bin/env python3
"""Resumable two-arm 30-claim CFR+ run; hourly O4 fits run after both arms stop."""
from __future__ import annotations

import argparse
import fcntl
import gc
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer
from scripts.run_cfr_plus_18_neural_o4_cpu import fit_one, freeze_input, publish_online, publish_current
from scripts.smoke_cfr_plus_30_cuda_aggregate import SPEC

ARMS = {'w512': 512, 'w2048': 2048}
REGRET_CAP = 2_000_000
STRATEGY_CAP = 4_000_000
RAMP_START = 1024
RAMP_END = 16384


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(temporary, path)


def append(path: Path, value: dict) -> None:
    with path.open('a', encoding='utf-8') as out:
        out.write(json.dumps(value, default=float, sort_keys=True) + '\n')
        out.flush()


def manifest(args: argparse.Namespace, arm: str) -> dict:
    return {
        'spec': json.loads(SPEC.to_json()), 'arm': arm, 'seed': 17,
        'regret_width': ARMS[arm], 'strategy_width': 512,
        'regret_train_steps': 24, 'strategy_train_steps': 6,
        'batch_size': 1024, 'learning_rate': 1e-3,
        'regret_target_mode': 'aggregate_then_clip',
        'regret_accumulation_mode': 'cumulative',
        'regret_increment_reach_mode': 'none',
        'regret_positive_weight': 0., 'strategy_weighting': 'linear',
        'regret_buffer_capacity': REGRET_CAP,
        'strategy_buffer_capacity': STRATEGY_CAP,
        'traversal_batch_size': 256, 'root_ramp': [RAMP_START, RAMP_END],
        'target_minutes': args.minutes, 'snapshot_minutes': args.snapshot_minutes,
        'o4_steps': args.fit_steps, 'o4_batch': args.fit_batch,
    }


def make_trainer(arm: str, spec=SPEC) -> DeepCFRPlusTrainer:
    width = ARMS[arm]
    return DeepCFRPlusTrainer(
        spec, device='cuda', seed=17, regret_hidden_sizes=(width, width),
        strategy_hidden_sizes=(512, 512), learning_rate=1e-3,
        batch_size=1024, regret_batch_size=1024,
        regret_train_steps=24, strategy_train_steps=6,
        regret_buffer_capacity=REGRET_CAP,
        strategy_buffer_capacity=STRATEGY_CAP,
        regret_target_mode='aggregate_then_clip',
        regret_increment_reach_mode='none', regret_accumulation_mode='cumulative',
        regret_positive_weight=0., strategy_weighting='linear',
        traversal_backend='gpu_native', traversal_batch_size=256,
        device_replay=True, fused_optimizer=False, validation_fraction=0.)


def roots(measured_s: float, target_s: float) -> int:
    fraction = min(1., max(0., measured_s / target_s))
    return min(RAMP_END, RAMP_START + 256 * math.floor((RAMP_END - RAMP_START) * fraction / 256))


def checkpoint(trainer: DeepCFRPlusTrainer, path: Path, progress: dict) -> None:
    # The old checkpoint remains valid until the replacement is fully written.
    old_size = path.stat().st_size if path.exists() else 0
    if shutil.disk_usage(path.parent).free < old_size + 3 * 1024**3:
        raise OSError(f'Insufficient disk space for atomic checkpoint: {path}')
    temporary = path.with_name(path.name + '.tmp')
    state = trainer.checkpoint_dict()
    state['experiment_progress'] = progress
    torch.save(state, temporary)
    os.replace(temporary, path)


def progress_of(path: Path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)['experiment_progress']


def stage_fit_input(trainer: DeepCFRPlusTrainer, path: Path, progress: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if Path('/dev/shm').is_dir():
        backing = Path('/dev/shm') / f'cfr30_{path.parents[2].name}_{path.parent.name}_fit.pt'
        freeze_input(trainer, backing, progress)
        if path.is_symlink() or path.exists():
            path.unlink()
        path.symlink_to(backing)
    else:
        freeze_input(trainer, path, progress)


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(2)
    root = args.output_root.resolve()
    run = root / args.arm
    run.mkdir(parents=True, exist_ok=True)
    config = manifest(args, args.arm)
    config_path = run / 'manifest.json'
    ckpt = run / 'latest_checkpoint.pt'
    if ckpt.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(f'Manifest differs from checkpoint: {run}')
        progress = progress_of(ckpt)
        trainer = DeepCFRPlusTrainer.load_checkpoint(ckpt, device='cuda')
        if trainer.iteration != progress['iteration']:
            raise ValueError('Checkpoint iteration mismatch')
        print(f'[resume] {args.arm} {progress["measured_training_s"]/60:.2f}m iter={trainer.iteration}', flush=True)
    else:
        if config_path.exists():
            raise RuntimeError(f'Manifest without checkpoint: {run}')
        trainer = make_trainer(args.arm)
        progress = {'iteration': 0, 'measured_training_s': 0.,
                    'next_snapshot_s': 60. * args.snapshot_minutes,
                    'cumulative_roots': 0, 'pending_snapshot': None}
        atomic_json(config_path, config)
        checkpoint(trainer, ckpt, progress)

    target_s = args.minutes * 60.
    if progress['pending_snapshot']:
        label = progress['pending_snapshot']
        directory = run / 'snapshots' / label
        if not (directory / 'FIT_INPUT.pt').exists() and not (directory / 'READY.json').exists():
            # A VM restart loses /dev/shm. The checkpoint contains this reservoir.
            staged = directory / 'FIT_INPUT.pt'
            stage_fit_input(trainer, staged, progress)
        publish_online(trainer, directory, progress)
        publish_current(trainer, directory, progress)
        marker = run / 'CONTINUE.json'
        if not (directory / 'READY.json').exists() or not marker.exists() or json.loads(marker.read_text())['snapshot'] != label:
            print(f'[pending] {args.arm} {label}', flush=True)
            return
        progress['pending_snapshot'] = None
        checkpoint(trainer, ckpt, progress)
        marker.unlink()

    stop = [False]
    def ask_stop(_signum, _frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, ask_stop)
    signal.signal(signal.SIGINT, ask_stop)
    next_log = time.monotonic() + 60
    while progress['measured_training_s'] < target_s and not stop[0] and not (root / 'PAUSE').exists():
        k = roots(progress['measured_training_s'], target_s)
        started = time.perf_counter()
        row = trainer.run_iteration(traversals_per_player=k)
        elapsed = time.perf_counter() - started
        if not all(math.isfinite(float(v)) for v in row['regret_loss'] + row['strategy_loss']):
            raise FloatingPointError('Non-finite loss; previous checkpoint remains valid')
        progress['measured_training_s'] += elapsed
        progress['iteration'] = trainer.iteration
        progress['cumulative_roots'] += k
        append(run / 'training.jsonl', {
            'utc': utc(), 'arm': args.arm, 'iteration': trainer.iteration,
            'measured_training_min': progress['measured_training_s'] / 60,
            'cumulative_roots': progress['cumulative_roots'], 'roots_per_player': k,
            'iteration_s': elapsed, 'timing': row['timing'],
            'regret_loss': row['regret_loss'], 'strategy_loss': row['strategy_loss'],
            'regret_records': row['new_regret_records'],
            'visited_infosets': row['visited_infosets'],
            'rows_per_visited_set': [
                count / groups if groups else None for count, groups in
                zip(row['new_regret_records'], row['visited_infosets'])],
            'strategy_records': row['new_strategy_records'],
            'strategy_buffer_sizes': row['strategy_buffer_sizes'],
            'strategy_records_seen': row['strategy_records_seen'],
        })
        if time.monotonic() >= next_log:
            next_log = time.monotonic() + 60
            print(f'[train] {args.arm} {progress["measured_training_s"]/60:.1f}m '
                  f'iter={trainer.iteration} K={k} iter_s={elapsed:.2f} '
                  f'rows={row["new_regret_records"]}', flush=True)
        if progress['measured_training_s'] >= min(progress['next_snapshot_s'], target_s):
            planned_s = min(progress['next_snapshot_s'], target_s)
            label = f'{round(planned_s/60):04d}m'
            directory = run / 'snapshots' / label
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / 'READY.json').exists():
                raise RuntimeError(f'Already completed snapshot: {directory}')
            staged = directory / 'FIT_INPUT.pt'
            stage_fit_input(trainer, staged, progress)
            progress['pending_snapshot'] = label
            progress['next_snapshot_s'] += 60. * args.snapshot_minutes
            checkpoint(trainer, ckpt, progress)
            publish_online(trainer, directory, progress)
            publish_current(trainer, directory, progress)
            append(run / 'snapshots.jsonl', {
                'utc': utc(), 'arm': args.arm, 'snapshot': label,
                'iteration': trainer.iteration,
                'measured_training_min': progress['measured_training_s']/60,
                'cumulative_roots': progress['cumulative_roots'],
                'status': 'awaiting_o4',
            })
            print(f'[snapshot] {args.arm} {label}', flush=True)
            return
    checkpoint(trainer, ckpt, progress)
    atomic_json(run / 'summary.json', {
        'status': 'target_reached' if progress['measured_training_s'] >= target_s else 'paused',
        **progress, 'updated_utc': utc(),
    })


def fit(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    for arm in ARMS:
        for staged in sorted((root / arm / 'snapshots').glob('*/FIT_INPUT.pt')):
            print(f'[fit] {arm} {staged.parent.name}', flush=True)
            fit_one(staged, args.fit_steps, args.fit_batch, 2, device='cuda')
            gc.collect()
            torch.cuda.empty_cache()


def controller(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_handle = (root / 'controller.lock').open('a+')
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f'Controller already running for {root}') from exc
    command = [sys.executable, '-u', str(Path(__file__).resolve()), 'train',
               '--output-root', str(root), '--minutes', str(args.minutes),
               '--snapshot-minutes', str(args.snapshot_minutes),
               '--fit-steps', str(args.fit_steps), '--fit-batch', str(args.fit_batch)]
    loops = 0
    while not (root / 'PAUSE').exists():
        loops += 1
        if loops > math.ceil(args.minutes / args.snapshot_minutes) + 3:
            raise RuntimeError('Too many controller cycles; check progress')
        # Two live trainers share the GPU. Both exit before O4 refitting starts.
        running = {}
        for arm in ARMS:
            log = (root / arm / 'train.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open('a', encoding='utf-8')
            process = subprocess.Popen([*command, '--arm', arm], cwd=ROOT,
                                       stdout=handle, stderr=subprocess.STDOUT)
            running[arm] = (process, handle)
        codes = {}
        for arm, (process, handle) in running.items():
            codes[arm] = process.wait()
            handle.close()
        print(f'[cycle] train exits {codes}', flush=True)
        if any(code != 0 for code in codes.values()):
            raise RuntimeError(f'Trainer failed: {codes}; see per-arm train.log')
        fit_command = [sys.executable, '-u', str(Path(__file__).resolve()), 'fit',
                       '--output-root', str(root), '--minutes', str(args.minutes),
                       '--snapshot-minutes', str(args.snapshot_minutes),
                       '--fit-steps', str(args.fit_steps), '--fit-batch', str(args.fit_batch)]
        with (root / 'fit.log').open('a', encoding='utf-8') as handle:
            subprocess.run(fit_command, cwd=ROOT, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        complete = True
        for arm in ARMS:
            ckpt = root / arm / 'latest_checkpoint.pt'
            progress = progress_of(ckpt)
            pending = progress['pending_snapshot']
            if pending:
                ready = root / arm / 'snapshots' / pending / 'READY.json'
                if not ready.exists():
                    raise RuntimeError(f'O4 missing after fit: {ready}')
                progress['pending_snapshot'] = None
                # The next train child reads checkpoint progress with pending
                # still set. It must clear this field before continuing.
                atomic_json(root / arm / 'CONTINUE.json', {'snapshot': pending})
            complete &= progress['measured_training_s'] >= args.minutes * 60
        if complete:
            for arm in ARMS:
                progress = progress_of(root / arm / 'latest_checkpoint.pt')
                atomic_json(root / arm / 'summary.json', {
                    'status': 'target_reached', **progress, 'updated_utc': utc(),
                    'final_o4_policy': str(root / arm / 'snapshots' /
                                           progress['pending_snapshot'] / 'average_policy'),
                })
            (root / 'ALL_DONE').touch()
            print('[complete] both arms reached target', flush=True)
            return
        if (root / 'PAUSE').exists():
            break
    print('[paused] remove PAUSE and rerun controller to resume', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'fit', 'controller'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--minutes', type=float, default=1440.)
    parser.add_argument('--snapshot-minutes', type=float, default=60.)
    parser.add_argument('--fit-steps', type=int, default=5000)
    parser.add_argument('--fit-batch', type=int, default=16384)
    args = parser.parse_args()
    if args.minutes <= 0 or args.snapshot_minutes <= 0 or args.fit_steps <= 0:
        parser.error('time and fit parameters must be positive')
    if args.command == 'train':
        if args.arm is None:
            parser.error('--arm required for train')
        train(args)
    elif args.command == 'fit':
        fit(args)
    else:
        controller(args)


if __name__ == '__main__':
    main()
