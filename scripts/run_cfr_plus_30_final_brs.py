#!/usr/bin/env python3
"""Five-minute fitted-return BR cross-check on each final 30-claim O4 policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from liars_poker.training.br_runs import run_best_responder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--minutes', type=float, default=5.)
    parser.add_argument('--eval-episodes', type=int, default=200_000)
    args = parser.parse_args()
    root = args.output_root.resolve()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer_kwargs = {
        'state_hidden_sizes': (512, 512), 'action_hidden_sizes': (128, 128),
        'embedding_dim': 256, 'device': device, 'replay_capacity': 1_000_000,
        'batch_size': 4096, 'learning_rate': 1e-3, 'train_steps': 100,
        'warmup_transitions': 20_000, 'epsilon_start': 1., 'epsilon_end': .05,
        'epsilon_decay_decisions': 500_000, 'rollouts_per_action': 1,
        'fused_optimizer': device.type == 'cuda', 'seed': 17,
    }
    for arm in ('w512', 'w2048'):
        run = root / arm
        final = json.loads((run / 'summary.json').read_text())
        if final['status'] != 'target_reached':
            raise RuntimeError(f'{arm} is not complete')
        policy = Path(final['final_o4_policy'])
        out = run / 'final_fitted_br'
        if (out / 'comparison.json').exists():
            print(f'[skip] {arm} comparison already complete', flush=True)
            continue
        if (out / 'summary.json').exists():
            records = [json.loads(line) for line in
                       (out / 'evaluations.jsonl').read_text().splitlines() if line.strip()]
            measured_min = float(json.loads((out / 'summary.json').read_text())['measured_training_s']) / 60
        else:
            result = run_best_responder(
                policy, method='action_conditioned_fitted_return',
                minutes=args.minutes, trainer_kwargs=trainer_kwargs,
                episodes_per_role=4096, rollout_batch_size=1024,
                evaluate_every_minutes=1., eval_episodes_per_role=args.eval_episodes,
                run_dir=out, debug=True)
            records = result.evaluation_records
            measured_min = result.measured_training_s / 60
        row = {
            'arm': arm, 'policy_dir': str(policy),
            'responder_training_min': measured_min,
            'best_discovered_estimate': max(float(x['exploitability_estimate']) for x in records),
            'final_estimate': float(records[-1]['exploitability_estimate']),
            'final_lower_bound': float(records[-1]['exploitability_lower_bound']),
            'episodes_per_role': args.eval_episodes,
        }
        (out / 'comparison.json').write_text(json.dumps(row, indent=2))
        print(row, flush=True)


if __name__ == '__main__':
    main()
