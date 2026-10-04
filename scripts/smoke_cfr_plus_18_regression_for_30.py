#!/usr/bin/env python3
"""18-claim exact-evaluation regression of the 30-claim trainer recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from liars_poker.core import GameSpec
from scripts.run_cfr_plus_30_claim_first_run import make_trainer
from scripts.run_cfr_plus_18_neural_o4_cpu import freeze_input, fit_one, publish_online

SPEC18 = GameSpec(ranks=4, suits=4, hand_size=2,
                  claim_kinds=('RankHigh', 'Pair', 'TwoPair', 'Trips'),
                  suit_symmetry=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--fit-steps', type=int, default=5000)
    args = parser.parse_args()
    run = args.output_root.resolve()
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    trainer = make_trainer('w512', SPEC18)
    start = time.perf_counter()
    for i in range(args.iterations):
        row = trainer.run_iteration(traversals_per_player=4096)
        if (i + 1) % 100 == 0:
            print(f'iter={i+1} elapsed_min={(time.perf_counter()-start)/60:.1f} '
                  f'loss={row["regret_loss"]}', flush=True)
    directory = run / 'policy_snapshots' / '0015m'
    directory.mkdir(parents=True, exist_ok=True)
    progress = {'measured_training_s': time.perf_counter()-start}
    freeze_input(trainer, directory / 'FIT_INPUT.pt', progress)
    publish_online(trainer, directory, progress)
    del trainer
    torch.cuda.empty_cache()
    fit_one(directory / 'FIT_INPUT.pt', args.fit_steps, 16384, 2, device='cuda')
    for kind, policy in [('online', directory / 'online_policy'),
                         ('o4', directory / 'average_policy')]:
        result = subprocess.check_output(
            [sys.executable, str(ROOT / 'scripts/evaluate_cfr_plus_18_fit_snapshot.py'),
             str(policy)], cwd=ROOT, text=True)
        value = json.loads(result.strip().splitlines()[-1])
        print(f'{kind}: exact exploitability={value["exploitability"]:.6f}', flush=True)
        (run / f'{kind}_exact.json').write_text(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
