#!/usr/bin/env python3
"""Summarize the completed 18-claim local/depth-limited BR calibration."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    args = parser.parse_args()
    root = args.result_dir
    rows = []
    for file in sorted(root.glob('*_exact.json')):
        exact = json.loads(file.read_text())
        label = exact['label']
        for depth, eps in ((1, '0'), (2, '0.001'), (2, '0.0001'),
                           (3, '0.001'), (3, '0.0001')):
            result_file = root / f'{label}_d{depth}_e{eps}.json'
            if not result_file.exists():
                continue
            result = json.loads(result_file.read_text())
            raw = result['discovered_exploitability']
            ceiling = exact['exploitability']
            rows.append({'label': label, 'setting': f'd{depth} eps={eps}',
                         'exact': ceiling, 'raw_discovered': raw,
                         'lower_bound': max(0.0, raw),
                         'recovery': max(0.0, raw) / ceiling,
                         'elapsed_s': result['elapsed_s'],
                         'p_first': result['p_first'], 'p_second': result['p_second'],
                         'exact_p_first': exact['p_first'],
                         'exact_p_second': exact['p_second']})
    if not rows:
        raise RuntimeError('No results found')
    errors = [r for r in rows if r['p_first'] > r['exact_p_first'] + 1e-5
              or r['p_second'] > r['exact_p_second'] + 1e-5]
    if errors:
        raise AssertionError(f'Approximate BR exceeds exact BR on {len(errors)} rows')
    with (root / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    settings = sorted({r['setting'] for r in rows})
    colors = plt.cm.tab10.colors
    for i, setting in enumerate(settings):
        group = [r for r in rows if r['setting'] == setting]
        axes[0].scatter([r['exact'] for r in group], [r['recovery'] for r in group],
                        s=38, label=setting, color=colors[i])
        axes[1].scatter([r['elapsed_s'] for r in group], [r['recovery'] for r in group],
                        s=38, label=setting, color=colors[i])
    axes[0].set_xscale('log')
    axes[1].set_xscale('log')
    axes[0].set_xlabel('Exact exploitability')
    axes[1].set_xlabel('Responder CPU seconds, two seats')
    for ax in axes:
        ax.set_ylabel('Fraction of exact exploitability found')
        ax.axhline(1, color='black', lw=1, alpha=.4)
        ax.grid(True, alpha=.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / 'recovery.png', dpi=160)
    plt.close(fig)
    for setting in settings:
        group = [r for r in rows if r['setting'] == setting]
        print(setting, 'median recovery', round(float(np.median([r['recovery'] for r in group])), 3),
              'median seconds', round(float(np.median([r['elapsed_s'] for r in group])), 2),
              'recovery exact>.004', round(float(np.median([r['recovery'] for r in group if r['exact'] > .004])), 3))
    print('rows', len(rows), 'per-seat ceiling violations', len(errors))


if __name__ == '__main__':
    main()
