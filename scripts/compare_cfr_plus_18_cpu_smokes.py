#!/usr/bin/env python3
"""Compare parallel CPU smoke-test throughput after early startup iterations."""

import argparse
import json
from pathlib import Path


def summarize(root: Path):
    values = []
    for path in sorted(root.glob("*/**/training.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) <= 6:
            raise ValueError(f"Need more than six iterations in {path}")
        sampled = rows[5:]
        elapsed = sampled[-1]["measured_training_s"] - rows[4]["measured_training_s"]
        rate = len(sampled) / elapsed
        values.append((path.parent.name, rate, len(rows)))
    print(f"\n{root}: {len(values)} arms")
    for name, rate, count in values:
        print(f"  {name}: {rate:.3f} iterations/s after warmup; {count} total iterations")
    print(f"  sum of concurrent arm rates: {sum(rate for _, rate, _ in values):.3f} iterations/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    for root in args.roots:
        summarize(root)
