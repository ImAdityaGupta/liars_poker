#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from typing import Any, Dict

from liars_poker.core import ARTIFACTS_ROOT, GameSpec
from liars_poker.fsp import train_fsp


def parse_simple_yaml(path: str) -> Dict[str, Any]:
    """Very small YAML-like parser for the demo config.
    Supports key: value and one-line list [a, b].
    """
    data: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1].strip()
                if inner:
                    items = [x.strip() for x in inner.split(",")]
                else:
                    items = []
                data[key] = items
            elif val.isdigit():
                data[key] = int(val)
            else:
                data[key] = val
    return data


def main() -> None:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "exp", "fsp_k2_demo.yaml")
    cfg_path = os.path.abspath(cfg_path)
    if not os.path.exists(cfg_path):
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = parse_simple_yaml(cfg_path)
    spec = GameSpec(
        ranks=int(cfg.get("ranks", 13)),
        suits=int(cfg.get("suits", 1)),
        hand_size=int(cfg.get("hand_size", 2)),
        starter=str(cfg.get("starter", "random")),
        claim_kinds=tuple(cfg.get("claim_kinds", ["RankHigh", "Pair"])),
    )
    save_root = cfg.get("save_root") or cfg.get("fsp.save_root") or ARTIFACTS_ROOT
    save_root = os.path.abspath(save_root)
    fsp_cfg = {
        "eta_schedule": cfg.get("eta_schedule", cfg.get("fsp.eta_schedule", "harmonic")),
        "mix": cfg.get("mix", cfg.get("fsp.mix", "commit_once")),
        "max_iters": int(cfg.get("max_iters", cfg.get("fsp.max_iters", 1))),
        "save_root": save_root,
        "seed": int(cfg.get("seed", cfg.get("fsp.seed", 0))),
    }
    out = train_fsp(spec, **fsp_cfg)
    print("Run directory:", out["run_dir"])
    print("Average policy id:", out["average_policy_id"])


if __name__ == "__main__":
    main()
