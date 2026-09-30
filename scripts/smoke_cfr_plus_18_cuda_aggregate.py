#!/usr/bin/env python3
"""Short CUDA check for 18-claim aggregate-then-clip checkpoint continuation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer, DeviceRecentBuffer


def gib(n: int) -> float:
    return n / 1024**3


def memory() -> dict[str, float]:
    return {
        "allocated_gib": round(gib(torch.cuda.memory_allocated()), 3),
        "reserved_gib": round(gib(torch.cuda.memory_reserved()), 3),
        "peak_allocated_gib": round(gib(torch.cuda.max_memory_allocated()), 3),
        "peak_reserved_gib": round(gib(torch.cuda.max_memory_reserved()), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--roots", type=int, default=4096)
    parser.add_argument("--soak-iterations", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(8)
    cpu = DeepCFRPlusTrainer.load_checkpoint(args.checkpoint, device="cpu")
    if cpu.regret_target_mode != "aggregate_then_clip" or cpu.regret_accumulation_mode != "cumulative":
        raise ValueError("Supply a cumulative aggregate-then-clip checkpoint")

    # Checkpoints may have empty recent regret buffers. Use real strategy-replay
    # infosets/masks and deterministic per-visit regret targets instead.
    source = cpu.strategy_buffers[0]
    n = min(source.size, 60_000)
    if not n:
        raise ValueError("The source checkpoint has no replay features")
    generator = torch.Generator().manual_seed(19)
    features = source.features[:n].detach().cpu()
    targets = 0.01 * torch.randn((n, source.action_dim), generator=generator)
    masks = source.legal_masks[:n].detach().cpu()
    weights = source.weights[:n].detach().cpu()
    buffers = []
    for device in ("cpu", "cuda"):
        buffer = DeviceRecentBuffer(n, source.input_dim, source.action_dim, device)
        buffer.add_many(features, targets, masks, weights)
        buffers.append(buffer)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for buffer in buffers:
        DeepCFRPlusTrainer._aggregate_regret_targets(buffer)
    torch.cuda.synchronize()
    parity = (buffers[0].targets[:n] - buffers[1].targets[:n].cpu()).abs().max().item()
    print(json.dumps({"event": "group_parity", "rows": n, "max_abs_diff": parity,
                      "elapsed_s": round(time.perf_counter() - start, 3), "cuda_memory": memory()}),
          flush=True)
    if parity > 1e-5:
        raise AssertionError("CPU and CUDA grouped targets differ")
    del buffers, cpu
    gc.collect()
    torch.cuda.empty_cache()

    trainer = DeepCFRPlusTrainer.load_checkpoint(args.checkpoint, device="cuda")
    print(json.dumps({"event": "checkpoint_loaded", "iteration": trainer.iteration,
                      "cuda_memory": memory()}), flush=True)
    for steps in (24, 24, 96, 384, 768):
        trainer.regret_train_steps = steps
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        row = trainer.run_iteration(traversals_per_player=args.roots)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if not all(torch.isfinite(torch.tensor(row[key])).all() for key in ("regret_loss", "strategy_loss")):
            raise AssertionError("Nonfinite fit loss")
        print(json.dumps({"event": "iteration", "steps": steps, "iteration": trainer.iteration,
                          "elapsed_s": round(elapsed, 3), "timing": row["timing"],
                          "regret_rows": row["new_regret_records"],
                          "cuda_memory": memory()}), flush=True)

    with tempfile.TemporaryDirectory(prefix="cfr18_cuda_smoke_") as tmp:
        checkpoint = Path(tmp) / "smoke_checkpoint.pt"
        trainer.save_checkpoint(checkpoint)
        expected_iteration = trainer.iteration
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        restored = DeepCFRPlusTrainer.load_checkpoint(checkpoint, device="cuda")
        if (restored.iteration != expected_iteration
                or restored.regret_train_steps != 768
                or restored.regret_accumulation_mode != "cumulative"):
            raise AssertionError("CUDA checkpoint did not restore the smoke-run state")
        print(json.dumps({"event": "checkpoint_restored", "iteration": restored.iteration,
                          "cuda_memory": memory()}), flush=True)
        if args.soak_iterations:
            restored.regret_train_steps = 24
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(args.soak_iterations):
                row = restored.run_iteration(traversals_per_player=args.roots)
                if not all(torch.isfinite(torch.tensor(row[key])).all()
                           for key in ("regret_loss", "strategy_loss")):
                    raise AssertionError("Nonfinite fit loss during soak")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            print(json.dumps({"event": "soak", "iterations": args.soak_iterations,
                              "elapsed_s": round(elapsed, 3),
                              "seconds_per_iteration": round(elapsed / args.soak_iterations, 3),
                              "final_iteration": restored.iteration,
                              "cuda_memory": memory()}), flush=True)


if __name__ == "__main__":
    main()
