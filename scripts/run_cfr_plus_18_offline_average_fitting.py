#!/usr/bin/env python3
"""Refit a frozen strategy reservoir and compare with its exact average."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.algo.deep_cfr import DeviceReservoirBuffer
from liars_poker.core import GameSpec
from liars_poker.policies.neural import NeuralPolicy, compile_neural_to_dense
from liars_poker.policies.tabular_dense import DenseTabularPolicy
from liars_poker.serialization import save_policy


MILESTONES = (1000, 5000)
BATCH_SIZE = 1024
SEED = 17030


def write_row(output: Path, row: dict) -> None:
    with (output / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    print("[result] " + json.dumps(row), flush=True)
    plot(output)


def plot(output: Path) -> None:
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    by_name = {row["name"]: row for row in rows}
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for name, color in (("exact", "#374151"), ("online", "#D97706")):
        if name in by_name:
            ax.axhline(by_name[name]["exploitability"], color=color,
                       linestyle="--", label=f"{name}: {by_name[name]['exploitability']:.5f}")
    for variant, color in (("warm", "#2563EB"), ("fresh", "#DC2626")):
        selected = sorted((row for row in rows if row.get("variant") == variant),
                          key=lambda row: row["refit_steps"])
        if selected:
            ax.plot([row["refit_steps"] for row in selected],
                    [row["exploitability"] for row in selected],
                    color=color, marker="o", label=variant)
    ax.set(xlabel="Additional fitting steps per player",
           ylabel="Exact exploitability",
           title="Offline average fitting from one 18-claim checkpoint")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output / "fit_curve.png", dpi=160)
    plt.close(fig)


def exact_average(spec: GameSpec, observer: dict) -> DenseTabularPolicy:
    total = observer["sum"]
    if total is None:
        raise ValueError("Checkpoint has no accumulated exact average")
    total = torch.as_tensor(total).numpy()
    policy = DenseTabularPolicy(spec)
    if total.shape != policy.S.shape:
        raise ValueError(f"Exact observer shape {total.shape} != {policy.S.shape}")
    denominators = total.sum(axis=2, keepdims=True)
    np.divide(total, denominators, out=policy.S, where=denominators > 0)
    policy.recompute_likelihoods()
    return policy


def neural_policy(spec: GameSpec, state: dict | None, device: str) -> NeuralPolicy:
    policy = NeuralPolicy(spec, hidden_sizes=(256, 256), device=device)
    if state is not None:
        policy.model_p1.load_state_dict(state[0])
        policy.model_p2.load_state_dict(state[1])
    return policy.eval()


def evaluate(policy, reference: DenseTabularPolicy) -> dict:
    start = time.perf_counter()
    dense = (policy if isinstance(policy, DenseTabularPolicy)
             else compile_neural_to_dense(policy, batch_size=65_536))
    compilation_s = time.perf_counter() - start
    if dense is reference:
        tv_uniform = tv_own_reach = 0.0
    else:
        tv = 0.5 * np.abs(reference.S - dense.S).sum(axis=2)
        legal = reference.legal_counts[:, None] > 0
        tv_uniform = float(tv[legal.repeat(tv.shape[1], axis=1)].mean())
        own_reach = np.where((reference.popcount & 1)[:, None] == 0,
                             reference.L_pid0, reference.L_pid1)
        weights = np.where(legal, own_reach, 0.0)
        tv_own_reach = float((tv * weights).sum() / weights.sum())
        del tv, weights
    start = time.perf_counter()
    _, meta = best_response_dense(reference.spec, dense, store_state_values=False)
    p_first, p_second = meta["computer"].exploitability()
    evaluation_s = time.perf_counter() - start
    del meta
    if dense is not reference:
        del dense
    gc.collect()
    return {
        "p_first": float(p_first), "p_second": float(p_second),
        "exploitability": float(p_first + p_second - 1),
        "tv_uniform": tv_uniform, "tv_own_reach": tv_own_reach,
        "compilation_s": compilation_s, "evaluation_s": evaluation_s,
    }


def fit_player(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
               buffer: DeviceReservoirBuffer, steps: int, variant: str,
               pid: int) -> float:
    model.train()
    start = time.perf_counter()
    for step in range(1, steps + 1):
        x, y, mask, weight = buffer.sample(BATCH_SIZE)
        weight = weight / weight.mean().clamp_min(1e-8)
        logits = model(x).masked_fill(~mask, -1e9)
        per_sample = -(y * torch.log_softmax(logits, dim=1)).sum(dim=1)
        loss = (per_sample * weight).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 250 == 0:
            print(f"[fit] {variant} player={pid + 1} step={step}/{steps} "
                  f"loss={loss.item():.5f} elapsed={time.perf_counter()-start:.1f}s",
                  flush=True)
    model.eval()
    return time.perf_counter() - start


def run(checkpoint: Path, output: Path, variants: tuple[str, ...],
        milestones: tuple[int, ...], *, evaluate_policies: bool = True) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for offline fitting")
    print(f"[load] {checkpoint}", flush=True)
    load_start = time.perf_counter()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    spec = GameSpec(**{**state["spec"],
                       "claim_kinds": tuple(state["spec"]["claim_kinds"])})
    iteration = int(state["iteration"])
    progress = state["experiment_progress"]
    print(f"[load] iter={iteration} train={progress['measured_training_s']/60:.2f}m "
          f"buffer_sizes={[int(b['size']) for b in state['strategy_buffers']]} "
          f"elapsed={time.perf_counter()-load_start:.1f}s", flush=True)
    (output / "source.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "iteration": iteration,
        "measured_training_min": progress["measured_training_s"] / 60,
        "buffer_sizes": [int(b["size"]) for b in state["strategy_buffers"]],
        "buffer_seen": [int(b["seen"]) for b in state["strategy_buffers"]],
        "seed": SEED, "batch_size": BATCH_SIZE,
        "milestones": milestones, "variants": variants,
    }, indent=2), encoding="utf-8")
    reference = exact_average(spec, state["exact_average_observer"])
    for name, policy in (("exact", reference),
                         ("online", neural_policy(spec, state["strategy_nets"], "cuda"))):
        save_policy(policy, str(output / name))
        if evaluate_policies:
            print(f"[eval] {name}", flush=True)
            write_row(output, {"name": name, "iteration": iteration,
                               "refit_steps": 0, "fit_s": 0.0,
                               **evaluate(policy, reference)})
        else:
            print(f"[saved] {name}", flush=True)
        if name == "online":
            del policy

    buffers = [DeviceReservoirBuffer.from_state_dict(b, device="cuda")
               for b in state["strategy_buffers"]]
    print(f"[buffers] GPU resident; free GiB="
          f"{torch.cuda.mem_get_info()[0]/2**30:.2f}", flush=True)
    for variant in variants:
        torch.manual_seed(SEED + (1 if variant == "warm" else 2))
        policy = neural_policy(spec, state["strategy_nets"] if variant == "warm" else None,
                               "cuda")
        models = (policy.model_p1, policy.model_p2)
        optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3) for model in models]
        if variant == "warm":
            for optimizer, saved in zip(optimizers, state["strategy_optimizers"]):
                optimizer.load_state_dict(saved)
        torch.cuda.manual_seed_all(SEED + 100)
        fitted = 0
        fit_s = 0.0
        for milestone in milestones:
            additional = milestone - fitted
            fit_s += sum(fit_player(model, optimizer, buffer, additional, variant, pid)
                         for pid, (model, optimizer, buffer)
                         in enumerate(zip(models, optimizers, buffers)))
            fitted = milestone
            name = f"{variant}_{milestone:05d}"
            save_policy(policy, str(output / name))
            if evaluate_policies:
                print(f"[eval] {name}", flush=True)
                write_row(output, {"name": name, "variant": variant,
                                   "iteration": iteration, "refit_steps": milestone,
                                   "fit_s": fit_s, **evaluate(policy, reference)})
            else:
                with (output / "fit_progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"name": name, "variant": variant,
                                             "iteration": iteration,
                                             "refit_steps": milestone,
                                             "fit_s": fit_s}) + "\n")
                print(f"[saved] {name} fit_s={fit_s:.2f}", flush=True)
        del policy, optimizers, models
        torch.cuda.empty_cache()
    print("[complete] " + str(output / "results.jsonl"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=("warm", "fresh"),
                        default=("warm", "fresh"))
    parser.add_argument("--milestones", nargs="+", type=int,
                        default=MILESTONES)
    parser.add_argument("--fit-only", action="store_true",
                        help="Save policies and fit timings; defer exact evaluations")
    args = parser.parse_args()
    if not args.milestones or sorted(set(args.milestones)) != args.milestones:
        parser.error("Milestones must be positive, increasing and unique")
    if args.milestones[0] <= 0:
        parser.error("Milestones must be positive")
    run(args.checkpoint.resolve(), args.output.resolve(), tuple(args.variants),
        tuple(args.milestones), evaluate_policies=not args.fit_only)
