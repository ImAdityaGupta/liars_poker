#!/usr/bin/env python3
"""Read-only checkpoint audit of exact, sampled, and fitted regret updates."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
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

from liars_poker.algo.cfr_plus_dense import CFRPlusDense
from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer
from liars_poker.algo.neural_cfr_plus_gpu import GPUDeepCFRPlusTraverser


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def available_ram_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("Cannot read MemAvailable")


def network_table(trainer: DeepCFRPlusTrainer, solver: CFRPlusDense, player: int) -> np.ndarray:
    """Evaluate regret network at every infoset belonging to one player."""
    hids = np.flatnonzero((solver.popcount & 1) == player)
    hands = trainer.encoder.encode_hands(solver.hands, ())
    rank_dim = trainer.spec.ranks
    bits = np.arange(trainer.encoder.k, dtype=np.int64)
    result = np.zeros((solver.H, solver.n_hands, solver.A), dtype=np.float32)
    model = trainer.regret_nets[player]
    with torch.inference_mode():
        for start in range(0, len(hids), 256):
            batch_hids = hids[start:start + 256]
            features = np.empty((len(batch_hids), solver.n_hands, trainer.encoder.input_dim), np.float32)
            features[:, :, :rank_dim] = hands[None, :, :rank_dim]
            features[:, :, rank_dim:] = ((batch_hids[:, None] >> bits) & 1)[:, None, :]
            values = model(torch.from_numpy(features.reshape(-1, trainer.encoder.input_dim)))
            result[batch_hids] = torch.relu(values).reshape(len(batch_hids), solver.n_hands, solver.A).numpy()
    result *= solver.legal_mask[:, None, :]
    return result


def exact_tables(trainer: DeepCFRPlusTrainer, player: int) -> tuple[CFRPlusDense, np.ndarray, np.ndarray, np.ndarray]:
    """Exact g and qg under the frozen neural current strategy."""
    dense = trainer.current_policy_dense()
    solver = CFRPlusDense(trainer.spec)
    solver.S[:] = dense.S
    del dense
    solver._recompute_likelihoods()
    q = np.zeros((solver.H, solver.n_hands), np.float32)
    g = np.zeros((solver.H, solver.n_hands, solver.A), np.float32)
    qg = np.zeros_like(g)
    from liars_poker.algo.br_exact_dense_to_dense import adjustment_factor
    from liars_poker.core import generate_deck
    deck_size = len(generate_deck(trainer.spec))
    own_deals = math.comb(deck_size, trainer.spec.hand_size)
    opp_deals = math.comb(deck_size - trainer.spec.hand_size, trainer.spec.hand_size)
    hand_prob = np.asarray([adjustment_factor(trainer.spec, (), h) / own_deals
                            for h in solver.hands], dtype=np.float64)
    row_scale = hand_prob / opp_deals
    A = solver.A0 if player == 0 else solver.A1

    def capture(hid: int, opponent_reach: np.ndarray, raw: np.ndarray) -> np.ndarray:
        cols = list(solver.legal_cols[hid])
        q_dense = A @ opponent_reach
        q[hid] = row_scale * q_dense
        g[hid][:, cols] = np.divide(raw, q_dense[None, :],
                                    out=np.zeros_like(raw), where=q_dense[None, :] > 0).T
        qg[hid][:, cols] = raw.T * row_scale[:, None]
        return np.zeros_like(raw)

    solver._update_player(player, 0.0, increment_transform=capture)
    return solver, q, g, qg


def sampled_target(trainer: DeepCFRPlusTrainer, solver: CFRPlusDense,
                   old: np.ndarray, player: int, roots: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Run one genuine traversal/update on the loaded copy; keep the source checkpoint untouched."""
    trainer.iteration += 1
    buffer = trainer.regret_buffers[player]
    buffer.clear()
    traverser = GPUDeepCFRPlusTraverser(trainer)
    for start in range(0, roots, trainer.traversal_batch_size):
        traverser.run_traversals(player, min(trainer.traversal_batch_size, roots - start))
    n = buffer.size
    if buffer.seen != n:
        raise RuntimeError("Regret buffer overflowed; sampled audit would be incomplete")
    if trainer.regret_increment_reach_mode != "none":
        raise ValueError("This audit requires the conditional, unweighted sampled target")
    trainer._aggregate_regret_targets(buffer)

    features = buffer.features[:n].numpy()
    ranks = trainer.spec.ranks
    hand_base = (3 ** np.arange(ranks)).astype(np.int64)
    hand_codes = trainer.encoder.encode_hands(solver.hands, ())[:, :ranks].astype(np.int64) @ hand_base
    lookup = np.full(3 ** ranks, -1, dtype=np.int32)
    lookup[hand_codes] = np.arange(solver.n_hands, dtype=np.int32)
    codes = features[:, :ranks].astype(np.int64) @ hand_base
    hands = lookup[codes]
    if np.any(hands < 0):
        raise AssertionError("Sampled hand encoding is absent from exact solver")
    bits = (1 << np.arange(trainer.encoder.k)).astype(np.int64)
    hids = features[:, ranks:].astype(np.int64) @ bits
    if np.any((solver.popcount[hids] & 1) != player):
        raise AssertionError("Sampled history/player mapping failed")
    keys = hids * solver.n_hands + hands
    unique, first, counts = np.unique(keys, return_index=True, return_counts=True)
    sampled = old.copy()  # No sampled target exists at unvisited infosets.
    sampled[hids[first], hands[first]] = buffer.targets[first].numpy()
    visits = np.zeros((solver.H, solver.n_hands), dtype=np.uint16)
    visits[hids[first], hands[first]] = counts.astype(np.uint16)
    loss = trainer._train_model(trainer.regret_nets[player], trainer.regret_optimizers[player],
                                buffer, trainer.regret_train_steps, strategy_loss=False)
    return sampled, visits, n, loss


def independent_fit_errors(checkpoint: Path, fitted: DeepCFRPlusTrainer,
                           player: int, roots: int) -> dict:
    """Score the fitted model on its update rows and a second frozen-policy traversal."""
    baseline = DeepCFRPlusTrainer.load_checkpoint(checkpoint, device="cpu")
    baseline.iteration += 1
    baseline.rng.seed(baseline.seed + 911_003 + baseline.iteration)
    torch.manual_seed(baseline.seed + 911_003 + baseline.iteration)
    held = baseline.regret_buffers[player]
    held.clear()
    traverser = GPUDeepCFRPlusTraverser(baseline)
    for start in range(0, roots, baseline.traversal_batch_size):
        traverser.run_traversals(player, min(baseline.traversal_batch_size, roots - start))
    if held.size != held.seen:
        raise RuntimeError("Held-out traversal overflowed its regret buffer")
    baseline._aggregate_regret_targets(held)

    def mse(buffer) -> float:
        if not buffer.size:
            return float("nan")
        rng = np.random.default_rng(17031 + fitted.iteration + player)
        indices = torch.as_tensor(rng.choice(buffer.size,
                                  min(8192, buffer.size), replace=False), dtype=torch.long)
        with torch.inference_mode():
            x = torch.as_tensor(buffer.features[:buffer.size]).index_select(0, indices)
            y = torch.as_tensor(buffer.targets[:buffer.size]).index_select(0, indices)
            mask = torch.as_tensor(buffer.legal_masks[:buffer.size]).index_select(0, indices)
            weight = torch.as_tensor(buffer.weights[:buffer.size]).index_select(0, indices)
            pred = fitted.regret_nets[player](x)
            row = (((pred - y).square() * mask).sum(dim=1)
                   / mask.sum(dim=1).clamp_min(1))
            return float((row * weight).sum() / weight.sum().clamp_min(1e-12))

    return {"training_mse": mse(fitted.regret_buffers[player]),
            "held_out_mse": mse(held), "held_out_records": held.size}


def regret_match(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    positive = np.maximum(values, 0.0) * mask
    totals = positive.sum(axis=1, keepdims=True)
    uniform = mask / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    return np.where(totals > 0, positive / np.maximum(totals, 1e-30), uniform)


def summarize(solver: CFRPlusDense, player: int, roots: int, q: np.ndarray,
              visits: np.ndarray, tables: dict[str, np.ndarray],
              infoset_path: Path | None = None) -> list[dict]:
    hids = np.flatnonzero(((solver.popcount & 1) == player) & (solver.legal_counts > 1))
    all_hids = np.repeat(hids, solver.n_hands)
    all_hands = np.tile(np.arange(solver.n_hands), len(hids))
    possible = q[all_hids, all_hands] > 0
    all_hids, all_hands = all_hids[possible], all_hands[possible]
    pairs = (("old", "exact_g"), ("exact_g", "old"),
             ("exact_g", "sampled"),
             ("sampled", "fitted"), ("exact_g", "fitted"),
             ("exact_g", "exact_qg"))
    bins = (("<0.1", 0, .1), ("0.1-1", .1, 1), ("1-10", 1, 10),
            (">=10", 10, float("inf")), ("all", 0, float("inf")))
    accum: dict[tuple[str, str, str], dict] = {}
    detailed: dict[str, list[np.ndarray]] = {"hids": [], "hands": [], "q": [], "visits": []}
    if infoset_path is not None:
        for left, right in pairs:
            for metric in ("tv", "kl", "regret_mse", "support_mismatch"):
                detailed[f"{left}_vs_{right}_{metric}"] = []
    for start in range(0, len(all_hids), 16_384):
        hh, hands = all_hids[start:start + 16_384], all_hands[start:start + 16_384]
        reach = q[hh, hands].astype(np.float64)
        if np.any(reach >= 1):
            raise ValueError("1 / -log(q) requires 0 < q < 1 for every included infoset")
        log_reach_weight = 1.0 / -np.log(reach)
        count = visits[hh, hands]
        if infoset_path is not None:
            detailed["hids"].append(hh.astype(np.int32))
            detailed["hands"].append(hands.astype(np.int16))
            detailed["q"].append(reach.astype(np.float32))
            detailed["visits"].append(count.astype(np.uint16))
        mask = solver.legal_mask[hh]
        policy = {name: regret_match(table[hh, hands], mask)
                  for name, table in tables.items()}
        for left, right in pairs:
            p, r = policy[left], policy[right]
            eps = 1e-5
            legal_count = mask.sum(axis=1, keepdims=True)
            ps = (p + eps * mask) / (1 + eps * legal_count)
            rs = (r + eps * mask) / (1 + eps * legal_count)
            ratio = np.divide(ps, rs, out=np.ones_like(ps), where=mask)
            kl = (ps * np.log(ratio)).sum(axis=1)
            tv = .5 * np.abs(p - r).sum(axis=1)
            support = (((p > 1e-8) != (r > 1e-8)) & mask).any(axis=1).astype(float)
            delta = ((tables[left][hh, hands] - tables[right][hh, hands]) ** 2 * mask).sum(axis=1) / legal_count[:, 0]
            if infoset_path is not None:
                stem = f"{left}_vs_{right}"
                detailed[f"{stem}_tv"].append(tv.astype(np.float32))
                detailed[f"{stem}_kl"].append(kl.astype(np.float32))
                detailed[f"{stem}_regret_mse"].append(delta.astype(np.float32))
                detailed[f"{stem}_support_mismatch"].append(support.astype(np.uint8))
            for scope in ("all", "visited", "unvisited"):
                selected = np.ones(len(hh), bool) if scope == "all" else ((count > 0) if scope == "visited" else (count == 0))
                for label, lo, hi in bins:
                    take = selected & (roots * reach >= lo) & (roots * reach < hi)
                    if not np.any(take):
                        continue
                    key = (scope, label, f"{left}_vs_{right}")
                    s = accum.setdefault(key, dict(n=0, q_sum=0., log_reach_weight_sum=0.,
                                                   kl=0., qkl=0., tv=0., qtv=0.,
                                                   log_reach_weighted_tv_sum=0.,
                                                   support=0., mse=0., qmse=0.))
                    w = reach[take]
                    log_w = log_reach_weight[take]
                    s["n"] += int(take.sum())
                    s["q_sum"] += float(w.sum())
                    s["log_reach_weight_sum"] += float(log_w.sum())
                    s["kl"] += float(kl[take].sum())
                    s["qkl"] += float((kl[take] * w).sum())
                    s["tv"] += float(tv[take].sum())
                    s["qtv"] += float((tv[take] * w).sum())
                    s["log_reach_weighted_tv_sum"] += float((tv[take] * log_w).sum())
                    s["support"] += float(support[take].sum())
                    s["mse"] += float(delta[take].sum())
                    s["qmse"] += float((delta[take] * w).sum())
    rows = []
    for (scope, label, pair), s in accum.items():
        rows.append({"scope": scope, "expected_visits_bin": label, "pair": pair,
                     "infosets": s["n"], "reach_weight_sum": s["q_sum"],
                     "mean_kl_nats": s["kl"] / s["n"],
                     "reach_weighted_kl_nats": s["qkl"] / s["q_sum"],
                     "mean_tv": s["tv"] / s["n"],
                     "reach_weighted_tv": s["qtv"] / s["q_sum"],
                     "inv_neg_log_reach_weighted_tv": (
                         s["log_reach_weighted_tv_sum"] / s["log_reach_weight_sum"]),
                     "support_mismatch_fraction": s["support"] / s["n"],
                     "regret_rmse": math.sqrt(s["mse"] / s["n"]),
                     "reach_weighted_regret_rmse": math.sqrt(s["qmse"] / s["q_sum"])})
    if not rows or any(not math.isfinite(float(row["mean_kl_nats"])) for row in rows):
        raise RuntimeError("Audit produced no finite KL metrics")
    if infoset_path is not None:
        np.savez_compressed(infoset_path, **{
            key: np.concatenate(parts) for key, parts in detailed.items()
        })
    return sorted(rows, key=lambda row: (row["scope"], row["expected_visits_bin"], row["pair"]))


def plot(rows: list[dict], path: Path) -> None:
    order = ["<0.1", "0.1-1", "1-10", ">=10"]
    pairs = (("exact_g_vs_sampled", "Exact → sampled", "#167a65"),
             ("sampled_vs_fitted", "Sampled → fitted", "#c45b2d"),
             ("exact_g_vs_fitted", "Exact → fitted", "#404d83"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for pair, label, color in pairs:
        by_bin = {r["expected_visits_bin"]: r for r in rows
                  if r["scope"] == "visited" and r["pair"] == pair}
        for ax, metric in zip(axes, ("mean_kl_nats", "mean_tv")):
            ax.plot([b for b in order if b in by_bin],
                    [max(by_bin[b][metric], 1e-9) for b in order if b in by_bin],
                    marker="o", label=label, color=color)
    axes[0].set_ylabel("Smoothed KL divergence (nats)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Total-variation distance")
    for ax in axes:
        ax.set_xlabel("Expected sampled visits Kq(I) · visited infosets")
        ax.grid(alpha=.25)
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--roots", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--save-infoset-metrics", action="store_true",
                        help="Store per-infoset reach, visits, TV, KL and regret error in compressed NPZ")
    parser.add_argument("--held-out", action="store_true",
                        help="Run an independent frozen-policy traversal for fit error")
    args = parser.parse_args()
    if args.roots <= 0 or not 1 <= args.threads <= 4:
        parser.error("roots must be positive and threads must be 1–4")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "summary.json").exists():
        parser.error("audit already completed in this output directory")
    if available_ram_gib() < 24 or shutil.disk_usage(args.output_dir).free < 10 * 1024**3:
        parser.error("insufficient VM RAM or disk headroom for the isolated audit")
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    print("loading", args.checkpoint, flush=True)
    trainer = DeepCFRPlusTrainer.load_checkpoint(args.checkpoint, device="cpu")
    if (trainer.regret_target_mode != "aggregate_then_clip"
            or trainer.regret_increment_reach_mode != "none"
            or trainer.traversal_backend != "gpu_native"):
        raise ValueError("Expected the 18-claim conditional aggregate-then-clip CPU checkpoint")
    print("building exact reference", flush=True)
    solver, q, g, qg = exact_tables(trainer, args.player)
    old = network_table(trainer, solver, args.player)
    check_hids = np.flatnonzero((solver.popcount & 1) == args.player)[::max(1, solver.H // 1000)]
    matched = regret_match(old[check_hids].reshape(-1, solver.A),
                           np.broadcast_to(solver.legal_mask[check_hids, None, :],
                                           old[check_hids].shape).reshape(-1, solver.A))
    np.testing.assert_allclose(matched.reshape(-1, solver.n_hands, solver.A),
                               solver.S[check_hids], atol=2e-6, rtol=2e-6)
    t = trainer.iteration + 1
    prior = old if trainer.regret_accumulation_mode == "cumulative" else (t - 1) / t * old
    increment = 1.0 if trainer.regret_accumulation_mode == "cumulative" else 1.0 / t
    legal = solver.legal_mask[:, None, :]
    exact_g = np.maximum(prior + increment * g, 0) * legal
    exact_qg = np.maximum(prior + increment * qg, 0) * legal
    del g, qg
    print("sampling and fitting one player update", flush=True)
    sampled, visits, n_records, loss = sampled_target(trainer, solver, old, args.player, args.roots)
    held_out = (independent_fit_errors(args.checkpoint, trainer, args.player, args.roots)
                if args.held_out else {})
    fitted = network_table(trainer, solver, args.player)
    tables = {"old": old, "exact_g": exact_g, "exact_qg": exact_qg,
              "sampled": sampled, "fitted": fitted}
    print("summarizing all legal infosets", flush=True)
    rows = summarize(solver, args.player, args.roots, q, visits, tables,
                     args.output_dir / "infoset_metrics.npz" if args.save_infoset_metrics else None)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output_dir / "kl_by_reach.png")
    summary = {"checkpoint": str(args.checkpoint.resolve()),
               "checkpoint_sha256": sha256_file(args.checkpoint),
               "utc": datetime.now(timezone.utc).isoformat(), "player": args.player,
               "source_iteration": t - 1, "audited_iteration": t, "roots": args.roots,
               "target_mode": trainer.regret_target_mode,
               "accumulation_mode": trainer.regret_accumulation_mode,
               "sampled_records": n_records, "visited_infosets": int((visits > 0).sum()),
               "regret_fit_loss": loss, "elapsed_s": time.perf_counter() - start,
               **held_out,
               "available_ram_gib_after": available_ram_gib(),
               "disk_free_gib_after": shutil.disk_usage(args.output_dir).free / 1024**3}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
