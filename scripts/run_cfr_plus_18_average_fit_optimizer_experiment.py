#!/usr/bin/env python3
"""Run the staged 18-claim offline average optimizer/objective experiment.

The source checkpoints are read-only. Fits are performed on fresh copies of
their strategy networks, optimizers and reservoirs; a small state file at each
milestone makes the sweep restartable without repeating completed steps.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import numpy as np
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from run_cfr_plus_18_offline_average_fitting import (
    SEED, exact_average, evaluate, neural_policy,
)
from liars_poker.algo.deep_cfr import DeviceReservoirBuffer
from liars_poker.serialization import save_policy
from liars_poker.policies.neural import InfosetEncoder

SOURCES = {
    "0030m": ("artifacts/cfr_plus_18_offline_average_study/exact4096_0030m_checkpoint.pt", 908),
    "0045m": ("artifacts/cfr_plus_18_offline_average_study/exact4096_0045m_checkpoint.pt", 1424),
    "0120m": ("artifacts/cfr_plus_18_offline_average_study/exact4096_0120m_checkpoint.pt", 3988),
}
MILESTONES = (250, 500, 1000, 2000, 5000)
COSINE_MILESTONES = (250, 500, 1000, 2000, 4600, 4800, 5000)
ARMS = {
    "O1_cosine_b1024_ce": (1e-3, 1e-5, 1024, "cosine", "cross_entropy"),
    "O2_lr1e-4_b1024_ce": (1e-4, 1e-4, 1024, "constant", "cross_entropy"),
    "O3_lr1e-3_b16384_ce": (1e-3, 1e-3, 16384, "constant", "cross_entropy"),
    "O4_cosine_b16384_ce": (1e-3, 1e-5, 16384, "cosine", "cross_entropy"),
}
O0 = "O0_lr1e-3_b1024_ce"
O0_CONFIG = (1e-3, 1e-3, 1024, "constant", "cross_entropy")
SCHEDULE_ARMS = {
    "W500": ("warm", 1e-3, 1e-5, 500),
    "W1k": ("warm", 1e-3, 1e-5, 1000),
    "W2k": ("warm", 1e-3, 1e-5, 2000),
    "L1k": ("warm", 3e-4, 1e-6, 1000),
    "L5k": ("warm", 3e-4, 1e-6, 5000),
    "R5k": ("warm_reset", 1e-3, 1e-5, 5000),
    "R1k_constant": ("warm_reset", 1e-3, 1e-3, 1000),
    "F40k": ("fresh", 1e-3, 1e-5, 40000),
    "F80k": ("fresh", 1e-3, 1e-5, 80000),
    "X": ("exact", 1e-3, 1e-5, 40000),
}


class ExactAverageSampler:
    """Draw (history, physical hand) rows by chance and own reach."""

    def __init__(self, policy, seed: int, device: str = "cuda"):
        self.policy = policy
        self.encoder = InfosetEncoder(policy.spec)
        self.hand_features = self.encoder.encode_hands(policy.hands, ())
        self.history_bits = np.arange(policy.k, dtype=np.int64)
        self.n_hands = len(policy.hands)
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.cdfs = []
        for pid, reach in enumerate((policy.L_pid0, policy.L_pid1)):
            active = ((policy.popcount & 1) == pid) & (policy.legal_counts > 0)
            # Physical hands have equal chance. Their probabilities therefore
            # cancel when the distribution is normalized.
            weights = np.where(active[:, None], reach, 0.0).ravel().astype(np.float64)
            cdf = np.cumsum(weights)
            if cdf[-1] <= 0:
                raise ValueError("Exact average has no reachable infosets")
            self.cdfs.append(cdf)

    def sample(self, pid: int, batch: int):
        cdf = self.cdfs[pid]
        positions = np.searchsorted(cdf, self.rng.random(batch) * cdf[-1])
        hids, hands = divmod(positions, self.n_hands)
        x = self.hand_features[hands].copy()
        x[:, self.policy.spec.ranks:] = ((hids[:, None] >> self.history_bits) & 1)
        return (torch.from_numpy(x).to(self.device),
                torch.from_numpy(self.policy.S[hids, hands].copy()).to(self.device),
                torch.from_numpy(self.policy.legal_mask[hids].copy()).to(self.device),
                torch.ones(batch, device=self.device))


def whole_reservoir_loss(model, buffer, batch_size: int = 16384) -> float:
    """Deterministic weighted cross-entropy over every retained row."""
    numerator = denominator = 0.0
    with torch.inference_mode():
        for start in range(0, buffer.size, batch_size):
            end = min(buffer.size, start + batch_size)
            x = buffer.features[start:end]
            y = buffer.targets[start:end]
            mask = buffer.legal_masks[start:end]
            weight = buffer.weights[start:end]
            logits = model(x).masked_fill(~mask, -1e9)
            ce = -(y * torch.log_softmax(logits, dim=1)).sum(dim=1)
            numerator += float((ce * weight).sum().item())
            denominator += float(weight.sum().item())
    return numerator / denominator if denominator else float("nan")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def arm_dir(root: Path, checkpoint: str, arm: str) -> Path:
    return root / checkpoint / arm


def configure_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def fit_one_arm(checkpoint: str, arm: str, config: tuple, output: Path,
                source_path: Path, *, fit_seed: int, steps: int,
                milestones: tuple[int, ...], loss_name: str,
                variant: str = "warm", evaluate_each: bool = True) -> list[dict]:
    lr_start, lr_end, batch_size, schedule, _ = config
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "resume_state.pt"
    result_path = output / "results.jsonl"
    completed = set()
    if result_path.exists():
        completed = {json.loads(line)["refit_steps"] for line in result_path.read_text(
            encoding="utf-8").splitlines() if line.strip()}
    state = torch.load(source_path, map_location="cpu", weights_only=False)
    spec = state["spec"]
    from liars_poker.core import GameSpec
    game_spec = GameSpec(**{**spec, "claim_kinds": tuple(spec["claim_kinds"])})
    iteration = int(state["iteration"])
    reference = exact_average(game_spec, state["exact_average_observer"])
    buffers = [DeviceReservoirBuffer.from_state_dict(b, device="cuda")
               for b in state["strategy_buffers"]]
    exact_sampler = ExactAverageSampler(reference, fit_seed) if variant == "exact" else None

    resume = None
    if state_path.exists():
        resume = torch.load(state_path, map_location="cuda", weights_only=False)
        # A stop can land after the milestone state is durable but before its
        # JSONL row is appended. Recover that row before continuing.
        if resume.get("last_row") is not None:
            known = {r.get("refit_steps") for r in read_rows(result_path)}
            if resume["last_row"].get("refit_steps") not in known:
                append_jsonl(result_path, resume["last_row"])
    torch.manual_seed(fit_seed)
    torch.cuda.manual_seed_all(fit_seed)
    if resume:
        policy = neural_policy(game_spec, resume["models"], "cuda")
        fitted = int(resume["fitted"])
        optimizers = [torch.optim.Adam(m.parameters(), lr=lr_start)
                      for m in (policy.model_p1, policy.model_p2)]
        for opt, saved in zip(optimizers, resume["optimizers"]):
            opt.load_state_dict(saved)
    else:
        initial = state["strategy_nets"] if variant in {"warm", "warm_reset"} else None
        policy = neural_policy(game_spec, initial, "cuda")
        fitted = 0
        optimizers = [torch.optim.Adam(m.parameters(), lr=lr_start)
                      for m in (policy.model_p1, policy.model_p2)]
        if variant == "warm":
            for opt, saved in zip(optimizers, state["strategy_optimizers"]):
                opt.load_state_dict(saved)

    if resume and resume.get("torch_rng_state") is not None:
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(resume["cuda_rng_states"])
        if exact_sampler is not None:
            exact_sampler.rng.bit_generator.state = resume["exact_rng_state"]

    models = (policy.model_p1, policy.model_p2)
    all_rows = []
    print(f"[arm] {checkpoint}/{arm}/{variant} seed={fit_seed} batch={batch_size} "
          f"loss={loss_name} fitted={fitted} cuda_free_gib="
          f"{torch.cuda.mem_get_info()[0]/2**30:.2f}", flush=True)
    for target_step in milestones:
        if target_step <= fitted:
            continue
        if target_step > steps:
            continue
        start_fit = time.perf_counter()
        for pid, (model, optimizer, buffer) in enumerate(zip(models, optimizers, buffers)):
            model.train()
            for step in range(fitted + 1, target_step + 1):
                if schedule == "cosine":
                    fraction = (step - 1) / max(steps - 1, 1)
                    lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * fraction))
                else:
                    lr = lr_start
                configure_optimizer_lr(optimizer, lr)
                x, y, mask, weight = (exact_sampler.sample(pid, batch_size)
                                      if exact_sampler is not None else buffer.sample(batch_size))
                weight = weight / weight.mean().clamp_min(1e-8)
                logits = model(x).masked_fill(~mask, -1e9)
                probs = torch.softmax(logits, dim=1)
                if loss_name == "prob_mse":
                    per_sample = ((probs - y).square() * mask).sum(dim=1)
                else:
                    per_sample = -(y * torch.log_softmax(logits, dim=1)).sum(dim=1)
                loss = (per_sample * weight).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step % 250 == 0 or step == target_step:
                    print(f"[fit] {checkpoint}/{arm}/{variant} p{pid+1} "
                          f"step={step}/{steps} loss={loss.item():.6g} lr={lr:.2g}",
                          flush=True)
            model.eval()
        fitted = target_step
        fit_elapsed = time.perf_counter() - start_fit
        policy_dir = output / f"{variant}_{target_step:05d}"
        save_policy(policy, str(policy_dir))
        row = {"checkpoint": checkpoint, "iteration": iteration,
               "measured_training_min": state["experiment_progress"]["measured_training_s"] / 60,
               "arm": arm, "variant": variant, "fit_seed": fit_seed,
               "learning_rate_start": lr_start, "learning_rate_end": lr_end,
               "schedule": schedule, "batch_size": batch_size, "loss": loss_name,
               "refit_steps": target_step, "fit_s_increment": fit_elapsed,
               "policy_dir": str(policy_dir)}
        row["whole_reservoir_ce"] = [
            whole_reservoir_loss(model, buffer)
            for model, buffer in zip(models, buffers)
        ]
        if evaluate_each:
            metrics = evaluate(policy, reference)
            row.update(metrics)
            print(f"[eval] {checkpoint}/{arm}/{variant}/{target_step} "
                  f"exploitability={metrics['exploitability']:.8f} "
                  f"eval_s={metrics['evaluation_s']:.1f}", flush=True)
        else:
            append_jsonl(output / "fit_progress.jsonl", row)
        resume_payload = {"fitted": fitted,
                    "models": [m.state_dict() for m in models],
                    "optimizers": [o.state_dict() for o in optimizers],
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all(),
                    "exact_rng_state": (exact_sampler.rng.bit_generator.state
                                        if exact_sampler is not None else None),
                    "last_row": row if evaluate_each else None}
        staged_state = state_path.with_suffix(".pt.tmp")
        torch.save(resume_payload, staged_state)
        os.replace(staged_state, state_path)
        if evaluate_each:
            known = {r.get("refit_steps") for r in read_rows(result_path)}
            if target_step not in known:
                append_jsonl(result_path, row)
        all_rows.append(row)
        del row
    del policy, models, optimizers, buffers, state, reference
    torch.cuda.empty_cache()
    gc.collect()
    return all_rows


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def best_stage1(root: Path) -> str:
    # Score each recipe by its final three evaluations at each checkpoint,
    # then average checkpoint scores equally.
    scores = {}
    for arm in (O0, *ARMS):
        per_source = []
        for checkpoint in SOURCES:
            rows = (existing_o0_rows(root, checkpoint) if arm == O0 else
                    [r for r in read_rows(arm_dir(root, checkpoint, arm) / "results.jsonl")
                     if r.get("variant") == "warm" and r.get("loss") == "cross_entropy"])
            rows.sort(key=lambda r: r["refit_steps"])
            if len(rows) < 3:
                continue
            per_source.append(sum(float(r["exploitability"]) for r in rows[-3:]) / 3)
        if len(per_source) == len(SOURCES):
            scores[arm] = sum(per_source) / len(per_source)
    if not scores:
        raise RuntimeError("Stage 1 results incomplete; cannot select an optimizer recipe")
    winner = min(scores, key=scores.get)
    (root / "stage1_selection.json").write_text(json.dumps(
        {"rule": "mean of each checkpoint's final three CE evaluations; equal checkpoint weighting",
         "scores": scores, "selected_arm": winner}, indent=2), encoding="utf-8")
    print(f"[selection] {winner}: {scores[winner]:.8f}; all={scores}", flush=True)
    return winner


def existing_o0_rows(root: Path, checkpoint: str) -> list[dict]:
    sweep = {"0030m": "0030m", "0045m": "0045m", "0120m": "0120m"}[checkpoint]
    path = ROOT / "artifacts/cfr_plus_18_offline_average_study" / f"step_sweep_{sweep}" / "results.jsonl"
    if not path.exists():
        path = ROOT / "docs/data/cfr_plus_18_offline_average_step_sweep" / f"step_sweep_{sweep}" / "results.jsonl"
    rows = read_rows(path)
    candidates = [r for r in rows if r.get("variant") == "warm" and
                  str(r.get("name", "")).startswith("warm_")]
    transformed = []
    for r in candidates:
        transformed.append({**r, "checkpoint": checkpoint, "arm": "O0_lr1e-3_b1024_ce",
                            "loss": "cross_entropy", "schedule": "constant",
                            "batch_size": 1024, "learning_rate_start": 1e-3,
                            "learning_rate_end": 1e-3, "fit_seed": SEED,
                            "refit_steps": int(r["refit_steps"]),
                            "exploitability": float(r["exploitability"])})
    return transformed


def run(root: Path, source_root: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU for fitting")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    manifest = {"seed": SEED, "sources": SOURCES, "milestones": MILESTONES,
                "arms": ARMS, "status": "running",
                "evaluation_mode": "inline, one CPU worker"}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Stage 1: use the existing O0 curves; fit the four non-baseline recipes.
    for checkpoint, (relative_source, expected_iteration) in SOURCES.items():
        source_path = source_root / relative_source
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        for arm, config in ARMS.items():
            fit_one_arm(checkpoint, arm, config, arm_dir(root, checkpoint, arm),
                        source_path, fit_seed=SEED + 1, steps=5000,
                        milestones=COSINE_MILESTONES if "cosine" in arm else MILESTONES,
                        loss_name="cross_entropy")
        # Add the prior O0 data as a normalized arm record for plotting/scoring.
        o0_dir = arm_dir(root, checkpoint, "O0_lr1e-3_b1024_ce")
        o0_dir.mkdir(parents=True, exist_ok=True)
        o0 = existing_o0_rows(root, checkpoint)
        if not o0:
            raise RuntimeError(f"Could not find existing O0 baseline results for {checkpoint}")
        dest = o0_dir / "results.jsonl"
        known = {r["refit_steps"] for r in read_rows(dest)}
        for row in o0:
            if row["refit_steps"] not in known:
                append_jsonl(dest, row)
        print(f"[source complete] {checkpoint} expected_iter={expected_iteration}", flush=True)

    selected = best_stage1(root)
    selected_config = O0_CONFIG if selected == O0 else ARMS[selected]

    # Stage 2: weighted probability-MSE under selected and baseline optimizers.
    mse_recipes = {
        f"M1_{selected}_prob_mse": (*selected_config[:4], "prob_mse"),
        "M2_O0_lr1e-3_b1024_prob_mse": (1e-3, 1e-3, 1024, "constant", "prob_mse"),
    }
    for checkpoint, (relative_source, _) in SOURCES.items():
        for arm, config in mse_recipes.items():
            fit_one_arm(checkpoint, arm, config, arm_dir(root, checkpoint, arm),
                        source_root / relative_source, fit_seed=SEED + 1,
                        steps=5000, milestones=MILESTONES, loss_name="prob_mse")

    # Choose the best complete Stage-1/Stage-2 recipe by the same final-three
    # rule. O0 CE is included as the measured reference.
    finalists = [O0, *ARMS.keys(), *mse_recipes.keys()]
    final_scores = {}
    for arm in finalists:
        per_source = []
        for checkpoint in SOURCES:
            rows = [r for r in read_rows(arm_dir(root, checkpoint, arm) / "results.jsonl")
                    if r.get("variant", "warm") == "warm"]
            if arm.startswith("O0_"):
                rows = existing_o0_rows(root, checkpoint)
            rows.sort(key=lambda r: r["refit_steps"])
            if len(rows) >= 3:
                per_source.append(sum(float(r["exploitability"]) for r in rows[-3:]) / 3)
        if len(per_source) == len(SOURCES):
            final_scores[arm] = sum(per_source) / len(per_source)
    best_final = min(final_scores, key=final_scores.get)
    (root / "final_selection.json").write_text(json.dumps(
        {"rule": "mean of each checkpoint's final three evaluations; equal checkpoint weighting",
         "scores": final_scores, "selected_recipe": best_final}, indent=2), encoding="utf-8")
    print(f"[final selection] {best_final}: {final_scores[best_final]:.8f}", flush=True)

    # Fresh-start check, 20k steps/player, selected best complete recipe.
    recipe_config = (O0_CONFIG if best_final == O0 else ARMS[best_final]
                     if best_final in ARMS else
                     (1e-3, 1e-3, 1024, "constant", "prob_mse")
                     if best_final == "M2_O0_lr1e-3_b1024_prob_mse"
                     else (*selected_config[:4], "prob_mse"))
    for checkpoint, (relative_source, _) in SOURCES.items():
        fit_one_arm(checkpoint, f"FRESH_{best_final}", recipe_config,
                    arm_dir(root, checkpoint, f"FRESH_{best_final}"),
                    source_root / relative_source, fit_seed=SEED + 500,
                    steps=20000, milestones=(5000, 10000, 20000),
                    loss_name=recipe_config[-1], variant="fresh")

    # Two additional fit seeds for O0 and the selected final recipe, final point.
    replicate_recipes = {
        O0: O0_CONFIG,
        best_final: recipe_config,
    }
    for checkpoint, (relative_source, _) in SOURCES.items():
        for arm, config in replicate_recipes.items():
            for replicate_seed in (SEED + 2, SEED + 3):
                rep_name = f"{arm}_seed{replicate_seed}"
                fit_one_arm(checkpoint, rep_name, config, arm_dir(root, checkpoint, rep_name),
                            source_root / relative_source, fit_seed=replicate_seed,
                            steps=5000, milestones=(5000,), loss_name=config[-1])

    manifest["status"] = "complete"
    manifest["selected_stage1_optimizer"] = selected
    manifest["selected_final_recipe"] = best_final
    manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_schedules(root: Path, source_root: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for average fitting")
    torch.set_num_threads(1)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": SOURCES, "arms": SCHEDULE_ARMS,
                "batch_size": 16384, "loss": "weighted cross entropy"}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Manifest mismatch; refusing to resume")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for checkpoint, (relative, expected_iteration) in SOURCES.items():
        source = source_root / relative
        if not source.exists():
            raise FileNotFoundError(source)
        for arm, (variant, start, end, steps) in SCHEDULE_ARMS.items():
            if arm == "R1k_constant":
                points = (250, 1000)
            elif variant in {"fresh", "exact"}:
                points = tuple(n for n in (10000, 20000, 40000, 80000) if n <= steps)
            else:
                points = tuple(sorted({max(1, round(steps * f)) for f in (0.25, 0.5, 0.75, 1)}))
            fit_one_arm(checkpoint, arm, (start, end, 16384,
                        "constant" if arm == "R1k_constant" else "cosine",
                        "cross_entropy"), arm_dir(root, checkpoint, arm),
                        source, fit_seed=SEED + 1, steps=steps,
                        milestones=points, loss_name="cross_entropy", variant=variant)
        print(f"[source done] {checkpoint} iteration={expected_iteration}", flush=True)

    # Replicate after selection, using final exact exploitability averaged
    # equally across checkpoints. This is deterministic on resumed runs.
    scores = {}
    for arm, (variant, _, _, steps) in SCHEDULE_ARMS.items():
        if arm in {"R1k_constant", "X"}:
            continue
        values = []
        for checkpoint in SOURCES:
            rows = read_rows(arm_dir(root, checkpoint, arm) / "results.jsonl")
            values.extend([r["exploitability"] for r in rows
                           if r.get("refit_steps") == steps])
        if len(values) == len(SOURCES):
            scores[arm] = sum(values) / len(values)
    winners = {}
    for group, variants in (("warm", {"warm", "warm_reset"}),
                            ("fresh", {"fresh"})):
        choices = {arm: score for arm, score in scores.items()
                   if SCHEDULE_ARMS[arm][0] in variants}
        winners[group] = min(choices, key=choices.get)
    (root / "selection.json").write_text(json.dumps(
        {"scores": scores, "winners": winners}, indent=2), encoding="utf-8")
    for checkpoint, (relative, _) in SOURCES.items():
        for winner in winners.values():
            variant, start, end, steps = SCHEDULE_ARMS[winner]
            for seed in (SEED + 2, SEED + 3):
                fit_one_arm(checkpoint, f"{winner}_seed{seed}",
                            (start, end, 16384, "cosine", "cross_entropy"),
                            arm_dir(root, checkpoint, f"{winner}_seed{seed}"),
                            source_root / relative, fit_seed=seed, steps=steps,
                            milestones=(steps,), loss_name="cross_entropy",
                            variant=variant)


def smoke_schedules(root: Path, source_root: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for average-fitting smoke test")
    checkpoint = "0030m"
    source = source_root / SOURCES[checkpoint][0]
    if not source.exists():
        raise FileNotFoundError(source)
    fit_one_arm(checkpoint, "W500_smoke", (1e-3, 1e-5, 16384, "cosine", "cross_entropy"),
                arm_dir(root, checkpoint, "W500_smoke"), source,
                fit_seed=SEED + 1, steps=25, milestones=(25,),
                loss_name="cross_entropy")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, default=ROOT)
    p.add_argument("--schedules", action="store_true",
                   help="Run the average-fit schedule and exact-target arms")
    p.add_argument("--smoke-schedules", action="store_true",
                   help="One 25-step warm fit and exact evaluation on the first checkpoint")
    a = p.parse_args()
    if a.schedules and a.smoke_schedules:
        p.error("Choose either --schedules or --smoke-schedules")
    action = smoke_schedules if a.smoke_schedules else run_schedules if a.schedules else run
    action(a.output_root.resolve(), a.source_root.resolve())
