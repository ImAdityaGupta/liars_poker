#!/usr/bin/env python3
"""Train neural CFR+ on CPU or CUDA; refit frozen average reservoirs off path.

Commands: train, fit, eval. A trainer checkpoint is committed before each frozen
reservoir becomes visible to the fit worker. The fit worker never reads live
reservoir tensors. Training time excludes snapshots, fitting and evaluation.
"""

from __future__ import annotations

import argparse
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from liars_poker.algo.deep_cfr import DeviceReservoirBuffer
from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer
from liars_poker.core import GameSpec
from liars_poker.policies.neural import NeuralPolicy
from liars_poker.policies.neural_regret import NeuralRegretMatchingPolicy
from liars_poker.serialization import save_policy
from scripts.run_cfr_plus_18_target_order_cpu_overnight import make_trainer, SPEC

ARMS = {"neural_o4_k1024": 1024, "neural_o4_k4096": 4096}
C_ARMS = {
    "c0": (1e-3, 1024, "constant"),
    "c_batch": (1e-3, 8192, "constant"),
    "c_anneal": (1e-3, 1024, "cosine"),
    "c_low": (3e-4, 1024, "constant"),
}
ARMS.update({name: 4096 for name in C_ARMS})
FIT_STEPS = 5000
FIT_BATCH = 16384
SNAPSHOT_MIN = 15.0
SEED = 17


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, sort_keys=True) + "\n")
        file.flush()


def checkpoint(trainer: DeepCFRPlusTrainer, path: Path, progress: dict) -> None:
    old_size = path.stat().st_size if path.exists() else 0
    if shutil.disk_usage(path.parent).free < old_size + 3 * 1024**3:
        raise OSError(f"Insufficient disk space to checkpoint {path}")
    state = trainer.checkpoint_dict()
    state["experiment_progress"] = progress
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def progress_of(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["experiment_progress"]


def expected_manifest(arm: str, snapshot_min: float) -> dict:
    result = {
        "arm": arm, "seed": SEED, "roots_per_player": ARMS[arm],
        "regret_target_mode": "aggregate_then_clip",
        "regret_positive_weight": 0.0,
        "regret_accumulation_mode": "cumulative",
        "regret_increment_reach_mode": "none",
        "regret_train_steps": 24, "strategy_train_steps": 6,
        "regret_buffer_capacity": 4_000_000,
        "strategy_buffer_capacity": 2_000_000,
        "traversal_backend": "gpu_native on CUDA" if arm in C_ARMS else "gpu_native on CPU",
        "traversal_batch_size": 512, "snapshot_minutes": snapshot_min,
        "average_refit": "O4: warm Adam, cosine 1e-3 to 1e-5, batch 16384, weighted CE, 5000 steps per player",
    }
    if arm in C_ARMS:
        lr, batch, schedule = C_ARMS[arm]
        result.update({"regret_fit_learning_rate": lr, "regret_batch_size": batch,
                       "regret_fit_schedule": schedule})
    return result


def freeze_input(trainer: DeepCFRPlusTrainer, path: Path, progress: dict) -> None:
    state = {
        "spec": json.loads(trainer.spec.to_json()),
        "strategy_hidden_sizes": list(trainer.strategy_hidden_sizes),
        "iteration": trainer.iteration,
        "measured_training_min": progress["measured_training_s"] / 60.0,
        "strategy_nets": [net.state_dict() for net in trainer.strategy_nets],
        "strategy_optimizers": [opt.state_dict() for opt in trainer.strategy_optimizers],
        "strategy_buffers": [buffer.state_dict() for buffer in trainer.strategy_buffers],
    }
    torch.save(state, path)


def publish_online(trainer: DeepCFRPlusTrainer, directory: Path, progress: dict) -> None:
    final = directory / "online_policy"
    ready = directory / "ONLINE_READY.json"
    if ready.exists():
        return
    tmp = directory / "online_policy.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    if not final.exists():
        save_policy(trainer.average_policy(), str(tmp))
        os.replace(tmp, final)
    atomic_json(ready, {
        "snapshot": directory.name, "policy_kind": "online",
        "policy_dir": str(final), "iteration": trainer.iteration,
        "measured_training_min": progress["measured_training_s"] / 60,
        "utc": utc(),
    })


def publish_current(trainer: DeepCFRPlusTrainer, directory: Path, progress: dict) -> None:
    ready = directory / "CURRENT_READY.json"
    if ready.exists():
        return
    final = directory / "current_policy"
    if not final.exists():
        tmp = directory / "current_policy.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        policy = NeuralRegretMatchingPolicy.from_models(
            trainer.spec, trainer.regret_nets,
            hidden_sizes=trainer.regret_hidden_sizes, device="cpu")
        save_policy(policy, str(tmp))
        os.replace(tmp, final)
    atomic_json(ready, {"snapshot": directory.name, "policy_kind": "current",
                        "policy_dir": str(final), "iteration": trainer.iteration,
                        "measured_training_min": progress["measured_training_s"] / 60,
                        "utc": utc()})


def finish_pending(trainer: DeepCFRPlusTrainer, run: Path, progress: dict) -> None:
    label = progress.get("pending_snapshot")
    if label is None:
        return
    directory = run / "policy_snapshots" / label
    staged = directory / "FIT_INPUT.pt.tmp"
    final = directory / "FIT_INPUT.pt"
    if (directory / "READY.json").exists():
        publish_online(trainer, directory, progress)
        if progress.get("save_current"):
            publish_current(trainer, directory, progress)
        return
    if not final.exists():
        if not staged.exists():
            raise RuntimeError(f"Checkpoint committed but frozen input missing: {directory}")
        os.replace(staged, final)
    publish_online(trainer, directory, progress)
    if progress.get("save_current"):
        publish_current(trainer, directory, progress)


def pending_inputs(run: Path) -> int:
    return sum(1 for _ in (run / "policy_snapshots").glob("*/FIT_INPUT.pt"))


def snapshot_label(seconds: float, interval_minutes: float) -> str:
    return (f"{int(round(seconds)):05d}s" if interval_minutes < 1
            else f"{int(round(seconds / 60)):04d}m")


def queue_audits(run: Path, checkpoint_path: Path, iteration: int,
                 *, final: bool = False) -> None:
    for milestone in (2000, 6000, "final"):
        if milestone == "final" and not final:
            continue
        if isinstance(milestone, int) and iteration < milestone:
            continue
        directory = run / "audits" / str(milestone)
        if (directory / "DONE.json").exists() or (directory / "input.pt").exists():
            continue
        if shutil.disk_usage(run).free < checkpoint_path.stat().st_size + 3 * 1024**3:
            print(f"[audit deferred] {run.name} {milestone}: insufficient disk", flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        staged = directory / "input.pt.tmp"
        shutil.copyfile(checkpoint_path, staged)
        os.replace(staged, directory / "input.pt")
        print(f"[audit queued] {run.name} {milestone} at iter={iteration}", flush=True)


def run_train(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    run = args.output_root.resolve() / args.arm
    run.mkdir(parents=True, exist_ok=True)
    manifest_path = run / "manifest.json"
    expected = expected_manifest(args.arm, args.snapshot_minutes)
    ckpt = run / "latest_checkpoint.pt"
    if ckpt.exists():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(actual.get(k) != v for k, v in expected.items()):
            raise ValueError("Manifest mismatch; refusing to resume")
        progress = progress_of(ckpt)
        trainer = DeepCFRPlusTrainer.load_checkpoint(
            ckpt, device="cuda" if args.arm in C_ARMS else "cpu")
        if trainer.iteration != progress["iteration"]:
            raise ValueError("Checkpoint progress disagrees with trainer iteration")
        finish_pending(trainer, run, progress)
        print(f"[resume] {args.arm} iter={trainer.iteration} train={progress['measured_training_s']/60:.2f}m", flush=True)
    else:
        if manifest_path.exists():
            raise RuntimeError("Manifest exists without checkpoint; refusing to overwrite")
        if args.arm in C_ARMS:
            lr, batch, schedule = C_ARMS[args.arm]
            trainer = DeepCFRPlusTrainer(
                SPEC,
                device="cuda", seed=SEED,
                regret_hidden_sizes=(512, 512), strategy_hidden_sizes=(256, 256),
                learning_rate=1e-3, batch_size=1024,
                regret_batch_size=batch, regret_fit_schedule=schedule,
                regret_fit_learning_rate=lr,
                regret_train_steps=24, strategy_train_steps=6,
                regret_buffer_capacity=4_000_000,
                strategy_buffer_capacity=2_000_000,
                regret_target_mode="aggregate_then_clip",
                regret_accumulation_mode="cumulative",
                regret_positive_weight=0.0, strategy_weighting="linear",
                traversal_backend="gpu_native", traversal_batch_size=512,
                device_replay=False, fused_optimizer=False, validation_fraction=0.0)
        else:
            trainer = make_trainer("aggregate_then_clip", SEED, "none", 4_000_000,
                                   "cumulative", 0.0)
        progress = {"iteration": 0, "measured_training_s": 0.0,
                    "next_snapshot_s": args.snapshot_minutes * 60,
                    "pending_snapshot": None, "save_current": args.arm in C_ARMS}
        atomic_json(manifest_path, {**expected, "created_utc": utc()})
        checkpoint(trainer, ckpt, progress)
    if trainer.strategy_train_steps != 6 or trainer.regret_positive_weight != 0:
        raise ValueError("Unexpected trainer fitting parameters")

    stop = [False]
    def request_stop(signum, _frame):
        stop[0] = True
        print(f"[signal] {signum}; checkpointing after current iteration", flush=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    target_s = args.minutes * 60
    last_print = -1
    while (progress["measured_training_s"] < target_s and not stop[0]
           and not (args.output_root / "PAUSE").exists()):
        start = time.perf_counter()
        record = trainer.run_iteration(traversals_per_player=ARMS[args.arm])
        iteration_s = time.perf_counter() - start
        if not np.isfinite(record["regret_loss"]).all() or not np.isfinite(record["strategy_loss"]).all():
            raise FloatingPointError("Non-finite fitting loss; previous checkpoint remains valid")
        progress["measured_training_s"] += iteration_s
        progress["iteration"] = trainer.iteration
        append_jsonl(run / "training.jsonl", {
            "utc": utc(), "arm": args.arm, "iteration": trainer.iteration,
            "measured_training_min": progress["measured_training_s"] / 60,
            "iteration_s": iteration_s, "timing": record["timing"],
            "roots_per_player": ARMS[args.arm],
            "regret_fit_learning_rate": trainer.regret_fit_learning_rate,
            "regret_fit_schedule": trainer.regret_fit_schedule,
            "regret_batch_size": trainer.regret_batch_size,
            "regret_loss": record["regret_loss"],
            "strategy_loss": record["strategy_loss"],
            "regret_records": record["new_regret_records"],
            "strategy_records": record["new_strategy_records"],
            "strategy_buffer_sizes": record["strategy_buffer_sizes"],
        })
        minute = int(progress["measured_training_s"] / 60)
        if minute > last_print and minute % 5 == 0:
            last_print = minute
            print(f"[train] {args.arm} {minute}m iter={trainer.iteration} "
                  f"iter_s={iteration_s:.3f} fit={record['timing']['regret_training_s']:.3f}/"
                  f"{record['timing']['strategy_training_s']:.3f}s", flush=True)

        if progress["measured_training_s"] >= progress["next_snapshot_s"]:
            # Bound disk use to one frozen reservoir waiting for each fit worker.
            while (pending_inputs(run) and not stop[0]
                   and not (args.output_root / "PAUSE").exists()):
                print(f"[wait] {args.arm} previous O4 fit still pending", flush=True)
                time.sleep(10)
            if stop[0] or (args.output_root / "PAUSE").exists():
                break
            label = snapshot_label(progress["next_snapshot_s"], args.snapshot_minutes)
            directory = run / "policy_snapshots" / label
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "READY.json").exists():
                raise RuntimeError(f"Refusing to replace completed snapshot {label}")
            staged = directory / "FIT_INPUT.pt.tmp"
            if staged.exists():
                staged.unlink()  # only an uncommitted temp file from this label
            started = time.perf_counter()
            freeze_input(trainer, staged, progress)
            while progress["next_snapshot_s"] <= progress["measured_training_s"]:
                progress["next_snapshot_s"] += args.snapshot_minutes * 60
            progress["pending_snapshot"] = label
            checkpoint(trainer, ckpt, progress)
            finish_pending(trainer, run, progress)
            if args.arm in C_ARMS and args.audits:
                queue_audits(run, ckpt, trainer.iteration)
            append_jsonl(run / "events.jsonl", {
                "event": "frozen_snapshot", "snapshot": label,
                "iteration": trainer.iteration,
                "measured_training_min": progress["measured_training_s"] / 60,
                "snapshot_s": time.perf_counter() - started, "utc": utc(),
            })
            print(f"[snapshot] {args.arm} {label} iter={trainer.iteration}", flush=True)
    progress["pending_snapshot"] = None
    checkpoint(trainer, ckpt, progress)
    if (args.arm in C_ARMS and args.audits
            and progress["measured_training_s"] >= target_s):
        queue_audits(run, ckpt, trainer.iteration, final=True)
    status = "paused" if stop[0] or (args.output_root / "PAUSE").exists() else "target_reached"
    atomic_json(run / "summary.json", {"status": status, "iteration": trainer.iteration,
        "measured_training_min": progress["measured_training_s"] / 60,
        "checkpoint": str(ckpt), "updated_utc": utc()})
    print(f"[done] {args.arm} {status} {progress['measured_training_s']/60:.2f}m", flush=True)


def fit_one(input_path: Path, steps: int, batch_size: int, threads: int,
            device: str = "cpu") -> None:
    torch.set_num_threads(threads)
    directory = input_path.parent
    ready_path = directory / "READY.json"
    if ready_path.exists():
        input_path.unlink(missing_ok=True)
        return
    start = time.perf_counter()
    state = torch.load(input_path, map_location="cpu", weights_only=False)
    spec_dict = dict(state["spec"])
    spec_dict["claim_kinds"] = tuple(spec_dict["claim_kinds"])
    spec = GameSpec(**spec_dict)
    policy = NeuralPolicy(spec, hidden_sizes=tuple(state.get("strategy_hidden_sizes", (256, 256))), device=device)
    models = (policy.model_p1, policy.model_p2)
    for model, weights in zip(models, state["strategy_nets"]):
        model.load_state_dict(weights)
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in models]
    for opt, saved in zip(optimizers, state["strategy_optimizers"]):
        opt.load_state_dict(saved)
    buffers = [DeviceReservoirBuffer.from_state_dict(b, device=device)
               for b in state["strategy_buffers"]]
    fit_seed = 17031 + int(directory.name[:-1])
    torch.manual_seed(fit_seed)
    for pid, (model, opt, buffer) in enumerate(zip(models, optimizers, buffers)):
        model.train()
        for step in range(steps):
            fraction = step / max(steps - 1, 1)
            lr = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * fraction))
            for group in opt.param_groups:
                group["lr"] = lr
            x, y, mask, weight = buffer.sample(batch_size)
            weight = weight / weight.mean().clamp_min(1e-8)
            logits = model(x).masked_fill(~mask, -1e9)
            per_sample = -(y * torch.log_softmax(logits, dim=1)).sum(dim=1)
            loss = (per_sample * weight).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (step + 1) % max(steps // 5, 1) == 0:
                print(f"[fit] {directory.parent.parent.name} {directory.name} "
                      f"p{pid+1} {step+1}/{steps} loss={loss.item():.6g}", flush=True)
        model.eval()
    final = directory / "average_policy"
    tmp = directory / "average_policy.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    if final.exists():
        shutil.rmtree(final)
    save_policy(policy.eval(), str(tmp))
    os.replace(tmp, final)
    atomic_json(ready_path, {
        "snapshot": directory.name, "policy_kind": "o4",
        "iteration": state["iteration"],
        "measured_training_min": state["measured_training_min"],
        "policy_dir": str(final), "fit_steps_per_player": steps,
        "fit_batch_size": batch_size, "fit_seed": fit_seed,
        "fit_s": time.perf_counter() - start, "utc": utc(),
    })
    backing = input_path.resolve()
    input_path.unlink()
    if backing != input_path and backing.is_file():
        backing.unlink()
    print(f"[ready] {directory.parent.parent.name} {directory.name} "
          f"fit_s={time.perf_counter()-start:.1f}", flush=True)


def run_fit(args: argparse.Namespace) -> None:
    arms = (args.arm,) if args.arm else tuple(C_ARMS)
    while not (args.output_root / "STOP_FITTER").exists():
        inputs = sorted(path for arm in arms for path in
                        (args.output_root.resolve() / arm / "policy_snapshots").glob("*/FIT_INPUT.pt"))
        if inputs:
            for path in inputs:
                arm = path.parents[2].name
                fit_one(path, args.fit_steps, args.fit_batch_size, args.threads,
                        "cuda" if arm in C_ARMS else "cpu")
            if args.once:
                return
        else:
            if args.once:
                return
            time.sleep(5)


def run_eval(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    root = args.output_root.resolve()
    evaluator = ROOT / "scripts/evaluate_cfr_plus_18_fit_snapshot.py"
    while not (root / "STOP_EVALUATOR").exists():
        changed = False
        for arm in ARMS:
            run = root / arm
            evaluations = run / "evaluations.jsonl"
            done = set()
            if evaluations.exists():
                for line in evaluations.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        done.add((row["snapshot"], row["policy_kind"]))
            for filename in ("ONLINE_READY.json", "CURRENT_READY.json", "READY.json"):
                for ready_path in sorted((run / "policy_snapshots").glob(f"*/{filename}")):
                    ready = json.loads(ready_path.read_text(encoding="utf-8"))
                    key = (ready["snapshot"], ready["policy_kind"])
                    if key in done:
                        continue
                    env = os.environ.copy()
                    env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
                    try:
                        result = subprocess.run(
                            [sys.executable, str(evaluator), ready["policy_dir"]],
                            cwd=ROOT, env=env, capture_output=True, text=True,
                            timeout=600, check=True)
                        score = json.loads(result.stdout.splitlines()[-1])
                    except Exception as exc:
                        print(f"[eval failed] {arm} {key}: {exc}", flush=True)
                        time.sleep(30)
                        continue
                    row = {"arm": arm, "snapshot": ready["snapshot"],
                           "policy_kind": ready["policy_kind"],
                           "iteration": ready["iteration"],
                           "measured_training_min": ready["measured_training_min"],
                           "policy_dir": ready["policy_dir"],
                           "p_first": score["p_first"],
                           "p_second": score["p_second"],
                           "exploitability": score["exploitability"],
                           "evaluation_s": score["evaluation_s"], "utc": utc()}
                    append_jsonl(evaluations, row)
                    done.add(key)
                    changed = True
                    print(f"[eval] {arm} {key} exploitability={score['exploitability']:.6f}", flush=True)
        if not changed:
            if args.once:
                return
            time.sleep(10)
        elif args.once:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "fit", "eval"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--minutes", type=float, default=600.0)
    parser.add_argument("--snapshot-minutes", type=float, default=SNAPSHOT_MIN)
    parser.add_argument("--fit-steps", type=int, default=FIT_STEPS)
    parser.add_argument("--fit-batch-size", type=int, default=FIT_BATCH)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--audits", action="store_true",
                        help="Queue exact regret audits at milestones (off by default)")
    parser.add_argument("--once", action="store_true", help="Process existing work and exit")
    args = parser.parse_args()
    if args.command == "train" and args.arm is None:
        parser.error("--arm is required for train")
    if args.minutes <= 0 or args.snapshot_minutes <= 0 or args.fit_steps <= 0 or args.fit_batch_size <= 0:
        parser.error("minutes, snapshot interval, fit steps and fit batch must be positive")
    if args.command == "train":
        run_train(args)
    elif args.command == "fit":
        run_fit(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
