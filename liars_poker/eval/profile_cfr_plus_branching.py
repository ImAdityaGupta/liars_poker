from __future__ import annotations

import argparse
import json
import math
import time
from typing import Dict, Iterable

import torch

from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer
from liars_poker.algo.neural_cfr_plus_gpu import GPUDeepCFRPlusTraverser
from liars_poker.core import GameSpec
from liars_poker.env import rules_for_spec


def analytical_branching_profile(
    spec: GameSpec,
    *,
    traverser: int,
) -> Dict[str, object]:
    """Expected uniform-policy frontier sizes per initial deal.

    Traverser nodes fully expand every legal claim. Opponent nodes sample one
    action from CALL plus every higher claim. The initial regret networks are
    zero, so the production CFR+ fallback is exactly uniform.
    """

    k = len(rules_for_spec(spec).claims)
    counts = {-1: 1.0}
    rows = []
    peak_rows = 1.0
    total_traverser_edges = 0.0
    total_opponent_continuations = 0.0
    total_records = 0.0

    for depth in range(k + 1):
        active = sum(counts.values())
        if active <= 0.0:
            break
        actor = depth & 1
        next_counts: Dict[int, float] = {}
        traverser_edges = 0.0
        opponent_continuations = 0.0

        if actor == traverser:
            for last, count in counts.items():
                for claim in range(last + 1, k):
                    next_counts[claim] = next_counts.get(claim, 0.0) + count
                    traverser_edges += count
            total_traverser_edges += traverser_edges
            regret_records = active
            strategy_records = 0.0
        else:
            for last, count in counts.items():
                legal_count = k - last
                continuation_mass = count / legal_count
                for claim in range(last + 1, k):
                    next_counts[claim] = (
                        next_counts.get(claim, 0.0) + continuation_mass
                    )
                    opponent_continuations += continuation_mass
            total_opponent_continuations += opponent_continuations
            regret_records = 0.0
            strategy_records = active

        active_out = sum(next_counts.values())
        total_records += regret_records + strategy_records
        peak_rows = max(peak_rows, active, active_out)
        rows.append(
            {
                "depth": depth,
                "actor": actor,
                "active_rows_in_per_deal": active,
                "traverser_claim_edges_per_deal": traverser_edges,
                "opponent_continuations_per_deal": opponent_continuations,
                "active_rows_out_per_deal": active_out,
                "regret_records_per_deal": regret_records,
                "strategy_records_per_deal": strategy_records,
            }
        )
        counts = next_counts

    word_count = math.ceil(k / 63)
    return {
        "claim_count": k,
        "traverser": int(traverser),
        "history_word_count": word_count,
        "history_bytes_per_row": 8 * word_count,
        "peak_rows_per_deal": peak_rows,
        "total_traverser_edges_per_deal": total_traverser_edges,
        "total_opponent_continuations_per_deal": total_opponent_continuations,
        "total_records_per_deal": total_records,
        "depth_profile": rows,
    }


def live_branching_profile(
    spec: GameSpec,
    *,
    traverser: int,
    batch_size: int,
    device: str = "cuda",
    hidden_sizes: Iterable[int] = (64, 64),
    action_sample_count: int | None = None,
    action_sample_fraction: float | None = None,
) -> Dict[str, object]:
    started = time.perf_counter()
    trainer = DeepCFRPlusTrainer(
        spec,
        regret_hidden_sizes=tuple(hidden_sizes),
        # Strategy networks are not used by a CFR+ traversal. Keep them tiny
        # so the reported peak reflects frontier work and regret inference.
        strategy_hidden_sizes=(8,),
        device=device,
        seed=7,
        regret_buffer_capacity=1,
        strategy_buffer_capacity=1,
        batch_size=1,
        regret_train_steps=1,
        strategy_train_steps=1,
        validation_fraction=0.0,
        validation_buffer_capacity=1,
        traversal_backend="gpu_native",
        traversal_batch_size=batch_size,
        traverser_action_sample_count=action_sample_count,
        traverser_action_sample_fraction=action_sample_fraction,
        device_replay=True,
        fused_optimizer=False,
    )
    trainer.iteration = 1
    traverser_impl = GPUDeepCFRPlusTraverser(trainer)
    setup_s = time.perf_counter() - started
    traversal_started = time.perf_counter()
    result = traverser_impl.run_traversals(
        int(traverser),
        int(batch_size),
        profile=True,
        commit_records=False,
    )
    traversal_s = time.perf_counter() - traversal_started
    result.update(
        {
            "status": "complete",
            "spec": spec.to_json(),
            "claim_count": traverser_impl.k,
            "traverser": int(traverser),
            "batch_size": int(batch_size),
            "device": str(device),
            "setup_s": setup_s,
            "traversal_s": traversal_s,
            "wall_s": time.perf_counter() - started,
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--suits", type=int, required=True)
    parser.add_argument("--hand-size", type=int, required=True)
    parser.add_argument("--claim-kinds", nargs="+", required=True)
    parser.add_argument("--suit-symmetry", action="store_true")
    parser.add_argument("--traverser", type=int, choices=(0, 1), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--action-sample-count", type=int)
    parser.add_argument("--action-sample-fraction", type=float)
    return parser


def main() -> None:
    args = _parser().parse_args()
    spec = GameSpec(
        ranks=args.ranks,
        suits=args.suits,
        hand_size=args.hand_size,
        claim_kinds=tuple(args.claim_kinds),
        suit_symmetry=args.suit_symmetry,
    )
    try:
        result = live_branching_profile(
            spec,
            traverser=args.traverser,
            batch_size=args.batch_size,
            device=args.device,
            hidden_sizes=args.hidden_sizes,
            action_sample_count=args.action_sample_count,
            action_sample_fraction=args.action_sample_fraction,
        )
    except BaseException as exc:
        result = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "spec": spec.to_json(),
            "traverser": args.traverser,
            "batch_size": args.batch_size,
        }
        if torch.cuda.is_available():
            result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
