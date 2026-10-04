#!/usr/bin/env python3
"""Re-evaluate saved distillation fits with their playable policy semantics."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.algo.cfr_discount_exact_average import ExactAverageTabularDiscountTrainer
from liars_poker.infoset import CALL
from liars_poker.policies.neural import NeuralPolicy, compile_neural_to_dense
from liars_poker.policies.neural_regret import NeuralRegretMatchingPolicy
from liars_poker.serialization import load_policy
from scripts.run_cfr_plus_18_regret_table_distillation import SOURCES


class ExactRegretLookup(torch.nn.Module):
    """Return the source table's regrets for the compiler's encoded inputs."""

    def __init__(self, table: torch.Tensor, hand_counts: torch.Tensor) -> None:
        super().__init__()
        self.table = table
        self.hand_counts = hand_counts
        self.history_powers = 1 << torch.arange(18, dtype=torch.int64)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        n_ranks = self.hand_counts.shape[1]
        count_match = (features[:, None, :n_ranks] == self.hand_counts[None]).all(dim=2)
        if not count_match.any(dim=1).all():
            raise AssertionError("Compiler encoded an unknown rank-count hand")
        hand_index = count_match.long().argmax(dim=1)
        history_index = (features[:, n_ranks:].long() * self.history_powers).sum(dim=1)
        return self.table[history_index, hand_index]


def exploitability(policy) -> float:
    response = best_response_dense(policy.spec, policy, store_state_values=False)[1]["computer"]
    p_first, p_second = response.exploitability()
    return float(p_first + p_second - 1)


def playable_policy(saved, arm: str):
    if arm != "P-visit":
        if not isinstance(saved, NeuralRegretMatchingPolicy):
            raise TypeError(f"Expected saved regret model for {arm}, got {type(saved).__name__}")
        return saved
    # Earlier P-visit fits were serialized as regret policies even though they
    # were trained with cross-entropy. Their weights are strategy logits.
    if isinstance(saved, NeuralPolicy):
        return saved
    strategy = NeuralPolicy(saved.spec, hidden_sizes=saved.hidden_sizes, device="cpu")
    strategy.model_p1.load_state_dict(saved.model_p1.state_dict())
    strategy.model_p2.load_state_dict(saved.model_p2.state_dict())
    return strategy.eval()


def check_playable_parity(policy, dense, *, samples: int = 128) -> float:
    """Compare dense compilation with the policy's independent infoset API."""
    rng = np.random.default_rng(10203)
    valid = np.flatnonzero(dense.legal_counts > 0)
    worst = 0.0
    for hid in rng.choice(valid, size=min(samples, len(valid)), replace=False):
        hand_index = int(rng.integers(len(dense.hands)))
        cols = np.flatnonzero(dense.legal_mask[hid])
        legal = tuple(CALL if col == 0 else int(col - 1) for col in cols)
        history = tuple(i for i in range(policy.encoder.k) if (int(hid) >> i) & 1)
        probs = policy._legal_probs(
            pid=int(dense.popcount[hid] & 1),
            hand=dense.hands[hand_index],
            history=history,
            legal=legal,
        )
        gap = float(np.max(np.abs(dense.S[hid, hand_index, cols] - probs)))
        worst = max(worst, gap)
    return worst


def policy_gap_metrics(source, table_dense, compiled):
    hand_map = source._dense_hand_order
    hand_counts = source.hand_counts.int().tolist()
    representatives = np.array([np.flatnonzero(hand_map == i)[0] for i in range(len(hand_counts))])
    chance = np.empty((len(hand_counts), len(hand_counts)), dtype=np.float64)
    for own, own_counts in enumerate(hand_counts):
        for other, other_counts in enumerate(hand_counts):
            chance[own, other] = math.prod(
                math.comb(4, x) * math.comb(4 - x, y)
                for x, y in zip(own_counts, other_counts)
            )
    chance /= chance.sum()
    visits = np.zeros((2, *table_dense.S.shape[:2]), dtype=np.float64)
    actor = table_dense.popcount & 1
    for pid in (0, 1):
        opponent_reach = (table_dense.L_pid1 if pid == 0 else table_dense.L_pid0)[:, representatives]
        weights = opponent_reach @ chance.T
        weights[actor != pid] = 0
        weights /= max(float(weights.sum()), 1e-30)
        visits[pid] = weights[:, hand_map] * 4096

    gap = 0.5 * np.abs(table_dense.S - compiled.S).sum(axis=2)
    own_reach = np.where(actor[:, None] == 0, table_dense.L_pid0, table_dense.L_pid1)
    chance_hand = np.array([
        math.prod(math.comb(4, v) for v in counts) / math.comb(16, 2)
        for counts in hand_counts
    ])[hand_map]
    reach_weight = own_reach * chance_hand[None, :]
    legal = table_dense.legal_counts > 0
    bins = {}
    for pid in (0, 1):
        for label, mask in (
            ("lt_0.1", visits[pid] < 0.1),
            ("0.1_1", (visits[pid] >= 0.1) & (visits[pid] < 1)),
            ("1_10", (visits[pid] >= 1) & (visits[pid] < 10)),
            ("ge_10", visits[pid] >= 10),
        ):
            selected = mask & (actor[:, None] == pid) & legal[:, None]
            if selected.any():
                w = reach_weight[selected]
                bins[f"p{pid + 1}_{label}_uniform"] = float(gap[selected].mean())
                bins[f"p{pid + 1}_{label}_reach"] = float(
                    np.sum(gap[selected] * w) / max(float(w.sum()), 1e-30)
                )
    return {
        "tv_uniform": float(gap[legal].mean()),
        "tv_reach": float(np.sum(gap * reach_weight) / max(float(reach_weight.sum()), 1e-30)),
        "tv_by_expected_visit_bin": bins,
    }


def publish(arm_dir: Path, corrected: dict) -> None:
    for name in ("result.json", "evaluations.jsonl", "summary.json"):
        path = arm_dir / name
        backup = arm_dir / f"{name}.legacy_softmax"
        if path.exists() and not backup.exists():
            shutil.copy2(path, backup)
    tmp = arm_dir / "result.json.tmp"
    tmp.write_text(json.dumps(corrected, indent=2))
    tmp.replace(arm_dir / "result.json")
    log_path = arm_dir / "evaluations.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["exploitability"] = corrected["network_exploitability"]
        row["evaluation_rule"] = corrected["evaluation_rule"]
    tmp = arm_dir / "evaluations.jsonl.tmp"
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tmp.replace(log_path)
    summary_path = arm_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["exploitability"] = corrected["network_exploitability"]
        summary["evaluation_rule"] = corrected["evaluation_rule"]
        tmp = arm_dir / "summary.json.tmp"
        tmp.write_text(json.dumps(summary, indent=2))
        tmp.replace(summary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--arms", nargs="+", default=["R-visit", "R-mix"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--check-table", action="store_true")
    parser.add_argument("--publish", action="store_true", help="Preserve legacy files and publish corrected metrics")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    source_path = args.artifacts / SOURCES[args.source]
    source = ExactAverageTabularDiscountTrainer.load_fork_checkpoint(source_path)
    table_dense = source.current_policy_exact_dense()
    table_x = exploitability(table_dense)
    if args.check_table:
        lookup = ExactRegretLookup(
            source.regret_table.reshape(1 << 18, source.n_rank_hands, source.encoder.action_dim),
            source.hand_counts,
        )
        table_policy = NeuralRegretMatchingPolicy(source.spec, hidden_sizes=(16,), device="cpu")
        table_policy.model_p1 = lookup
        table_policy.model_p2 = lookup
        table_compiled = compile_neural_to_dense(table_policy, batch_size=16_384)
        table_error = float(np.max(np.abs(table_dense.S - table_compiled.S)))
        if table_error > 2e-6:
            raise AssertionError(f"Table round-trip changed its policy: {table_error}")
        table_roundtrip_x = exploitability(table_compiled)
        if abs(table_roundtrip_x - table_x) > 2e-6:
            raise AssertionError(f"Table round-trip changed exploitability: {table_roundtrip_x} vs {table_x}")
        print(json.dumps({"table_compile_max_error": table_error, "table_exploitability": table_x}), flush=True)
    root = args.artifacts / "cfr_plus_18_regret_table_distillation" / "main_20261002" / args.source
    for arm in args.arms:
        arm_dir = root / arm
        if not (arm_dir / "result.json").exists():
            print(f"{args.source}/{arm}: no finished fit", flush=True)
            continue
        saved, spec = load_policy(str(arm_dir / "policy"))
        if spec != source.spec:
            raise ValueError(f"Spec mismatch for {arm_dir}")
        policy = playable_policy(saved, arm)
        compiled = compile_neural_to_dense(policy, batch_size=16_384)
        parity = check_playable_parity(policy, compiled)
        # Single-infoset and batched MLP inference use different GEMM kernels.
        # Their float32 regret predictions can differ slightly near a zero sum.
        if parity > 1e-4:
            raise AssertionError(f"Compiled and playable policy disagree: {parity}")
        x = exploitability(compiled)
        gap = policy_gap_metrics(source, table_dense, compiled)
        old = json.loads((arm_dir / "result.json").read_text())
        corrected = {
            **old,
            "source_exploitability": table_x,
            "network_exploitability": x,
            "ratio": x / table_x,
            **gap,
            "evaluation_rule": "regret_matching" if arm != "P-visit" else "masked_softmax",
        }
        row = {
            "source": args.source,
            "arm": arm,
            "source_exploitability": table_x,
            "corrected_exploitability": x,
            "corrected_ratio": x / table_x,
            "corrected_tv_uniform": gap["tv_uniform"],
            "corrected_tv_reach": gap["tv_reach"],
            "corrected_tv_by_expected_visit_bin": gap["tv_by_expected_visit_bin"],
            "compile_vs_playable_max_error": parity,
            "old_result": old,
        }
        (arm_dir / "corrected_diagnosis.json").write_text(json.dumps(row, indent=2))
        if args.publish:
            publish(arm_dir, corrected)
        print(json.dumps({k: v for k, v in row.items() if k != "old_result"}), flush=True)


if __name__ == "__main__":
    main()
