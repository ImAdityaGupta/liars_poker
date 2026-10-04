"""O4 refit of exact4096's final checkpoint (31,538 iterations), using Part A's fit_one_arm.

Asks whether O4 still lands near the exact average (~1.15x) after a long run.
Reads the source checkpoint only; writes under OUTPUT.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path("/root/liars_poker")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import torch  # noqa: E402
from run_cfr_plus_18_average_fit_optimizer_experiment import fit_one_arm  # noqa: E402

SOURCE = REPO / "artifacts/cfr_plus_18_batched_bridge_controls/main_20260930/exact4096/latest_checkpoint.pt"
OUTPUT = REPO / "artifacts/cfr_plus_18_average_fit_long_run_check/main_20261002"
O4 = (1e-3, 1e-5, 16384, "cosine", "cross_entropy")
ARMS = [("O4", 17031, "warm", 5000), ("O4_seed17032", 17032, "warm", 5000),
        ("O4_seed17033", 17033, "warm", 5000), ("F40k", 17031, "fresh", 40000)]

OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "manifest.json").write_text(json.dumps({
    "source": str(SOURCE), "source_iteration": 31538, "source_exact_average_exploitability": 0.001005141776837748,
    "recipe": O4, "arms": ARMS, "harness": "scripts/run_cfr_plus_18_average_fit_optimizer_experiment.py:fit_one_arm",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
torch.set_num_threads(4)
for arm, seed, variant, steps in ARMS:
    milestones = (steps,) if variant == "warm" else (20000, 40000)
    fit_one_arm("1080m", arm, O4, OUTPUT / "1080m" / arm, SOURCE, fit_seed=seed, steps=steps,
                milestones=milestones, loss_name="cross_entropy", variant=variant)
print("[done]", flush=True)
