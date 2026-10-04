"""Chained O4: start each checkpoint's refit from the previous checkpoint's refit.

Plain O4 warm-starts from the online average network at every snapshot. Here
each link instead starts from the previous link's fitted network and Adam
state, then anneals on the new checkpoint's reservoir. The chain starts from
Part A's O4 fits at 908 iterations and runs 1,424 -> 3,988 -> 31,538.

Reuses fit_one_arm unchanged: each link's output folder is pre-seeded with a
resume state holding the previous link's weights and fitted=0.
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

ART = REPO / "artifacts"
OUTPUT = ART / "cfr_plus_18_average_fit_chain_check/main_20261003"
START = {  # Part A O4 fits at 908 iterations, by fit seed
    17031: ART / "cfr_plus_18_average_fit_optimizer/main_20261001/0030m/O4_cosine_b16384_ce/resume_state.pt",
    17032: ART / "cfr_plus_18_average_fit_optimizer/main_20261001/0030m/O4_cosine_b16384_ce_seed17032/resume_state.pt",
}
LINKS = [
    ("0045m", ART / "cfr_plus_18_offline_average_study/exact4096_0045m_checkpoint.pt"),
    ("0120m", ART / "cfr_plus_18_offline_average_study/exact4096_0120m_checkpoint.pt"),
    ("1080m", ART / "cfr_plus_18_batched_bridge_controls/main_20260930/exact4096/latest_checkpoint.pt"),
]
# name: (lr_start, lr_end, steps)
RECIPES = {"CH_hi_2k": (1e-3, 1e-5, 2000), "CH_lo_2k": (1e-4, 1e-6, 2000), "CH_hi_5k": (1e-3, 1e-5, 5000)}

OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "manifest.json").write_text(json.dumps({
    "start": {str(k): str(v) for k, v in START.items()}, "links": [(c, str(p)) for c, p in LINKS],
    "recipes": RECIPES, "batch": 16384, "loss": "weighted cross entropy",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
torch.set_num_threads(4)
for seed, start in START.items():
    for name, (lr_start, lr_end, steps) in RECIPES.items():
        previous = start
        for checkpoint, source in LINKS:
            out = OUTPUT / checkpoint / f"{name}_s{seed}"
            out.mkdir(parents=True, exist_ok=True)
            state = out / "resume_state.pt"
            if not state.exists():
                prev = torch.load(previous, map_location="cpu", weights_only=False)
                torch.save({"fitted": 0, "models": prev["models"], "optimizers": prev["optimizers"],
                            "last_row": None}, state)
            fit_one_arm(checkpoint, f"{name}_s{seed}", (lr_start, lr_end, 16384, "cosine", "cross_entropy"),
                        out, source, fit_seed=seed, steps=steps, milestones=(steps,),
                        loss_name="cross_entropy", variant="warm")
            previous = state
print("[done]", flush=True)
