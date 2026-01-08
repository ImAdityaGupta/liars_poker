import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. Setup Environment / Path
# -----------------------------------------------------------------------------

def find_repo_root(start_dir: str) -> str:
    """Finds the repository root by looking for 'liars_poker' or 'pyproject.toml'."""
    cur = os.path.abspath(start_dir)
    for _ in range(6):
        if os.path.isdir(os.path.join(cur, "liars_poker")) or os.path.exists(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(os.path.join(start_dir, "..", ".."))

NB_DIR = os.getcwd()
REPO_ROOT = find_repo_root(NB_DIR)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ARTIFACTS_ROOT = os.path.join(REPO_ROOT, "artifacts")
os.makedirs(ARTIFACTS_ROOT, exist_ok=True)

print(f"Repo root   : {REPO_ROOT}")
print(f"Artifacts   : {ARTIFACTS_ROOT}")

# -----------------------------------------------------------------------------
# 2. Imports (after sys.path fix)
# -----------------------------------------------------------------------------

from liars_poker import GameSpec
from liars_poker.training.dense_fsp import dense_fsp_loop
from liars_poker.serialization import load_policy
from liars_poker.training.fsp_utils import (
    save_fsp_run,
    dense_fsp_resume,
    basic_eta_control,
    faster_eta_control,
    powerlaw_eta_control,
)

# -----------------------------------------------------------------------------
# 3. Main Logic
# -----------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Dense FSP Loop for Liar's Poker.")
    
    # Spec arguments
    parser.add_argument("--ranks", type=int, required=False, help="Number of ranks in the deck.")
    parser.add_argument("--suits", type=int, required=False, help="Number of suits in the deck.")
    parser.add_argument("--hand_size", type=int, required=False, help="Number of cards per player.")
    parser.add_argument(
        "--claim_kinds",
        nargs='+',
        required=False,
        help="List of allowed claim kinds (e.g. RankHigh Pair Trips)"
    )
    
    # Training arguments
    parser.add_argument("--episodes", type=int, required=True, help="Number of FSP episodes to run.")
    parser.add_argument(
        "--save_every",
        type=int,
        default=None,
        help="Checkpoint every N episodes (default: run all episodes in one chunk).",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Resume an existing run directory under run_folder.",
    )
    parser.add_argument(
        "--run_folder",
        type=str,
        default=os.path.join(ARTIFACTS_ROOT, "benchmark_runs"),
        help="Base folder for benchmark runs (default: artifacts/benchmark_runs).",
    )
    parser.add_argument(
        "--eta_control",
        choices=("basic", "faster", "powerlaw"),
        default="basic",
        help="Eta schedule to use (default: basic).",
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    print("\n--- Configuration ---")
    print(f"Episodes    : {args.episodes}")
    if args.save_every is not None:
        print(f"Save Every  : {args.save_every}")
    print(f"Eta Control : {args.eta_control}")
    if args.run_dir is not None:
        print(f"Run Dir     : {args.run_dir}")
    else:
        print(f"Ranks       : {args.ranks}")
        print(f"Suits       : {args.suits}")
        print(f"Hand Size   : {args.hand_size}")
        print(f"Claim Kinds : {args.claim_kinds}")
    print(f"Run Folder  : {args.run_folder}")
    print("---------------------\n")

    if args.run_dir is None:
        if args.ranks is None or args.suits is None or args.hand_size is None or not args.claim_kinds:
            raise ValueError("Spec args (--ranks/--suits/--hand_size/--claim_kinds) are required unless --run_dir is provided.")
        # Construct the GameSpec based on arguments
        # Note: suit_symmetry is assumed True as per instructions
        spec = GameSpec(
            ranks=args.ranks,
            suits=args.suits,
            hand_size=args.hand_size,
            claim_kinds=tuple(args.claim_kinds),
            suit_symmetry=True
        )
        print(f"Initialized Spec: {spec}\nStarting Training...")

    run_folder = args.run_folder
    if not os.path.isabs(run_folder):
        run_folder = os.path.join(REPO_ROOT, run_folder)
    os.makedirs(run_folder, exist_ok=True)

    if args.run_dir is None:
        time_right_now_string = datetime.now().strftime("%Y%m%d-%H%M%S")
        short_form = spec.to_short_str() + '___' + time_right_now_string
        run_dir = os.path.join(run_folder, short_form)
    else:
        run_dir = args.run_dir
        if not os.path.isabs(run_dir):
            candidate = os.path.join(run_folder, run_dir)
            if os.path.exists(candidate):
                run_dir = candidate
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        short_form = os.path.basename(run_dir)
        policy_dir = os.path.join(run_dir, "policy")
        _, spec = load_policy(policy_dir)

    eta_map = {
        "basic": basic_eta_control,
        "faster": faster_eta_control,
        "powerlaw": powerlaw_eta_control,
    }
    eta_control = eta_map[args.eta_control]
    save_every = args.save_every
    if save_every is None or save_every <= 0:
        save_every = args.episodes

    remaining = args.episodes
    first_chunk = args.run_dir is None
    pol = None
    info = None
    while remaining > 0:
        chunk = min(save_every, remaining)
        if first_chunk:
            # Run the dense FSP loop from scratch.
            pol, info = dense_fsp_loop(
                spec=spec,
                episodes=chunk,
                eta_control=eta_control,
                episodes_test=0,
                efficient=True,
            )
            first_chunk = False
        else:
            # Resume from the last saved checkpoint.
            pol, info = dense_fsp_resume(
                run_dir,
                remaining_episodes=chunk,
                eta_control=eta_control,
                episodes_test=0,
                efficient=True,
            )

        print(f"\nCheckpoint saving to: {short_form} (episodes remaining after save: {remaining - chunk})")
        save_fsp_run(run_id=short_form, policy=pol, info=info, spec=spec, root=Path(run_folder))
        remaining -= chunk

    print("Done.")

if __name__ == "__main__":
    main()
