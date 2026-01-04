import os
import sys
import argparse
from datetime import datetime

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
from liars_poker.training.fsp_utils import save_fsp_run, dense_fsp_resume

# -----------------------------------------------------------------------------
# 3. Main Logic
# -----------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Dense FSP Loop for Liar's Poker.")
    
    # Spec arguments
    parser.add_argument("--ranks", type=int, required=True, help="Number of ranks in the deck.")
    parser.add_argument("--suits", type=int, required=True, help="Number of suits in the deck.")
    parser.add_argument("--hand_size", type=int, required=True, help="Number of cards per player.")
    parser.add_argument(
        "--claim_kinds", 
        nargs='+', 
        required=True, 
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
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    print("\n--- Configuration ---")
    print(f"Ranks       : {args.ranks}")
    print(f"Suits       : {args.suits}")
    print(f"Hand Size   : {args.hand_size}")
    print(f"Claim Kinds : {args.claim_kinds}")
    print(f"Episodes    : {args.episodes}")
    if args.save_every is not None:
        print(f"Save Every  : {args.save_every}")
    print("---------------------\n")

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

    time_right_now_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    short_form = spec.to_short_str() + '___' + time_right_now_string
    run_dir = os.path.join(ARTIFACTS_ROOT, "benchmark_runs", short_form)

    save_every = args.save_every
    if save_every is None or save_every <= 0:
        save_every = args.episodes

    remaining = args.episodes
    first_chunk = True
    pol = None
    info = None
    while remaining > 0:
        chunk = min(save_every, remaining)
        if first_chunk:
            # Run the dense FSP loop from scratch.
            pol, info = dense_fsp_loop(
                spec=spec,
                episodes=chunk,
                episodes_test=0,
                efficient=True,
            )
            first_chunk = False
        else:
            # Resume from the last saved checkpoint.
            pol, info = dense_fsp_resume(
                run_dir,
                remaining_episodes=chunk,
                episodes_test=0,
                efficient=True,
            )

        print(f"\nCheckpoint saving to: {short_form} (episodes remaining after save: {remaining - chunk})")
        save_fsp_run(run_id=short_form, policy=pol, info=info, spec=spec)
        remaining -= chunk

    print("Done.")

if __name__ == "__main__":
    main()
