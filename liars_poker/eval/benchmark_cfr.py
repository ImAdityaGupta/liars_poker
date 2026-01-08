import os
import sys
import argparse
from datetime import datetime


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


from liars_poker import GameSpec
from liars_poker.training.cfr_dense import cfr_dense_loop, cfr_dense_resume, save_cfr_run


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Dense CFR Loop for Liar's Poker.")

    parser.add_argument("--ranks", type=int, required=False, help="Number of ranks in the deck.")
    parser.add_argument("--suits", type=int, required=False, help="Number of suits in the deck.")
    parser.add_argument("--hand_size", type=int, required=False, help="Number of cards per player.")
    parser.add_argument(
        "--claim_kinds",
        nargs="+",
        required=False,
        help="List of allowed claim kinds (e.g. RankHigh Pair Trips TwoPair)",
    )

    parser.add_argument("--episodes", type=int, required=True, help="Number of CFR iterations to run.")
    parser.add_argument(
        "--save_every",
        type=int,
        default=None,
        help="Checkpoint every N iterations (default: run all iterations in one chunk).",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Resume an existing run directory under artifacts/benchmark_runs.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=0,
        help="Run best-response evaluation every N iterations (0 disables).",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("\n--- Configuration ---")
    print(f"Episodes    : {args.episodes}")
    if args.save_every is not None:
        print(f"Save Every  : {args.save_every}")
    print(f"Eval Every  : {args.eval_every}")
    if args.run_dir is not None:
        print(f"Run Dir     : {args.run_dir}")
    else:
        print(f"Ranks       : {args.ranks}")
        print(f"Suits       : {args.suits}")
        print(f"Hand Size   : {args.hand_size}")
        print(f"Claim Kinds : {args.claim_kinds}")
    print("---------------------\n")

    if args.run_dir is None:
        if args.ranks is None or args.suits is None or args.hand_size is None or not args.claim_kinds:
            raise ValueError(
                "Spec args (--ranks/--suits/--hand_size/--claim_kinds) are required unless --run_dir is provided."
            )
        spec = GameSpec(
            ranks=args.ranks,
            suits=args.suits,
            hand_size=args.hand_size,
            claim_kinds=tuple(args.claim_kinds),
            suit_symmetry=True,
        )
        print(f"Initialized Spec: {spec}\nStarting Training...")

    if args.run_dir is None:
        time_right_now_string = datetime.now().strftime("%Y%m%d-%H%M%S")
        short_form = spec.to_short_str() + "___" + time_right_now_string
        run_dir = os.path.join(ARTIFACTS_ROOT, "benchmark_runs", short_form)
    else:
        run_dir = args.run_dir
        if not os.path.isabs(run_dir):
            candidate = os.path.join(ARTIFACTS_ROOT, "benchmark_runs", run_dir)
            if os.path.exists(candidate):
                run_dir = candidate
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        short_form = os.path.basename(run_dir)

    save_every = args.save_every
    if save_every is None or save_every <= 0:
        save_every = args.episodes

    remaining = args.episodes
    first_chunk = args.run_dir is None
    policy = None
    logs = None
    total_iters = 0

    while remaining > 0:
        chunk = min(save_every, remaining)
        if first_chunk:
            policy, logs, cfr = cfr_dense_loop(
                spec=spec,
                iterations=chunk,
                start_iter=0,
                eval_every=args.eval_every,
                debug=False,
            )
            total_iters = chunk
            first_chunk = False
        else:
            policy, logs, cfr, spec, total_iters = cfr_dense_resume(
                run_dir,
                remaining_iterations=chunk,
                eval_every=args.eval_every,
                debug=False,
            )

        print(f"\nCheckpoint saving to: {short_form} (iterations remaining after save: {remaining - chunk})")
        save_cfr_run(
            run_id=short_form,
            policy=policy,
            cfr=cfr,
            logs=logs or {},
            spec=spec,
            iterations_done=total_iters,
            eval_every=args.eval_every,
        )
        remaining -= chunk

    print("Done.")


if __name__ == "__main__":
    main()
