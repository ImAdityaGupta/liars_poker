# Running experiments on rented machines

This is the operating procedure for an agent working from the local `liars_poker`
checkout. The laptop is the source of truth for code and durable results. A rented
Vast.ai instance is temporary compute, even when its disk survives a stop.

**End-of-cycle rule:** after the requested experiment has finished and its results
have been copied and checked locally, **stop the instance** and report its ID and
status. Never destroy an instance unless the user explicitly asks to destroy it.
Stopping preserves its disk but continues storage charges; restarting may have to
wait for the GPU to become available. Destroying permanently deletes its data.
See [Vast's instance guide](https://docs.vast.ai/guides/instances/manage-instances).

## 1. Define the run before renting

Record the game spec, hypothesis, code revision, trainer settings, seeds, duration
or stopping rule, expected RAM/VRAM/disk needs, evaluation method, output path,
checkpoint interval, and maximum hourly rate. Decide what constitutes a short
smoke test and what files must come home. Check `git status --short`: uncommitted
and untracked code will not appear in a remote Git clone.

Inspect current offers and their **total** hourly price at the intended disk size.
For example, on the laptop:

```powershell
vastai search offers 'cpu_cores_effective>=64 cpu_ram>=120 rentable=true reliability>=0.98 dph_total<=1.0' --storage 50 --raw
```

Choose an on-demand offer unless the user wants an interruptible run. Compare CPU
model, RAM, GPU and VRAM, reliability, direct SSH, disk, network, and price. Check
the account balance. Offer IDs and prices change, so search again immediately
before creation. Disk is set when the instance is created. The CLI's
[`create instance` reference](https://github.com/vast-ai/vast-cli/blob/master/vastai/SKILL.md)
lists the supported flags.

## 2. Authenticate and create

The user creates the API key in the Vast console and stores it locally with
`vastai set api-key`; never ask them to paste it into chat or put it in this repo.
Use `vastai show user` to check access. Ensure a local SSH **public** key is
registered before renting. Keep private keys on the laptop.

```powershell
$created = vastai create instance OFFER_ID --image vastai/pytorch:@vastai-automatic-tag --disk 50 --ssh --direct --cancel-unavail --label liars-poker-experiment --raw | ConvertFrom-Json
$created | Select-Object success,new_contract
```

Replace `OFFER_ID` with a freshly checked offer. **Do not print the raw creation
response:** it can contain an `instance_api_key`. Similarly, `vastai show ssh-keys
--raw` can contain private-key fields. Filter to the specific non-secret fields
needed for a check. Do not place API keys in shell history, logs, or experiment
manifests.

Keep the returned instance ID. Poll `vastai show instance INSTANCE_ID` until it
is running, then verify the actual rate and hardware. Obtain its SSH URL with
`vastai ssh-url INSTANCE_ID`. If the registered key is rejected, attach the
local `.pub` key with `vastai attach ssh INSTANCE_ID PATH_TO_PUBLIC_KEY` and retry
after a short delay. Vast may provide both proxy and direct SSH addresses; the
direct address is preferable for large copies and can change.

## 3. Transfer exact code and set up Python

Prefer a committed revision. Record `git rev-parse HEAD` in the run manifest. A
remote clone/pull is convenient **only if it can fetch that revision**; a private
repository needs a narrowly scoped deploy credential. Do not copy a personal
GitHub private key to a rental. Alternatively, package the local source and send
it over SSH, recording the archive hash. This is necessary when work is still
uncommitted. A source archive is a one-time copy: later edits in VS Code on the
VM or on the laptop do not synchronize automatically. Avoid uploading `.venv`,
all of `artifacts`, local credentials, or unrelated files.

On the machine, extract to `/root/liars_poker`, create an isolated environment,
install project requirements, and verify imports. The current machine uses
`/root/liars_poker/.venv/bin/python`; the repository's `requirements.txt` covers
the project packages. Check `torch.cuda.is_available()` and run a tiny operation
on CUDA if GPU work is intended. Also run one tiny project iteration through the
selected traversal path before starting an expensive run. A rented image's name
does not guarantee that its Python environment has PyTorch installed.

## 4. Launch and monitor

Create the run directory **before** piping output to `tee`:

```bash
cd /root/liars_poker
mkdir -p artifacts/RUN_NAME
tmux new -s RUN_NAME
set -o pipefail
.venv/bin/python -u scripts/EXPERIMENT.py --output-root artifacts/RUN_NAME 2>&1 | tee -a artifacts/RUN_NAME/train.log
```

Detach from `tmux` with `Ctrl-b`, then `d`. Reattach with `tmux attach -t RUN_NAME`.
For the 18-claim **CPU** runner, prefix the Python command with
`CUDA_VISIBLE_DEVICES=""`; it intentionally refuses to run when CUDA is visible.
Set each process's PyTorch/BLAS thread count when running several CPU arms in
parallel so their total does not oversubscribe the machine. GPU arms should be
profiled for VRAM before parallel execution.

Use the script's real checkpoint/resume options. Save policies, metrics, and
checkpoints during the run, not only at the end. A `tmux` process survives an SSH
disconnect, **not a host or container restart**. After a restart, inspect the
last valid checkpoint and logs and resume from them. Check progress, disk space,
memory, GPU use, and the account's ongoing cost periodically. Never infer that
the job is healthy solely because the instance says `running`.

## 5. Retrieve and verify results

Copy run logs, manifests, evaluation tables, figures, snapshots, and any
checkpoints needed for future continuation to a local `artifacts` directory.
Use `scp`/`rsync` over direct SSH for large transfers, or `vastai copy`:

```text
vastai copy INSTANCE_ID:/root/liars_poker/artifacts/RUN_NAME/ local:artifacts/RUN_NAME/
```

The [Vast data-movement guide](https://docs.vast.ai/guides/instances/storage/data-movement)
describes these options and notes that proxy SSH can be slow for large copies.
Compare file counts/sizes and open the summaries and plots locally. For a long
run, copy important outputs periodically so a host failure does not lose the
entire experiment. Record the instance ID, code revision/archive hash, command,
package versions, machine details, and any interruptions in the run manifest.

Keep large model files and raw logs under ignored `artifacts/`. Commit curated
experiment documents, small data extracts, and figures under `docs/` when they
are ready for review. Git is for source and reproducible explanations, not a
replacement for artifact storage.

## 6. End the cycle: stop, then report

After verifying the local copy, stop the rental and check its status:

```text
vastai stop instance INSTANCE_ID
vastai show instance INSTANCE_ID
```

Report what completed, where local results live, the checkpoint to resume from,
the instance ID, and whether it is stopped. Storage charges continue while it
exists. **Do not call `vastai destroy instance` without the user's explicit
instruction to destroy that instance.** If a run fails before results can be
retrieved, preserve its files by stopping it and report the recovery path.

Current example (September 2026): instance `53176196` is the 50 GB EPYC 7763 /
RTX 4060 Ti 16 GB rental. Its proxy SSH host is `ssh9.vast.ai:16196`, using the
local `arena_key`. Its price was about `$0.4006/hour` when checked; recheck its
status and rate rather than treating these details as permanent.
