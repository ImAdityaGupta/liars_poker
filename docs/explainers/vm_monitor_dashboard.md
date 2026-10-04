# VM overview dashboard

The general VM overview runs on port **8765**. It shows CPU, memory, disk and GPU history, plus active and recently updated experiment arms with status, training minutes, iteration, latest exact exploitability, evaluation count and checkpoint size. It scans both `/root/liars_poker/artifacts` and `/root/liars_poker_20261001/artifacts` when both checkouts exist. It reads logs only; it does not run evaluations or touch trainer state. The page binds to VM loopback and is intended to be used through SSH forwarding.

The experiment-specific dashboard on **8769** remains separate. The old experiment monitors on 8766–8768 have been retired, and the old 8765 monitor was replaced by this overview. The start script only stops processes whose command line identifies them as experiment monitors and always preserves 8769.

## Start on the VM

From the repository root:

```bash
bash scripts/start_vm_overview_dashboard.sh
```

This starts a restart loop inside tmux session `vm_overview`. It restarts the dashboard if its Python process exits. Check it with:

```bash
tmux attach -t vm_overview
```

Detach with `Ctrl-B`, then `D`. A VM instance restart destroys its tmux server, so start the session again after the instance itself reboots. The SSH tunnel reconnects independently; it cannot restart a process on a rebooted VM.

## Keep the browser tunnel reconnecting

On Windows, start a PowerShell window and run:

```powershell
.\scripts\tunnel_vm_dashboards.ps1
```

The script forwards local ports **18765** and **18769** to VM ports 8765 and 8769, then reconnects five seconds after SSH exits. The previous local ports 8765/8769 were denied by Windows when SSH tried to bind them. Keep that PowerShell process running. After Wi-Fi returns, SSH keepalives detect the broken connection and the loop restores the forwards. Open `http://127.0.0.1:18765` for VM status and `http://127.0.0.1:18769` for the retained experiment dashboard. You can override these local ports with `-LocalOverviewPort` and `-LocalExperimentPort` if needed.

This is the Windows equivalent of using `autossh`: it repairs the local tunnel after connectivity loss. `tmux` keeps the remote monitor detached from SSH; the monitor's loop restarts its Python server if it exits. Neither mechanism survives a full VM restart by itself.
