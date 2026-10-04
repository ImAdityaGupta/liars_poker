param(
    [string]$SshTarget = 'liars-vast-cpu',
    [int]$LocalOverviewPort = 18765,
    [int]$LocalExperimentPort = 18769,
    [int]$LocalRegretExperimentPort = 18770,
    [int]$LocalRootO4Port = 18771,
    [int]$LocalBrCalibrationPort = 18772,
    [int]$Local30ClaimPort = 18774
)

# Keep this script running locally. SSH keepalives detect broken Wi-Fi; the
# loop reconnects and restores the VM overview and retained 8769 page.
$ErrorActionPreference = 'Continue'
Write-Host "Local dashboards: http://127.0.0.1:$LocalOverviewPort, http://127.0.0.1:$LocalExperimentPort, http://127.0.0.1:$LocalRegretExperimentPort, http://127.0.0.1:$LocalRootO4Port, http://127.0.0.1:$LocalBrCalibrationPort, and http://127.0.0.1:$Local30ClaimPort"
while ($true) {
    Write-Host "Opening dashboard tunnel to $SshTarget ..."
    & ssh -N -T `
        -o ExitOnForwardFailure=yes `
        -o ServerAliveInterval=20 `
        -o ServerAliveCountMax=3 `
        -L "127.0.0.1:${LocalOverviewPort}:127.0.0.1:8765" `
        -L "127.0.0.1:${LocalExperimentPort}:127.0.0.1:8769" `
        -L "127.0.0.1:${LocalRegretExperimentPort}:127.0.0.1:8770" `
        -L "127.0.0.1:${LocalRootO4Port}:127.0.0.1:8771" `
        -L "127.0.0.1:${LocalBrCalibrationPort}:127.0.0.1:8772" `
        -L "127.0.0.1:${Local30ClaimPort}:127.0.0.1:8774" `
        $SshTarget
    Write-Host "Tunnel ended (exit $LASTEXITCODE). Retrying in 5 seconds..."
    Start-Sleep -Seconds 5
}
