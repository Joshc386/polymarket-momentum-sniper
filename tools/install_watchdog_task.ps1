<#
.SYNOPSIS
    Register (or remove) the kill-switch watchdog as a Windows Scheduled Task
    that runs at logon and AUTO-RESTARTS on failure.

.DESCRIPTION
    ADR-0002 step 6: the watchdog must be supervised by the OS so that if it
    crashes, something restarts it. The bot is deliberately NOT auto-restarted
    (HALT is sticky and needs a human) — only the watchdog is.

    The task runs `python -m tools.watchdog` from the project root, at user
    logon, with no time limit, restarting every 1 minute up to 999 times if it
    exits. Only one instance runs at a time.

.PARAMETER Remove
    Unregister the task instead of installing it.

.PARAMETER TaskName
    Scheduled Task name. Default: "PolymarketKillSwitchWatchdog".

.PARAMETER Python
    Path to the python executable. Default: resolved from PATH via Get-Command.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\install_watchdog_task.ps1
    powershell -ExecutionPolicy Bypass -File tools\install_watchdog_task.ps1 -Remove
#>
[CmdletBinding()]
param(
    [switch]$Remove,
    [string]$TaskName = "PolymarketKillSwitchWatchdog",
    [string]$Python
)

$ErrorActionPreference = "Stop"

# Project root = parent of this script's directory (tools\..).
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

if ($Remove) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Output "Removed scheduled task '$TaskName'."
    } else {
        Write-Output "No scheduled task '$TaskName' to remove."
    }
    return
}

# Resolve python if not supplied.
if (-not $Python) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "python not found on PATH. Pass -Python <path>." }
    $Python = $cmd.Source
}
if (-not (Test-Path $Python)) { throw "Python executable not found: $Python" }

Write-Output "Project root : $ProjectRoot"
Write-Output "Python       : $Python"
Write-Output "Task name    : $TaskName"

$action = New-ScheduledTaskAction -Execute $Python -Argument "-m tools.watchdog" -WorkingDirectory $ProjectRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn
# Auto-restart on failure; run indefinitely; single instance.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -RestartInterval (New-TimeSpan -Minutes 1) -RestartCount 999 `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

# Replace any existing registration.
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal `
    -Description "Kill-switch watchdog for the Polymarket BTC bot (ADR-0002). Auto-restarts on failure." | Out-Null

Write-Output ""
Write-Output "Registered. Useful commands:"
Write-Output "  Start now : Start-ScheduledTask -TaskName '$TaskName'"
Write-Output "  Status    : Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Output "  Remove    : tools\install_watchdog_task.ps1 -Remove"
