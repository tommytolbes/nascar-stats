# NASCAR Stats - Weekly Task Scheduler Setup
# Run this ONCE in PowerShell (as Administrator) to register the Monday update.
#
# How to run:
#   1. Press the Windows key, search "PowerShell"
#   2. Right-click it and choose "Run as administrator"
#   3. Paste this entire script and press Enter

$taskName   = "NASCAR Stats Weekly Update"
$batFile    = "C:\Users\thoma\OneDrive\Desktop\Misc\Claude\Projects\NASCAR\.claude\worktrees\fervent-maxwell\update.bat"
$triggerDay = "Monday"
$triggerTime = "12:00"

# Remove existing task with the same name if it exists
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Create the trigger: every Monday at 12:00 PM
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $triggerDay -At $triggerTime

# Run as the current user, only when logged in
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

# The action: run the batch file
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$batFile`""

# Register the task
Register-ScheduledTask -TaskName $taskName `
                       -Trigger $trigger `
                       -Action $action `
                       -Principal $principal `
                       -Description "Pulls new NASCAR race results and rebuilds fantasy scores every Monday at noon."

Write-Host ""
Write-Host "Task '$taskName' registered successfully." -ForegroundColor Green
Write-Host "It will run every $triggerDay at $triggerTime."
Write-Host "Logs are saved to: update_log.txt in the project folder."
