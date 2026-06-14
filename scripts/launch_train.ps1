# Launch training in an independent PowerShell window.
# This window stays open even if the IDE is closed.
# Training logs go to E:\humanoid_squat_pick_rl\logs\train_output.log

$ProjectDir = "E:\humanoid_squat_pick_rl"
$Python = "$ProjectDir\.venv\Scripts\python.exe"
$Script = "$ProjectDir\scripts\train.py"
$LogFile = "$ProjectDir\logs\train_output.log"

# Ensure log dir exists
New-Item -ItemType Directory -Force -Path "$ProjectDir\logs" | Out-Null

Write-Host "Starting training in background..."
Write-Host "  Log file: $LogFile"
Write-Host "  To monitor: Get-Content $LogFile -Wait"

# Run training, redirect all output to log file
& $Python $Script --total-timesteps 5000000 --n-envs 8 2>&1 | Tee-Object -FilePath $LogFile

Write-Host "`nTraining finished. Check $LogFile for results."
pause
