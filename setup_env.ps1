# setup_env.ps1
param(
    [string]$PythonExe = "python"
)

Write-Host "Creating virtual environment .venv ..."
& $PythonExe -m venv .venv

Write-Host "Activating virtual environment ..."
# This activates only inside this script; that's fine for installation.
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (-Not (Test-Path $activateScript)) {
    Write-Error "Could not find $activateScript"
    exit 1
}
. $activateScript

Write-Host "Upgrading pip ..."
pip install --upgrade pip

Write-Host "Installing requirements from requirements.txt ..."
pip install -r requirements.txt

Write-Host "Downloading yolov8x.pt via Ultralytics ..."
python download_model.py

Write-Host ""
Write-Host "✅ Setup complete."
Write-Host "To use the environment later in this PowerShell window, run:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host ".\setup_env.ps1"
Write-Host ""