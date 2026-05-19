# PowerShell script to install WSL2 with Ubuntu 24.04
# Run as Administrator in PowerShell

Write-Host "Installing WSL2..." -ForegroundColor Green

# Enable WSL and Virtual Machine Platform
wsl --install --no-distribution

# Enable required Windows features
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Install Ubuntu 24.04
wsl --install -d Ubuntu-24.04

Write-Host "WSL2 + Ubuntu 24.04 installation initiated." -ForegroundColor Green
Write-Host "Reboot required. After reboot, run: wsl -d Ubuntu-24.04" -ForegroundColor Yellow
Write-Host "Then inside WSL, run: /mnt/e/HAGI/setup/ubuntu-setup.sh" -ForegroundColor Cyan
