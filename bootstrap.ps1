#!/usr/bin/env pwsh
# Bootstrap script to download native dependencies
# Run this script before building the project for the first time

Write-Host "i3 Native Dependencies Bootstrap" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Check if cargo is available
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] cargo not found in PATH" -ForegroundColor Red
    Write-Host "        Please install Rust from https://rustup.rs/" -ForegroundColor Yellow
    exit 1
}

Write-Host "[INFO] Downloading native dependencies (SDL2, etc.)..." -ForegroundColor Yellow
Write-Host ""

# Run the download script
$process = Start-Process -FilePath "cargo" -ArgumentList "run", "--manifest-path", "third_party/Cargo.toml", "--bin", "download" -NoNewWindow -Wait -PassThru

if ($process.ExitCode -eq 0) {
    Write-Host ""
    Write-Host "[SUCCESS] Bootstrap completed successfully!" -ForegroundColor Green
    Write-Host "          You can now build the project with: cargo build" -ForegroundColor Cyan
    exit 0
} else {
    Write-Host ""
    Write-Host "[ERROR] Bootstrap failed with exit code $($process.ExitCode)" -ForegroundColor Red
    Write-Host "        Check the error messages above for details" -ForegroundColor Yellow
    exit $process.ExitCode
}
