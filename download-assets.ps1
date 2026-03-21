#!/usr/bin/env pwsh
# download-assets.ps1
# Script to download testing assets for i3 engine

$AssetDir = Join-Path $PSScriptRoot "assets"
if (-not (Test-Path $AssetDir)) {
    Write-Host "[INFO] Creating asset directory: $AssetDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $AssetDir | Out-Null
}

# 1. Khronos Sample Assets (Full repository)
$KhronosRepo = "https://github.com/KhronosGroup/glTF-Sample-Assets.git"
$KHDir = Join-Path $AssetDir "gltf-sample-assets"

Write-Host "[INFO] Cloning Khronos glTF Sample Assets..." -ForegroundColor Cyan
if (-not (Test-Path $KHDir)) {
    git clone --depth 1 $KhronosRepo $KHDir
} else {
    Write-Host "[INFO] glTF Sample Assets already present. Checking for updates..." -ForegroundColor Yellow
    Push-Location $KHDir
    git pull
    Pop-Location
}

# Optimization for PowerShell overhead
$ProgressPreference = 'SilentlyContinue'

# 2. NVIDIA ORCA Bistro
$BistroZipUrl = "https://developer.nvidia.com/downloads/bistro"
$BistroZip = Join-Path $AssetDir "Bistro_v5_2.zip"
$BistroDir = Join-Path $AssetDir "Bistro_v5_2"

Write-Host "[INFO] Checking NVIDIA ORCA Bistro..." -ForegroundColor Cyan
if (-not (Test-Path $BistroDir)) {
    Write-Host "[INFO] Downloading NVIDIA ORCA Bistro from NVIDIA (this may take a while)..." -ForegroundColor Cyan
    try {
        Invoke-WebRequest -Uri $BistroZipUrl -OutFile $BistroZip -UseBasicParsing
        Write-Host "[INFO] Extracting Bistro..." -ForegroundColor Yellow
        
        # Extract to a temporary location to handle any internal folder structure
        $TempExtractDir = Join-Path $AssetDir "temp_bistro"
        Expand-Archive -Path $BistroZip -DestinationPath $TempExtractDir -Force
        
        # Ensure target directory exists
        if (-not (Test-Path $BistroDir)) { New-Item -ItemType Directory -Path $BistroDir | Out-Null }
        
        # If the zip already contained a 'Bistro_v5_2' folder, move its contents.
        # Otherwise, move everything from the temp dir.
        $InternalDir = Join-Path $TempExtractDir "Bistro_v5_2"
        if (Test-Path $InternalDir) {
            Get-ChildItem -Path $InternalDir | Move-Item -Destination $BistroDir -Force
        } else {
            Get-ChildItem -Path $TempExtractDir | Move-Item -Destination $BistroDir -Force
        }
        
        # Cleanup
        Remove-Item $TempExtractDir -Recurse -Force
        Remove-Item $BistroZip
    } catch {
        Write-Host "[ERROR] Failed to download Bistro automatically." -ForegroundColor Red
        Write-Host "        Please download it manually from: https://developer.nvidia.com/orca/amazon-lumberyard-bistro" -ForegroundColor Gray
        Write-Host "        And extract it to: $BistroDir" -ForegroundColor Gray
    }
} else {
    Write-Host "[INFO] Bistro already present." -ForegroundColor Green
}

Write-Host ""
Write-Host "[SUCCESS] Asset download completed!" -ForegroundColor Green
Write-Host "          Assets are located in: $KHDir" -ForegroundColor Gray
