# i3-cargo.ps1 - Proxy command to run cargo with automated AI-friendly diagnostics
$TempFile = [System.IO.Path]::GetTempFileName()
try {
    # Run cargo with provided arguments and inject --message-format=json
    # Note: We use @Args to pass all parameters through
    cargo $args --message-format=json 2>$null | Out-File -FilePath $TempFile -Encoding UTF8
    $CargoExitCode = $LASTEXITCODE
    
    # Run diagnostic tool on the result
    cargo run --quiet --manifest-path tools/rust_diagnostics/Cargo.toml -- $TempFile
    
    # Propagate cargo exit code if diagnostic tool didn't already exit with error
    if ($CargoExitCode -ne 0 -and $LASTEXITCODE -eq 0) {
        exit $CargoExitCode
    }
}
finally {
    if (Test-Path $TempFile) {
        Remove-Item $TempFile
    }
}
