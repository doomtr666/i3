# i3-vrun.ps1 - Run i3 examples with Vulkan diagnostics
# Usage: .\tools\i3-vrun.ps1 <example_name> [-Werror] [-release]

param (
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Example,

    [Parameter(ValueFromRemainingArguments=$true)]
    $ExtraArgs
)

# Set environment variable for validation layers
$env:VK_INSTANCE_LAYERS = "VK_LAYER_KHRONOS_validation"

# Detect -Werror flag
$RunArgs = @()
$DiagArgs = @()

foreach ($arg in $ExtraArgs) {
    if ($arg -eq "-Werror" -or $arg -eq "--werror") {
        $DiagArgs += "--werror"
    } elseif ($arg -eq "-v" -or $arg -eq "--verbose") {
        $DiagArgs += "--verbose"
    } else {
        $RunArgs += $arg
    }
}

try {
    # Run the example and pipe output to vulkan_diagnostics
    # We merge stderr into stdout (2>&1) and pipe to our parser
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    cargo run --quiet -p $Example @RunArgs *>&1 | cargo run --quiet --manifest-path tools/vulkan_diagnostics/Cargo.toml -- @DiagArgs
    $ExitCode = $LASTEXITCODE
    exit $ExitCode
}
finally {
    # No cleanup needed
}
