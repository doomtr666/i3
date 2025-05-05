# This script updates the C# dependencies in the project.
dotnet restore
Push-Location
cd csharp/deps
dotnet paket install
$deps = Join-Path $pwd "paket.dependencies"
bazelisk run @rules_dotnet//tools/paket2bazel -- --dependencies-file "$deps" --output-folder "$pwd"
Pop-Location
