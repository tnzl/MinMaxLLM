<#
.SYNOPSIS
Build script for MinMaxLLM project with AVX2 optimized CPU operations.

.DESCRIPTION
This script provides options to configure, build, and clean the CMake project with
AVX2 optimization flags for CPU operations, using Visual Studio as the generator.

.PARAMETER help
Show this help message and exit.

.PARAMETER type
Specify the build type (Debug, Release, RelWithDebInfo, MinSizeRel).

.PARAMETER clean
Clean the build directory before building.

.PARAMETER noavx
Disable AVX2 optimizations (build without AVX2).

.EXAMPLE
./build.ps1
./build.ps1 -type Debug
./build.ps1 -clean
./build.ps1 -noavx
./build.ps1 -help
#>

param(
    [switch]$help = $false,
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$type = "Release",
    [switch]$clean = $false,
    [switch]$noavx = $false
)

if ($help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

$projectRoot = $PSScriptRoot
$buildDir = Join-Path $projectRoot "build"

if ($clean -and (Test-Path $buildDir)) {
    Write-Host "Cleaning build directory..."
    Remove-Item $buildDir -Recurse -Force
}

if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

# Find Visual Studio installation
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -property installationPath
    $vsVersion = & $vswhere -latest -property catalog_productLineVersion
    Write-Host "Found Visual Studio $vsVersion at $vsPath"
    
    # Import Visual Studio environment variables
    $vsDevCmd = Join-Path $vsPath "Common7\Tools\vsdevcmd.bat"
    if (Test-Path $vsDevCmd) {
        Write-Host "Setting up Visual Studio environment..."
        cmd /c "`"$vsDevCmd`" -arch=x64 && set" | ForEach-Object {
            if ($_ -match "^(.*?)=(.*)$") {
                Set-Content "env:\$($matches[1])" $matches[2]
            }
        }
    }
}

# Prepare CMake configuration command
$cmakeArgs = @(
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DCMAKE_BUILD_TYPE=$type",
    "-B", $buildDir,
    "-S", $projectRoot
)

# Add AVX2 flags unless disabled
if (-not $noavx) {
    $cmakeArgs += "-DUSE_AVX2=ON"
    Write-Host "Building with AVX2 optimizations" -ForegroundColor Cyan
} else {
    Write-Host "Building without AVX2 optimizations" -ForegroundColor Yellow
}

# Configure the project
Write-Host "Configuring project with build type: $type"
cmake $cmakeArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

# Build the project
Write-Host "Building project..."
cmake --build $buildDir --config $type -- /m

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Build completed successfully!" -ForegroundColor Green