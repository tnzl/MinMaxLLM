# PowerShell script to run all test executables in the build/Release directory

$testDir = Join-Path $PSScriptRoot 'build/Release'

if (-Not (Test-Path $testDir)) {
    Write-Host "Test directory not found: $testDir"
    exit 1
}

$testExecutables = @(
    'test_gqa.exe',
    'test_matmul.exe',
    'test_rotary_embedding.exe',
    'test_silu_avx2.exe',
    'test_SkipSimplifiedLayerNormalization_AVX2.exe',
    'test_SimplifiedLayerNormalization_AVX2.exe',
    'test_softmax_avx2.exe'
)

$failed = $false

foreach ($exe in $testExecutables) {
    $exePath = Join-Path $testDir $exe
    if (Test-Path $exePath) {
        Write-Host "Running $exe..."
        & $exePath
        if ($LASTEXITCODE -ne 0) {
            Write-Host "$exe failed with exit code $LASTEXITCODE" -ForegroundColor Red
            $failed = $true
        } else {
            Write-Host "$exe passed." -ForegroundColor Green
        }
    } else {
        Write-Host "$exe not found in $testDir" -ForegroundColor Yellow
        $failed = $true
    }
}

if ($failed) {
    Write-Host "Some tests failed." -ForegroundColor Red
    exit 1
} else {
    Write-Host "All tests passed!" -ForegroundColor Green
    exit 0
}
