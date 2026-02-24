Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

New-Item -ItemType Directory -Force ".tmp", ".uv-cache" | Out-Null
$env:TMP = (Resolve-Path ".tmp").Path
$env:TEMP = $env:TMP
$env:UV_CACHE_DIR = (Resolve-Path ".uv-cache").Path

python -m pip install --upgrade pip
python -m pip install uv
uv sync --all-extras

@'
import importlib

mods = ["streamlit", "torch", "open_clip", "google.cloud.vision", "pytest"]
print("Environment check:")
for mod in mods:
    try:
        importlib.import_module(mod)
        print(f"  {mod}: OK")
    except Exception as exc:
        print(f"  {mod}: FAIL -> {exc}")
'@ | python -

Write-Host ""
Write-Host "Setup complete."
Write-Host "TMP/TEMP set for this session to: $env:TMP"
Write-Host "UV cache set for this session to: $env:UV_CACHE_DIR"
