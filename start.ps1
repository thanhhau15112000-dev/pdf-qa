$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    python -m venv .venv
    & '.venv\Scripts\python.exe' -m pip install -r requirements.lock
    if ($LASTEXITCODE -ne 0) { throw 'Không cài được dependency.' }
}
& '.venv\Scripts\python.exe' -m uvicorn app.main:app --host 127.0.0.1 --port 8000
