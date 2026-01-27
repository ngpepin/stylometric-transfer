@echo off
setlocal enabledelayedexpansion

REM Move to repo root (two levels up from this script)
cd /d "%~dp0\..\.."

set VENV_DIR=.venv-build
if not exist "%VENV_DIR%" (
  python -m venv "%VENV_DIR%"
)
set VENV_PY=%VENV_DIR%\Scripts\python.exe

REM Install dependencies in venv
%VENV_PY% -m pip install --upgrade pip >nul 2>&1
%VENV_PY% -m pip install --upgrade -r requirements.txt

REM Check PyInstaller
%VENV_PY% -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
  echo PyInstaller is not available after install.
  exit /b 1
)

REM Clean previous builds
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM Build fingerprint_style executable
%VENV_PY% -m PyInstaller --clean --onefile --name fingerprint_style --add-data "LICENSE.md;." fingerprint_style.py
if errorlevel 1 exit /b 1

REM Build apply_fingerprint executable
%VENV_PY% -m PyInstaller --clean --onefile --name apply_fingerprint --add-data "LICENSE.md;." apply_fingerprint.py
if errorlevel 1 exit /b 1

echo Build complete. Binaries are in %CD%\dist
endlocal
