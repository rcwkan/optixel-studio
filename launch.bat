@echo off
rem One-click launcher: starts Optixel Studio in dev mode (close this window to quit).
cd /d "%~dp0"
set CARGO_INCREMENTAL=0
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
if not exist node_modules call npm install
npm run tauri dev
pause
