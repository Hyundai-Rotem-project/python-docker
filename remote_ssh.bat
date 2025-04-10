@echo off
REM PowerShell 스크립트 실행 정책 우회해서 실행
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0remote_ssh.ps1"
pause
