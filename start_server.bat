@echo off
echo 🏈 Starting NFL QB Predictor HTTP Server...
echo ✅ This solves CORS issues by serving over HTTP instead of file://
echo.
powershell -ExecutionPolicy Bypass -File "Start-Server.ps1"
pause