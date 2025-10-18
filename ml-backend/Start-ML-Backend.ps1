# BetYard ML Backend Starter Script
# Run this to start the ML prediction API server

Write-Host "🏈 Starting BetYard ML Backend..." -ForegroundColor Green
Write-Host "=" -repeat 40 -ForegroundColor Green

# Change to ML backend directory
$mlBackendPath = "c:\Users\david\OneDrive\Workspace\Python Projects\UI.UX\ml-backend"
Set-Location $mlBackendPath

# Check if virtual environment exists
if (!(Test-Path "venv")) {
    Write-Host "📦 Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install dependencies if needed
if (!(Test-Path "qb_model.pkl")) {
    Write-Host "🔧 First time setup - installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "🧠 Training initial XGBoost model..." -ForegroundColor Yellow
}

# Start the Flask API server
Write-Host "🚀 Starting ML Backend API Server..." -ForegroundColor Green
Write-Host "   API URL: http://localhost:5000" -ForegroundColor Cyan
Write-Host "   Health Check: http://localhost:5000/health" -ForegroundColor Cyan
Write-Host "   Model Info: http://localhost:5000/model/info" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python app.py