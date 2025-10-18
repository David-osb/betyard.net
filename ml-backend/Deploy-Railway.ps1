# 🚀 BetYard ML Backend - Railway Deployment Setup
Write-Host "🚀 BetYard ML Backend - Railway Deployment Setup" -ForegroundColor Green
Write-Host "=" * 50

# Check if Railway CLI is installed
if (!(Get-Command railway -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Railway CLI not found. Installing..." -ForegroundColor Yellow
    npm install -g @railway/cli
    Write-Host "✅ Railway CLI installed!" -ForegroundColor Green
}

# Login check
Write-Host "🔐 Checking Railway authentication..." -ForegroundColor Cyan
$loginCheck = railway whoami 2>$null
if (!$loginCheck) {
    Write-Host "🔑 Please login to Railway:" -ForegroundColor Yellow
    railway login
}

# Initialize project
Write-Host "📦 Initializing Railway project..." -ForegroundColor Cyan
railway init

# Deploy
Write-Host "🚀 Deploying ML Backend to Railway..." -ForegroundColor Green
railway up

Write-Host "=" * 50
Write-Host "✅ Deployment Complete!" -ForegroundColor Green
Write-Host "📡 Your ML Backend is now live on Railway" -ForegroundColor Cyan
Write-Host "🔗 Check your Railway dashboard for the URL" -ForegroundColor Cyan
Write-Host "💡 Update frontend ml-integration.js with your new URL" -ForegroundColor Yellow
Write-Host "=" * 50

# Open Railway dashboard
Write-Host "🌐 Opening Railway dashboard..." -ForegroundColor Cyan
Start-Process "https://railway.app/dashboard"