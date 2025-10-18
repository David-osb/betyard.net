# 🚀 AUTOMATED CLOUD DEPLOYMENT
# This script prepares and deploys your ML backend to Railway

Write-Host "🚀 BetYard ML Backend - Cloud Deployment Automation" -ForegroundColor Green
Write-Host "=" * 60

# Step 1: Create GitHub repository for ML backend
Write-Host "📂 Step 1: Creating separate ML backend repository..." -ForegroundColor Cyan

$mlRepo = "betyard-ml-backend"
Write-Host "Creating GitHub repo: $mlRepo" -ForegroundColor Yellow

# Initialize git in ml-backend directory
Push-Location ml-backend
git init
git add .
git commit -m "🚀 Initial ML backend for cloud deployment"
git branch -M main

Write-Host "✅ Git repository initialized" -ForegroundColor Green

# Step 2: Instructions for GitHub
Write-Host ""
Write-Host "📋 Step 2: Create GitHub Repository" -ForegroundColor Cyan
Write-Host "1. Go to https://github.com/new" -ForegroundColor Yellow
Write-Host "2. Repository name: $mlRepo" -ForegroundColor Yellow
Write-Host "3. Make it Public" -ForegroundColor Yellow
Write-Host "4. Click 'Create repository'" -ForegroundColor Yellow
Write-Host ""
Write-Host "🔗 Then run this command:" -ForegroundColor Magenta
Write-Host "git remote add origin https://github.com/David-osb/$mlRepo.git" -ForegroundColor White
Write-Host "git push -u origin main" -ForegroundColor White

Write-Host ""
Write-Host "📋 Step 3: Deploy to Railway" -ForegroundColor Cyan
Write-Host "1. Go to https://railway.app" -ForegroundColor Yellow
Write-Host "2. Click 'Start a New Project'" -ForegroundColor Yellow
Write-Host "3. Select 'Deploy from GitHub repo'" -ForegroundColor Yellow
Write-Host "4. Choose your $mlRepo repository" -ForegroundColor Yellow
Write-Host "5. Railway will auto-deploy! 🎉" -ForegroundColor Yellow

Write-Host ""
Write-Host "📋 Step 4: Update Frontend" -ForegroundColor Cyan
Write-Host "1. Copy your Railway URL (like: https://$mlRepo-production.up.railway.app)" -ForegroundColor Yellow
Write-Host "2. Update PRODUCTION_URL in assets/js/ml-integration.js" -ForegroundColor Yellow
Write-Host "3. Commit and push to deploy live!" -ForegroundColor Yellow

Write-Host ""
Write-Host "🎯 Alternative: One-Click Deploy Templates" -ForegroundColor Magenta
Write-Host "• Railway: https://railway.app/new/template" -ForegroundColor White
Write-Host "• Render: https://render.com/docs/deploy-flask" -ForegroundColor White  
Write-Host "• Heroku: https://devcenter.heroku.com/articles/getting-started-with-python" -ForegroundColor White

Pop-Location

Write-Host ""
Write-Host "✅ Cloud deployment preparation complete!" -ForegroundColor Green
Write-Host "🚀 Your ML backend is ready for the cloud!" -ForegroundColor Cyan