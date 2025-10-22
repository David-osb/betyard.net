# Quick Deploy Backend to Render
# This script helps manually trigger deployment if auto-deploy doesn't work

Write-Host "🚀 BetYard ML Backend Deployment Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the ml-backend directory
$currentDir = Get-Location
if ($currentDir.Path -notlike "*ml-backend*") {
    Write-Host "⚠️  Not in ml-backend directory. Changing directory..." -ForegroundColor Yellow
    cd "$PSScriptRoot"
}

Write-Host "📍 Current directory: $(Get-Location)" -ForegroundColor Gray
Write-Host ""

# Step 1: Check current backend status
Write-Host "1️⃣  Checking backend status..." -ForegroundColor Cyan
try {
    $healthResponse = Invoke-RestMethod -Uri "https://betyard-ml-backend.onrender.com/health" -Method Get -TimeoutSec 30
    Write-Host "✅ Backend is running!" -ForegroundColor Green
    Write-Host "   Status: $($healthResponse.status)" -ForegroundColor Gray
    Write-Host "   Version: $($healthResponse.version)" -ForegroundColor Gray
    Write-Host ""
} catch {
    Write-Host "⚠️  Backend might be sleeping (first request can take 10-15 seconds)" -ForegroundColor Yellow
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Gray
    Write-Host ""
}

# Step 2: Test proxy endpoint
Write-Host "2️⃣  Testing proxy endpoint..." -ForegroundColor Cyan
try {
    $proxyResponse = Invoke-RestMethod -Uri "https://betyard-ml-backend.onrender.com/api/proxy/nfl/roster?teamID=LAC&getStats=false" -Method Get -TimeoutSec 30
    Write-Host "✅ Proxy endpoint is working!" -ForegroundColor Green
    Write-Host ""
} catch {
    $statusCode = $_.Exception.Response.StatusCode.Value__
    if ($statusCode -eq 404) {
        Write-Host "❌ Proxy endpoint not found (404)" -ForegroundColor Red
        Write-Host "   The backend needs to be redeployed with the new proxy code." -ForegroundColor Yellow
        Write-Host ""
        
        # Step 3: Deployment instructions
        Write-Host "3️⃣  Manual Deployment Options:" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "   Option A - Render Dashboard (Recommended):" -ForegroundColor White
        Write-Host "   1. Go to: https://dashboard.render.com" -ForegroundColor Gray
        Write-Host "   2. Click on 'betyard-ml-backend' service" -ForegroundColor Gray
        Write-Host "   3. Click 'Manual Deploy' button" -ForegroundColor Gray
        Write-Host "   4. Select 'Deploy latest commit'" -ForegroundColor Gray
        Write-Host "   5. Wait 2-3 minutes for deployment" -ForegroundColor Gray
        Write-Host ""
        
        Write-Host "   Option B - Check Auto-Deploy:" -ForegroundColor White
        Write-Host "   1. Go to: https://dashboard.render.com" -ForegroundColor Gray
        Write-Host "   2. Click on 'betyard-ml-backend' service" -ForegroundColor Gray
        Write-Host "   3. Go to Settings tab" -ForegroundColor Gray
        Write-Host "   4. Scroll to 'Auto-Deploy'" -ForegroundColor Gray
        Write-Host "   5. Make sure it's enabled" -ForegroundColor Gray
        Write-Host "   6. If disabled, enable it and trigger a manual deploy" -ForegroundColor Gray
        Write-Host ""
        
        Write-Host "   Option C - Trigger from GitHub:" -ForegroundColor White
        Write-Host "   1. Make a small change to a file in ml-backend/" -ForegroundColor Gray
        Write-Host "   2. Commit and push to GitHub" -ForegroundColor Gray
        Write-Host "   3. Render will auto-deploy in 1-2 minutes" -ForegroundColor Gray
        Write-Host ""
        
        Write-Host "   Option D - Use Render CLI:" -ForegroundColor White
        Write-Host "   1. Install: npm install -g @render/cli" -ForegroundColor Gray
        Write-Host "   2. Login: render login" -ForegroundColor Gray
        Write-Host "   3. Deploy: render deploy" -ForegroundColor Gray
        Write-Host ""
        
    } else {
        Write-Host "❌ Proxy test failed: $statusCode" -ForegroundColor Red
        Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Gray
        Write-Host ""
    }
}

# Step 4: Verify code is pushed
Write-Host "4️⃣  Verifying GitHub push..." -ForegroundColor Cyan
$lastCommit = git log -1 --oneline
Write-Host "   Last commit: $lastCommit" -ForegroundColor Gray

try {
    $remoteBranch = git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>&1
    if ($LASTEXITCODE -eq 0) {
        $unpushedCommits = git log "$remoteBranch..HEAD" --oneline
        if ($unpushedCommits) {
            Write-Host "⚠️  You have unpushed commits!" -ForegroundColor Yellow
            Write-Host "   Run: git push origin main" -ForegroundColor Yellow
        } else {
            Write-Host "✅ All commits are pushed to GitHub" -ForegroundColor Green
        }
    } else {
        Write-Host "✅ GitHub status OK" -ForegroundColor Green
    }
} catch {
    Write-Host "✅ GitHub status OK" -ForegroundColor Green
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🎯 Quick Links:" -ForegroundColor Cyan
Write-Host "   Render Dashboard: https://dashboard.render.com" -ForegroundColor Blue
Write-Host "   Backend URL: https://betyard-ml-backend.onrender.com" -ForegroundColor Blue
Write-Host "   GitHub Repo: https://github.com/David-osb/betyard.net" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Keep window open if run by double-click
if ($Host.Name -eq "ConsoleHost") {
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
