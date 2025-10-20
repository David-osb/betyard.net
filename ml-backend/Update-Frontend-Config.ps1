# Update Frontend ML Config After Render Deployment
# Run this script after getting your Render URL

param(
    [Parameter(Mandatory=$true)]
    [string]$RenderUrl,
    
    [Parameter(Mandatory=$false)]
    [string]$ConfigFile = "assets\js\ml-config.js"
)

Write-Host "🔧 Updating ML Backend Configuration" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Validate URL
if ($RenderUrl -notmatch '^https?://') {
    Write-Host "❌ Invalid URL format. Must start with http:// or https://" -ForegroundColor Red
    Write-Host "   Example: https://betyard-ml-backend.onrender.com" -ForegroundColor Yellow
    exit 1
}

# Remove trailing slash if present
$RenderUrl = $RenderUrl.TrimEnd('/')

Write-Host "📝 Configuration Details:" -ForegroundColor Yellow
Write-Host "   Render URL: $RenderUrl" -ForegroundColor Gray
Write-Host "   Config File: $ConfigFile" -ForegroundColor Gray
Write-Host ""

# Check if file exists
if (-not (Test-Path $ConfigFile)) {
    Write-Host "❌ Config file not found: $ConfigFile" -ForegroundColor Red
    exit 1
}

# Read the file
$content = Get-Content $ConfigFile -Raw

# Update the RENDER URL
$content = $content -replace "RENDER: '[^']*'", "RENDER: '$RenderUrl'"

# Update ACTIVE to RENDER
$content = $content -replace "ACTIVE: '[^']*'", "ACTIVE: 'RENDER'"

# Write back to file
Set-Content $ConfigFile -Value $content -NoNewline

Write-Host "✅ Configuration Updated!" -ForegroundColor Green
Write-Host ""
Write-Host "Changes made:" -ForegroundColor Yellow
Write-Host "   ✓ RENDER URL set to: $RenderUrl" -ForegroundColor Gray
Write-Host "   ✓ ACTIVE changed to: RENDER" -ForegroundColor Gray
Write-Host ""

# Show the updated config section
Write-Host "Updated configuration:" -ForegroundColor Yellow
$configSection = $content -match '(?s)const ML_CONFIG = \{.*?\};'
if ($Matches) {
    Write-Host $Matches[0] -ForegroundColor Gray
}
Write-Host ""

# Test the URL
Write-Host "🧪 Testing Render URL..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "$RenderUrl/health" -Method GET -TimeoutSec 30
    
    if ($response.status -eq "healthy") {
        Write-Host "✅ Render Backend is ONLINE and HEALTHY!" -ForegroundColor Green
        Write-Host ""
        Write-Host "   Models Loaded:" -ForegroundColor Gray
        $response.models_loaded.PSObject.Properties | ForEach-Object {
            $status = if ($_.Value) { "✅" } else { "❌" }
            Write-Host "      $status $($_.Name): $($_.Value)" -ForegroundColor Gray
        }
    } else {
        Write-Host "⚠️ Backend responding but status: $($response.status)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Could not connect to Render backend!" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "   Possible causes:" -ForegroundColor Yellow
    Write-Host "   1. Service is still deploying (wait 2-3 more minutes)" -ForegroundColor Gray
    Write-Host "   2. Service is sleeping (first request takes 10-15 seconds)" -ForegroundColor Gray
    Write-Host "   3. URL is incorrect (check Render dashboard)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "   Try again in a minute or check Render dashboard logs." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "✅ Configuration Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Review the changes above" -ForegroundColor Gray
Write-Host "2. Commit and push:" -ForegroundColor Gray
Write-Host "   git add $ConfigFile" -ForegroundColor DarkGray
Write-Host "   git commit -m 'feat: Connect frontend to Render ML backend'" -ForegroundColor DarkGray
Write-Host "   git push origin main" -ForegroundColor DarkGray
Write-Host ""
Write-Host "3. Wait 2 minutes for deployment" -ForegroundColor Gray
Write-Host "4. Test on live site: https://betyard.net" -ForegroundColor Gray
Write-Host "5. Open browser console (F12) to verify connection" -ForegroundColor Gray
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
