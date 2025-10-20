# Test Render Deployment Script
# Run this after deploying to Render to verify everything works

param(
    [Parameter(Mandatory=$true)]
    [string]$RenderUrl
)

Write-Host "🧪 Testing Render Deployment: $RenderUrl" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Health Check
Write-Host "Test 1: Health Check" -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$RenderUrl/health" -Method GET
    Write-Host "✅ Health Check: " -NoNewline -ForegroundColor Green
    Write-Host "Status=$($health.status)"
    Write-Host "   Models Loaded:" -ForegroundColor Gray
    $health.models_loaded.PSObject.Properties | ForEach-Object {
        $status = if ($_.Value) { "✅" } else { "❌" }
        Write-Host "      $status $($_.Name): $($_.Value)" -ForegroundColor Gray
    }
} catch {
    Write-Host "❌ Health Check FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Test 2: QB Prediction
Write-Host "Test 2: QB Prediction" -ForegroundColor Yellow
try {
    $qbBody = @{
        player_name = "Patrick Mahomes"
        team_code = "KC"
        opponent_code = "LV"
        position = "QB"
    } | ConvertTo-Json

    $qbResult = Invoke-RestMethod -Uri "$RenderUrl/predict" -Method POST -Body $qbBody -ContentType "application/json"
    Write-Host "✅ QB Prediction:" -ForegroundColor Green
    Write-Host "   Passing Yards: $($qbResult.passing_yards)" -ForegroundColor Gray
    Write-Host "   Passing TDs: $($qbResult.passing_touchdowns)" -ForegroundColor Gray
    Write-Host "   Confidence: $($qbResult.confidence_score)%" -ForegroundColor Gray
} catch {
    Write-Host "❌ QB Prediction FAILED: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 3: RB Prediction
Write-Host "Test 3: RB Prediction" -ForegroundColor Yellow
try {
    $rbBody = @{
        player_name = "Travis Etienne"
        team_code = "JAX"
        opponent_code = "TEN"
        position = "RB"
    } | ConvertTo-Json

    $rbResult = Invoke-RestMethod -Uri "$RenderUrl/predict" -Method POST -Body $rbBody -ContentType "application/json"
    Write-Host "✅ RB Prediction:" -ForegroundColor Green
    Write-Host "   Rushing Yards: $($rbResult.rushing_yards)" -ForegroundColor Gray
    Write-Host "   Rushing Attempts: $($rbResult.rushing_attempts)" -ForegroundColor Gray
    Write-Host "   Rushing TDs: $($rbResult.rushing_touchdowns)" -ForegroundColor Gray
    Write-Host "   Confidence: $($rbResult.confidence_score)%" -ForegroundColor Gray
} catch {
    Write-Host "❌ RB Prediction FAILED: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 4: WR Prediction
Write-Host "Test 4: WR Prediction" -ForegroundColor Yellow
try {
    $wrBody = @{
        player_name = "Justin Jefferson"
        team_code = "MIN"
        opponent_code = "DET"
        position = "WR"
    } | ConvertTo-Json

    $wrResult = Invoke-RestMethod -Uri "$RenderUrl/predict" -Method POST -Body $wrBody -ContentType "application/json"
    Write-Host "✅ WR Prediction:" -ForegroundColor Green
    Write-Host "   Receiving Yards: $($wrResult.receiving_yards)" -ForegroundColor Gray
    Write-Host "   Receptions: $($wrResult.receptions)" -ForegroundColor Gray
    Write-Host "   Targets: $($wrResult.targets)" -ForegroundColor Gray
    Write-Host "   Receiving TDs: $($wrResult.receiving_touchdowns)" -ForegroundColor Gray
    Write-Host "   Confidence: $($wrResult.confidence_score)%" -ForegroundColor Gray
} catch {
    Write-Host "❌ WR Prediction FAILED: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 5: TE Prediction
Write-Host "Test 5: TE Prediction" -ForegroundColor Yellow
try {
    $teBody = @{
        player_name = "Travis Kelce"
        team_code = "KC"
        opponent_code = "LV"
        position = "TE"
    } | ConvertTo-Json

    $teResult = Invoke-RestMethod -Uri "$RenderUrl/predict" -Method POST -Body $teBody -ContentType "application/json"
    Write-Host "✅ TE Prediction:" -ForegroundColor Green
    Write-Host "   Receiving Yards: $($teResult.receiving_yards)" -ForegroundColor Gray
    Write-Host "   Receptions: $($teResult.receptions)" -ForegroundColor Gray
    Write-Host "   Targets: $($teResult.targets)" -ForegroundColor Gray
    Write-Host "   Receiving TDs: $($teResult.receiving_touchdowns)" -ForegroundColor Gray
    Write-Host "   Confidence: $($teResult.confidence_score)%" -ForegroundColor Gray
} catch {
    Write-Host "❌ TE Prediction FAILED: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 6: Model Info
Write-Host "Test 6: Model Info" -ForegroundColor Yellow
try {
    $modelInfo = Invoke-RestMethod -Uri "$RenderUrl/model/info" -Method GET
    Write-Host "✅ Model Info Retrieved:" -ForegroundColor Green
    $modelInfo.PSObject.Properties | ForEach-Object {
        Write-Host "   $($_.Name): $($_.Value.n_features) features" -ForegroundColor Gray
    }
} catch {
    Write-Host "❌ Model Info FAILED: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "✅ Deployment Test Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Update ml-config.js with: $RenderUrl" -ForegroundColor Gray
Write-Host "2. Set ACTIVE to 'RENDER'" -ForegroundColor Gray
Write-Host "3. Commit and push changes" -ForegroundColor Gray
Write-Host "4. Test on live site: betyard.net" -ForegroundColor Gray
Write-Host ""
