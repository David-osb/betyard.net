# 🆓 FREE DEPLOYMENT TO RENDER
Write-Host "🆓 FREE ML Backend Deployment to Render.com" -ForegroundColor Green
Write-Host "=" * 50

Write-Host "💡 Why Render?" -ForegroundColor Cyan
Write-Host "✅ Completely FREE forever" -ForegroundColor Green
Write-Host "✅ No credit card required" -ForegroundColor Green  
Write-Host "✅ 750 hours/month (more than enough!)" -ForegroundColor Green
Write-Host "✅ Automatic HTTPS & GitHub integration" -ForegroundColor Green
Write-Host ""

Write-Host "🚀 STEP 1: Create Render Account" -ForegroundColor Yellow
Write-Host "1. Go to https://render.com" -ForegroundColor White
Write-Host "2. Click 'Get Started for Free'" -ForegroundColor White
Write-Host "3. Sign up with GitHub (no credit card needed!)" -ForegroundColor White
Write-Host ""

Write-Host "🚀 STEP 2: Deploy Your ML Backend" -ForegroundColor Yellow
Write-Host "1. Click 'New' → 'Web Service'" -ForegroundColor White
Write-Host "2. Connect your GitHub account" -ForegroundColor White
Write-Host "3. Select repository: betyard.net" -ForegroundColor White
Write-Host "4. Root Directory: ml-backend" -ForegroundColor White
Write-Host "5. Build Command: pip install -r requirements.txt" -ForegroundColor White
Write-Host "6. Start Command: python app.py" -ForegroundColor White
Write-Host "7. Click 'Create Web Service'" -ForegroundColor White
Write-Host ""

Write-Host "🔗 STEP 3: Get Your URL" -ForegroundColor Yellow
Write-Host "Render will provide a URL like:" -ForegroundColor White
Write-Host "https://betyard-ml-backend-XXXX.onrender.com" -ForegroundColor Magenta
Write-Host ""

Write-Host "⚙️ STEP 4: Update Your Website" -ForegroundColor Yellow
Write-Host "1. Copy your Render URL" -ForegroundColor White
Write-Host "2. Update RENDER URL in assets/js/ml-config.js" -ForegroundColor White
Write-Host "3. Ensure ACTIVE: 'RENDER' is set" -ForegroundColor White
Write-Host "4. Commit and push → Live ML predictions!" -ForegroundColor White
Write-Host ""

Write-Host "📊 What You Get FREE:" -ForegroundColor Cyan
Write-Host "• 750 hours/month runtime" -ForegroundColor Green
Write-Host "• Unlimited deployments" -ForegroundColor Green  
Write-Host "• Automatic HTTPS" -ForegroundColor Green
Write-Host "• Custom domains" -ForegroundColor Green
Write-Host "• Auto-deploy on GitHub push" -ForegroundColor Green
Write-Host ""

Write-Host "⚠️ Only Limitation:" -ForegroundColor Yellow
Write-Host "Apps sleep after 15min inactivity (10-15s cold start)" -ForegroundColor White
Write-Host "Perfect for development and most use cases!" -ForegroundColor Green
Write-Host ""

Write-Host "🌐 Opening Render.com..." -ForegroundColor Cyan
Start-Process "https://render.com"

Write-Host ""
Write-Host "🎉 Ready to deploy your FREE ML backend!" -ForegroundColor Green