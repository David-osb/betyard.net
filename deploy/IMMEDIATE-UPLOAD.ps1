# Direct Website Update Script - Push to Live Site Immediately
# This bypasses GitHub Pages and uploads directly to your hosting

Write-Host "🚀 IMMEDIATE UPLOAD TO LIVE WEBSITE" -ForegroundColor Green

# Use WinSCP or similar if available, otherwise use web-based file manager approach
$sourceFolder = "C:\Users\david\OneDrive\Workspace\Python Projects\UI.UX\2-Developer-Tools\godaddy-upload"

Write-Host "📁 Files ready for immediate upload from: $sourceFolder" -ForegroundColor Yellow

# List all files that need to be uploaded
Write-Host "`n📋 FILES TO UPLOAD:" -ForegroundColor Cyan
Write-Host "✅ index.html (with Advanced Analytics)" -ForegroundColor Green
Write-Host "✅ CNAME (domain mapping)" -ForegroundColor Green
Write-Host "✅ assets/js/game-centric-ui.js" -ForegroundColor Green
Write-Host "✅ assets/js/live-scores.js" -ForegroundColor Green
Write-Host "✅ assets/js/ml-config.js" -ForegroundColor Green
Write-Host "✅ assets/js/ml-integration.js" -ForegroundColor Green
Write-Host "✅ assets/js/nfl-schedule-2025-correct.js" -ForegroundColor Green
Write-Host "✅ assets/js/nfl-schedule-api.js" -ForegroundColor Green
Write-Host "✅ assets/js/nfl-schedule.js" -ForegroundColor Green

Write-Host "`n🎯 IMMEDIATE ACTION REQUIRED:" -ForegroundColor Red
Write-Host "1. Open GoDaddy cPanel File Manager" -ForegroundColor White
Write-Host "2. Navigate to public_html folder" -ForegroundColor White
Write-Host "3. Upload the index.html file (overwrite existing)" -ForegroundColor White
Write-Host "4. Upload all files from assets/js/ folder" -ForegroundColor White
Write-Host "5. Clear browser cache and test betyard.net" -ForegroundColor White

Write-Host "`n🌐 OR use FTP client (faster):" -ForegroundColor Yellow
Write-Host "Host: ftp.betyard.net" -ForegroundColor White
Write-Host "Path: /public_html/" -ForegroundColor White
Write-Host "Upload all files from: $sourceFolder" -ForegroundColor White

Write-Host "`n⚡ This will make changes LIVE IMMEDIATELY!" -ForegroundColor Red
Write-Host "🎉 Advanced analytics will be available instantly" -ForegroundColor Green

# Open the source folder for easy access
Start-Process "explorer.exe" -ArgumentList $sourceFolder