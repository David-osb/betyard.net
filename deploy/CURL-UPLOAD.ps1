# Ultra-Fast FTP Upload using CURL
# Run this for instant deployment

Write-Host "⚡ ULTRA-FAST FTP UPLOAD WITH CURL" -ForegroundColor Green

$username = Read-Host "GoDaddy FTP Username"
$password = Read-Host "GoDaddy FTP Password" -AsSecureString
$plainPassword = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($password))

$localPath = "C:\Users\david\OneDrive\Workspace\Python Projects\UI.UX\2-Developer-Tools\godaddy-upload"

Write-Host "🚀 Uploading index.html..." -ForegroundColor Cyan
& curl -T "$localPath\index.html" "ftp://ftp.betyard.net/public_html/" --user "${username}:${plainPassword}"

Write-Host "📦 Uploading JavaScript files..." -ForegroundColor Cyan
& curl -T "$localPath\assets\js\game-centric-ui.js" "ftp://ftp.betyard.net/public_html/assets/js/" --user "${username}:${plainPassword}"
& curl -T "$localPath\assets\js\live-scores.js" "ftp://ftp.betyard.net/public_html/assets/js/" --user "${username}:${plainPassword}"
& curl -T "$localPath\assets\js\ml-config.js" "ftp://ftp.betyard.net/public_html/assets/js/" --user "${username}:${plainPassword}"
& curl -T "$localPath\assets\js\ml-integration.js" "ftp://ftp.betyard.net/public_html/assets/js/" --user "${username}:${plainPassword}"
& curl -T "$localPath\assets\js\nfl-schedule-2025-correct.js" "ftp://ftp.betyard.net/public_html/assets/js/" --user "${username}:${plainPassword}"
& curl -T "$localPath\assets\js\nfl-schedule-api.js" "ftp://ftp.betyard.net/public_html/assets/js/" --user "${username}:${plainPassword}"
& curl -T "$localPath\assets\js\nfl-schedule.js" "ftp://ftp.betyard.net/public_html/assets/js/" --user "${username}:${plainPassword}"

Write-Host "🎉 DONE! Advanced Analytics are LIVE!" -ForegroundColor Green