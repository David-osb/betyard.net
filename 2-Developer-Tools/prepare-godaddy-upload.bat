@echo off
echo 🌐 Creating GoDaddy Upload Package for BetYard.net
echo ================================================
echo.

REM Create a ZIP-ready folder
echo 📦 Preparing files for GoDaddy upload...
if not exist "godaddy-upload" mkdir godaddy-upload

REM Copy deployment files to upload folder
echo 📋 Copying website files...
copy "betyard-deployment\index.html" "godaddy-upload\" >nul
copy "betyard-deployment\nfl-qb-predictor.html" "godaddy-upload\" >nul
copy "betyard-deployment\robots.txt" "godaddy-upload\" >nul
copy "betyard-deployment\sitemap.xml" "godaddy-upload\" >nul

REM Create assets directory
if not exist "godaddy-upload\assets" mkdir godaddy-upload\assets

REM Create upload instructions
echo 🏈 BetYard.net - GoDaddy Upload Instructions > godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo ============================================= >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo. >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo 1. Log into your GoDaddy account >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo 2. Go to My Products ^> Web Hosting ^> Manage >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo 3. Open File Manager >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo 4. Navigate to public_html folder >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo 5. DELETE any existing files >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo 6. UPLOAD all files from this folder >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo 7. Visit https://betyard.net to test >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo. >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo Files to upload: >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo - index.html (Homepage) >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo - nfl-qb-predictor.html (NFL App) >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo - robots.txt (SEO) >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo - sitemap.xml (Search Engines) >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt
echo - assets/ (Images folder) >> godaddy-upload\UPLOAD_INSTRUCTIONS.txt

echo.
echo ✅ GoDaddy upload package ready!
echo.
echo 📁 Folder: godaddy-upload\
echo 🌐 Contains: All files needed for betyard.net
echo 📋 Instructions: UPLOAD_INSTRUCTIONS.txt included
echo.
echo 🚀 Next steps:
echo 1. Go to your GoDaddy cPanel File Manager
echo 2. Navigate to public_html folder
echo 3. Upload all files from 'godaddy-upload' folder
echo 4. Test https://betyard.net
echo.
echo 🏆 Your NFL analysis platform will be live!
echo.
pause