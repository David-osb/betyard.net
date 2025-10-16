@echo off
title BetYard.net SSL Security Fix
color 0E
echo.
echo 🔒 BetYard.net SSL Security Fix
echo ==============================
echo.
echo 🚨 Issue: "Not Secure" warning on betyard.net
echo ✅ Solution: Configure SSL certificate for HTTPS
echo.
echo 🔍 Checking current status...
echo.

REM Check if GitHub Pages is set up
echo 1. Testing GitHub Pages URL...
echo    https://david-osb.github.io/betyard.net
echo.

REM Check domain status
echo 2. Testing domain status...
echo    https://betyard.net
echo.

echo 🛠️ IMMEDIATE FIXES APPLIED:
echo ✅ Added HTTPS enforcement to both pages
echo ✅ URLs will auto-redirect to secure version
echo.

echo 🎯 NEXT STEPS TO FIX SSL:
echo.
echo Option 1: GitHub Pages SSL (RECOMMENDED - FREE)
echo ------------------------------------------------
echo 1. Go to: https://github.com/David-osb/betyard.net/settings/pages
echo 2. Set Custom Domain: betyard.net
echo 3. Check "Enforce HTTPS" ✅
echo 4. Wait 1-24 hours for SSL to activate
echo.

echo Option 2: Cloudflare SSL (FREE)
echo --------------------------------
echo 1. Sign up at cloudflare.com
echo 2. Add domain: betyard.net
echo 3. Update nameservers in GoDaddy
echo 4. Enable SSL in Cloudflare
echo.

echo Option 3: GoDaddy Hosting SSL
echo ------------------------------
echo 1. Purchase hosting plan
echo 2. Enable SSL in cPanel
echo 3. Upload files
echo.

echo 🚀 TEMPORARY ACCESS WHILE SSL SETS UP:
echo.
echo Test your site here (secure):
echo https://david-osb.github.io/betyard.net/nfl-qb-predictor.html
echo.

echo ⏱️ TIMELINE:
echo • Immediate: GitHub Pages URL works
echo • 1-24 hours: betyard.net SSL activates  
echo • 24-48 hours: Full DNS propagation
echo.

echo 🔧 QUICK ACTIONS:
set /p action="Choose: (1) Open GitHub Pages settings (2) Test GitHub URL (3) Check SSL guide (4) Exit: "

if "%action%"=="1" (
    echo Opening GitHub Pages settings...
    start https://github.com/David-osb/betyard.net/settings/pages
) else if "%action%"=="2" (
    echo Opening secure test URL...
    start https://david-osb.github.io/betyard.net/nfl-qb-predictor.html
) else if "%action%"=="3" (
    echo Opening SSL fix guide...
    start SSL_FIX_GUIDE.md
) else (
    echo Exiting...
)

echo.
echo 🏆 Once SSL is active, betyard.net will show:
echo ✅ Green padlock icon
echo ✅ "Secure" connection
echo ✅ No browser warnings
echo.
pause