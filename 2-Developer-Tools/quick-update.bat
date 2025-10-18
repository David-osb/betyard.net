@echo off
title BetYard.net Quick Update Tool
color 0A
echo.
echo 🚀 BetYard.net Quick Update System
echo ================================
echo.
echo Current Status: ✅ Live at betyard.net
echo.
echo What would you like to update?
echo.
echo 1. 🏈 NFL Data/Stats
echo 2. 🎨 Design/Layout  
echo 3. 📱 Mobile Experience
echo 4. ⚡ Performance
echo 5. 🔧 Bug Fix
echo 6. ✨ New Feature
echo 7. 📊 Analytics
echo 8. 🔒 Security
echo 9. 📝 Content Update
echo.
set /p choice="Enter choice (1-9): "

echo.
set /p message="Describe your update: "

echo.
echo 📁 Preparing update...
git add .

echo 🔄 Committing changes...
if "%choice%"=="1" (
    git commit -m "data: %message%"
) else if "%choice%"=="2" (
    git commit -m "style: %message%"
) else if "%choice%"=="3" (
    git commit -m "mobile: %message%"
) else if "%choice%"=="4" (
    git commit -m "perf: %message%"
) else if "%choice%"=="5" (
    git commit -m "fix: %message%"
) else if "%choice%"=="6" (
    git commit -m "feat: %message%"
) else if "%choice%"=="7" (
    git commit -m "analytics: %message%"
) else if "%choice%"=="8" (
    git commit -m "security: %message%"
) else if "%choice%"=="9" (
    git commit -m "content: %message%"
) else (
    git commit -m "update: %message%"
)

echo 🌐 Pushing to betyard.net...
git push origin main

echo.
echo ✅ SUCCESS! Your update is live at betyard.net
echo ⏱️  Changes will appear in 30-60 seconds
echo.
echo 🎯 Pro tip: Check https://betyard.net to verify changes
echo.
pause