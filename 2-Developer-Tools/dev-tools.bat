@echo off
title BetYard.net Development Tools
color 0B
echo.
echo 🛠️  BetYard.net Development Dashboard
echo ===================================
echo.
echo 🌐 Live Site: https://betyard.net
echo 📁 Local Files: %CD%
echo.
echo Choose your action:
echo.
echo 1. 🚀 Quick Update (Push changes live)
echo 2. 🧪 Test Locally (Start local server)
echo 3. 💾 Backup Current Version
echo 4. 📊 Check Site Status
echo 5. 🔄 Sync from GitHub
echo 6. 📝 View Recent Changes
echo 7. 🎯 Performance Check
echo 8. 📱 Mobile Test
echo 9. ✨ New Feature Template
echo.
set /p action="Enter choice (1-9): "

if "%action%"=="1" (
    call quick-update.bat
) else if "%action%"=="2" (
    echo 🧪 Starting local test server...
    echo 📍 Open http://localhost:8000 in your browser
    echo 🛑 Press Ctrl+C to stop server
    python simple_server.py
) else if "%action%"=="3" (
    echo 💾 Creating backup...
    git branch backup-%date:~10,4%%date:~4,2%%date:~7,2%-%time:~0,2%%time:~3,2%
    git push origin backup-%date:~10,4%%date:~4,2%%date:~7,2%-%time:~0,2%%time:~3,2%
    echo ✅ Backup created!
) else if "%action%"=="4" (
    echo 📊 Checking betyard.net status...
    curl -s -o nul -w "Status Code: %%{http_code}\nResponse Time: %%{time_total}s\n" https://betyard.net
) else if "%action%"=="5" (
    echo 🔄 Syncing from GitHub...
    git pull origin main
    echo ✅ Local files updated!
) else if "%action%"=="6" (
    echo 📝 Recent changes:
    git log --oneline -10
) else if "%action%"=="7" (
    echo 🎯 Running performance check...
    echo Opening PageSpeed Insights...
    start https://pagespeed.web.dev/analysis?url=https://betyard.net
) else if "%action%"=="8" (
    echo 📱 Opening mobile testing tools...
    start https://search.google.com/test/mobile-friendly?url=https://betyard.net
) else if "%action%"=="9" (
    echo ✨ What type of feature?
    echo 1. NFL Data Enhancement
    echo 2. UI/UX Improvement  
    echo 3. Performance Optimization
    echo 4. Analytics Integration
    set /p ftype="Feature type (1-4): "
    echo.
    echo 📝 Feature template created in new-feature.md
    echo // TODO: Implement new feature > new-feature.md
    echo // Type: %ftype% >> new-feature.md
    echo // Date: %date% >> new-feature.md
    echo ✅ Template ready for development!
)

echo.
echo 🏈 BetYard.net Development Tools Ready!
pause