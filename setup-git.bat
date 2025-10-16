@echo off
title BetYard.net Git Setup
color 0E
echo.
echo 🔧 BetYard.net Git Repository Setup
echo ==================================
echo.
echo This will connect your local files to your GitHub repository
echo for instant updates to betyard.net
echo.
set /p username="Enter your GitHub username: "
set /p repo="Enter repository name (default: betyard-net): "

if "%repo%"=="" set repo=betyard-net

echo.
echo 🔄 Setting up Git connection...

REM Initialize git if not already done
if not exist ".git" (
    echo 📁 Initializing Git repository...
    git init
    git branch -M main
)

REM Add remote origin
echo 🌐 Adding GitHub remote...
git remote remove origin 2>nul
git remote add origin https://github.com/%username%/%repo%.git

REM Add all files
echo 📋 Adding files to Git...
git add .

REM Initial commit
echo 💾 Creating initial commit...
git commit -m "Initial BetYard.net setup - NFL QB Predictor live!"

REM Push to GitHub
echo 🚀 Pushing to GitHub...
git push -u origin main

echo.
echo ✅ SUCCESS! Git setup complete!
echo.
echo 🎯 Your workflow is now:
echo 1. Make changes to files
echo 2. Run 'quick-update.bat'
echo 3. Changes go live at betyard.net in 30-60 seconds!
echo.
echo 🔄 Quick commands:
echo - quick-update.bat : Push changes live
echo - dev-tools.bat   : Development dashboard
echo.
echo 🏈 BetYard.net is ready for continuous improvements!
echo.
pause