@echo off
title Connect to GitHub Repository
color 0C
echo.
echo 🔗 Connect Local Files to GitHub Repository
echo ==========================================
echo.
echo Your BetYard.net is live! Now let's connect your local files
echo to GitHub for instant updates.
echo.
echo You'll need:
echo 1. Your GitHub username
echo 2. Your repository name (e.g., betyard-net)
echo.
set /p username="Enter your GitHub username: "
set /p repo="Enter repository name: "

echo.
echo 🔄 Setting up Git connection...

REM Set the main branch
git branch -M main

REM Add remote origin
echo 🌐 Adding GitHub remote...
git remote remove origin 2>nul
git remote add origin https://github.com/%username%/%repo%.git

echo.
echo ✅ Git connection configured!
echo.
echo 🚀 Next steps:
echo 1. Make sure your GitHub repository exists
echo 2. Run 'quick-update.bat' to push changes
echo 3. Your site will update automatically at betyard.net
echo.
echo 💡 If you haven't created the GitHub repository yet:
echo 1. Go to https://github.com
echo 2. Click "New Repository"
echo 3. Name it: %repo%
echo 4. Make it Public
echo 5. Don't initialize with README (your files are already here)
echo.
pause