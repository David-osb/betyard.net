@echo off
echo 🏈 BetYard.net Deployment Preparation
echo =====================================
echo.

REM Create deployment folder
echo 📁 Creating deployment folder...
if not exist "betyard-deployment" mkdir betyard-deployment
if not exist "betyard-deployment\assets" mkdir betyard-deployment\assets

REM Copy essential files
echo 📋 Copying files for deployment...
copy "index.html" "betyard-deployment\" >nul
copy "nfl-qb-predictor.html" "betyard-deployment\" >nul
copy "simple_server.py" "betyard-deployment\" >nul
copy "Start-Server.ps1" "betyard-deployment\" >nul
copy "start_server.bat" "betyard-deployment\" >nul

REM Create CNAME file for GitHub Pages
echo betyard.net > betyard-deployment\CNAME

REM Create robots.txt
echo User-agent: * > betyard-deployment\robots.txt
echo Allow: / >> betyard-deployment\robots.txt
echo. >> betyard-deployment\robots.txt
echo Sitemap: https://betyard.net/sitemap.xml >> betyard-deployment\robots.txt

REM Create basic sitemap.xml
echo ^<?xml version="1.0" encoding="UTF-8"?^> > betyard-deployment\sitemap.xml
echo ^<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"^> >> betyard-deployment\sitemap.xml
echo   ^<url^> >> betyard-deployment\sitemap.xml
echo     ^<loc^>https://betyard.net/^</loc^> >> betyard-deployment\sitemap.xml
echo     ^<lastmod^>%date:~10,4%-%date:~4,2%-%date:~7,2%^</lastmod^> >> betyard-deployment\sitemap.xml
echo     ^<changefreq^>weekly^</changefreq^> >> betyard-deployment\sitemap.xml
echo     ^<priority^>1.0^</priority^> >> betyard-deployment\sitemap.xml
echo   ^</url^> >> betyard-deployment\sitemap.xml
echo   ^<url^> >> betyard-deployment\sitemap.xml
echo     ^<loc^>https://betyard.net/nfl-qb-predictor.html^</loc^> >> betyard-deployment\sitemap.xml
echo     ^<lastmod^>%date:~10,4%-%date:~4,2%-%date:~7,2%^</lastmod^> >> betyard-deployment\sitemap.xml
echo     ^<changefreq^>daily^</changefreq^> >> betyard-deployment\sitemap.xml
echo     ^<priority^>0.9^</priority^> >> betyard-deployment\sitemap.xml
echo   ^</url^> >> betyard-deployment\sitemap.xml
echo ^</urlset^> >> betyard-deployment\sitemap.xml

echo.
echo ✅ Deployment files ready in 'betyard-deployment' folder
echo.
echo 🌐 Next steps:
echo 1. Upload 'betyard-deployment' folder contents to your hosting
echo 2. Configure DNS to point betyard.net to your hosting
echo 3. Ensure SSL certificate is active
echo 4. Test all functionality
echo.
echo 🎯 Your professional NFL analysis platform is ready!
echo.
pause