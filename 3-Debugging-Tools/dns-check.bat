@echo off
echo.
echo ╔═══════════════════════════════════════╗
echo ║      🔧 BetYard.net DNS Checker       ║
echo ╚═══════════════════════════════════════╝
echo.

echo 🔍 Checking current DNS configuration...
echo.

echo ⚡ Testing betyard.net DNS:
nslookup betyard.net 8.8.8.8
echo.

echo 📊 Expected Result (GitHub Pages):
echo ✅ Should show EXACTLY 4 IP addresses:
echo    185.199.108.153
echo    185.199.109.153
echo    185.199.110.153  
echo    185.199.111.153
echo.
echo ❌ Should NOT show: 160.153.0.50
echo.

echo 🧪 Testing GitHub Pages directly:
nslookup david-osb.github.io 8.8.8.8
echo.

echo 🎯 Next Steps:
echo 1. If you see 5 IPs → Go to GoDaddy DNS, remove 160.153.0.50
echo 2. If you see 4 IPs → DNS is correct, wait for SSL activation
echo 3. Check again in 15 minutes after DNS changes
echo.

echo 📱 Quick Links:
echo GoDaddy DNS: https://dcc.godaddy.com/control/portfolio
echo GitHub Pages: https://github.com/David-osb/betyard.net/settings/pages
echo.

pause