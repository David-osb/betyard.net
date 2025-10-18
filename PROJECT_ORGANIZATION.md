# 🏈 BetYard.net Project Organization

## 📁 Folder Structure Overview

Your project has been organized into 4 main categories for better maintainability and clarity:

### 🌐 **1-Website-Core** (ESSENTIAL - Keep All)
**Required for your live website at betyard.net**
```
1-Website-Core/
├── index.html              ✅ Homepage
├── nfl-qb-predictor.html   ✅ Main NFL application
├── robots.txt              ✅ SEO - Search engine instructions
├── sitemap.xml             ✅ SEO - Site structure for search engines
├── CNAME                   ✅ GitHub Pages custom domain configuration
└── assets/                 ✅ JavaScript modules, images, styles
    └── js/
        ├── https-enforcement.js
        ├── xgboost-simulation.js
        ├── syntax-validation.js
        └── main.js
```

### 🛠️ **2-Developer-Tools** (USEFUL - Keep Most)
**Tools for maintaining and updating your website**
```
2-Developer-Tools/
├── quick-update.bat        ⭐ VERY USEFUL - Deploy updates to live site
├── deploy.bat              📦 Deployment automation
├── deploy.sh               📦 Linux deployment script
├── dev-tools.bat           🔧 Development dashboard
├── connect-github.bat      🔗 Git setup (one-time use)
├── prepare-godaddy-upload.bat 📤 Upload preparation
├── betyard-deployment/     📁 Deployment folder with copies
└── godaddy-upload/         📁 GoDaddy upload folder
```

### 🐛 **3-Debugging-Tools** (OPTIONAL - Keep if Needed)
**Testing, troubleshooting, and diagnostic tools**
```
3-Debugging-Tools/
├── advanced-js-check.ps1           🔍 JavaScript syntax checking
├── js-syntax-check.ps1             🔍 JavaScript validation
├── domain-status-check.ps1         🌐 Domain connectivity testing
├── dns-check.bat                   🌐 DNS troubleshooting
├── check-godaddy-account.bat       🏢 GoDaddy account verification
├── dark-theme-test.html            🎨 Theme testing
├── phone-access.html               📱 Mobile testing
├── nfl-qb-predictor-backup-*.html  💾 Code backups
├── syntax-test.html                🧪 Syntax testing
├── start_server.bat                🖥️ Local server (basic)
├── ssl-fix.bat                     🔒 SSL troubleshooting
└── *Setup_Instructions.md          📖 API setup guides
```

### 🗑️ **4-Unnecessary-Files** (OPTIONAL - Can Delete)
**Documentation, completed setup guides, and reference files**
```
4-Unnecessary-Files/
├── README.md                       📖 Basic project description
├── DEPLOYMENT_SUMMARY.md           📋 Completed deployment log
├── MODULARIZATION_SUMMARY.md       📋 Code organization log
├── LIVE_UPDATE_SYSTEM.md           📋 Update system documentation
├── Error_Fix_Summary.md            🐛 Historical error fixes
├── DNS_FIX_GUIDE.md               🌐 DNS troubleshooting guide
├── GODADDY_*.md                   🏢 GoDaddy setup guides
├── DOMAIN_*.md                    🌐 Domain setup guides
├── HTTPS_ENFORCEMENT_FIX.md       🔒 SSL setup guide
├── github-pages-setup.html        📋 GitHub Pages setup
├── godaddy-step-by-step.html      📋 GoDaddy setup guide
├── live-update-dashboard.html     📊 Update dashboard
└── NFL API Data.html              📊 API reference data
```

## 🎯 Recommendations by Category

### ✅ **MUST KEEP** - 1-Website-Core
- **ALL files in this folder are essential**
- Required for betyard.net to function
- Never delete these files

### ⭐ **HIGHLY RECOMMENDED** - 2-Developer-Tools  
- `quick-update.bat` - Your most valuable tool for updates
- `betyard-deployment/` - Backup deployment folder
- Keep most files for ongoing maintenance

### 🔧 **KEEP IF YOU DEBUG** - 3-Debugging-Tools
- Keep if you do local testing and troubleshooting
- Delete if you only work directly on live site
- Backup files are safe to delete after confirming current version works

### 🗑️ **SAFE TO DELETE** - 4-Unnecessary-Files
- Documentation and completed setup guides
- Can be deleted to reduce clutter
- No impact on website functionality

## 🚀 Quick Actions

### For Clean Workspace:
```powershell
# Delete unnecessary files (optional)
Remove-Item "4-Unnecessary-Files" -Recurse -Force

# Keep only essential debugging (optional)
Remove-Item "3-Debugging-Tools\*backup*" -Force
```

### For Active Development:
- Keep all folders
- Use `2-Developer-Tools\quick-update.bat` for updates
- Use `3-Debugging-Tools\*-check.ps1` for testing

## 📊 Storage Impact
- **Website-Core**: ~2-3 MB (essential)
- **Developer-Tools**: ~5-10 MB (very useful)
- **Debugging-Tools**: ~15-20 MB (optional)
- **Unnecessary-Files**: ~5-10 MB (can delete)

---
*Organization completed: October 18, 2025*
*Live website: https://betyard.net*