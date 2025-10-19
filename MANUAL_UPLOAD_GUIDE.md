# 🚀 URGENT: Manual Upload Guide for NFL Timing Fix

## 🎯 Problem
The live website https://betyard.net still shows Kansas City game as "LIVE" when it should be "SCHEDULED" until 1:00 PM ET.

## ✅ Solution Ready
All files with timing validation are ready in: `godaddy-upload/` folder

## 📁 Files to Upload to GoDaddy

### Required Files with Timing Validation:
- `index.html` (with v=2025-10-19-DEBUG&t=1760886999)
- `assets/js/live-scores.js` (with Eastern Time validation)
- `assets/js/nfl-schedule-api.js` (with Tank01 real data)
- `assets/js/enhanced-api-test.js` (working endpoints only)
- All other JS files in `assets/js/` folder

## 🔧 Manual Upload Steps

### Option 1: GoDaddy cPanel File Manager
1. Login to your GoDaddy hosting account
2. Open cPanel → File Manager
3. Navigate to `public_html` folder
4. Delete old files (backup first!)
5. Upload all files from `godaddy-upload/` folder
6. Ensure folder structure: `public_html/assets/js/` contains all JS files

### Option 2: FTP Client (Recommended)
1. Use FileZilla or similar FTP client
2. Connect to your GoDaddy FTP:
   - Host: ftp.betyard.net (or your assigned FTP server)
   - Username: Your GoDaddy FTP username
   - Password: Your GoDaddy FTP password
3. Upload all `godaddy-upload/*` files to the root directory

### Option 3: GitHub Pages (If Connected)
If your GoDaddy is connected to GitHub:
1. We already pushed to GitHub (commit fdfb84c)
2. Check if auto-deployment is working
3. Force refresh GitHub Pages in your repo settings

## 🧪 Verification Steps
After upload:
1. Visit: https://betyard.net/?cb=1760887999 (cache buster)
2. Open Browser Console (F12)
3. Look for debug messages:
   - `🔍 mapGameStatus called:`
   - `🚨 TIME ZONE OVERRIDE:`
   - `🔄 OVERRIDING: LIVE → SCHEDULED`

## 🚨 Critical Files for Timing Fix
These MUST be uploaded for the Kansas City timing issue to be resolved:

```
index.html                           (Updated script tags)
assets/js/live-scores.js            (mapGameStatus timing validation)
assets/js/nfl-schedule-api.js       (Tank01 real data)
assets/js/enhanced-api-test.js      (404 fixes)
```

## 🎯 Expected Result
Kansas City vs Las Vegas game should show:
- **Before 1:00 PM ET**: "SCHEDULED" 
- **After 1:00 PM ET**: "LIVE" (if game is actually live)

Console will show: `🏈 KC vs LV TIME VALIDATION: API says LIVE but...`

The timing validation is ready - it just needs to be uploaded to your live hosting!