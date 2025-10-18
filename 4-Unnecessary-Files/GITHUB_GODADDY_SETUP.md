# 🚀 BetYard.net - GoDaddy Domain + GitHub Pages Setup Guide

## Current Status
✅ Repository: `betyard.net` exists  
✅ CNAME file: Fixed formatting  
✅ Content: Ready for deployment  

## Step-by-Step Setup

### 1. 📁 GitHub Repository Setup

Your repository `David-osb/betyard.net` should contain:
```
betyard.net/
├── CNAME                    # Contains: betyard.net
├── index.html              # Your main landing page
├── nfl-qb-predictor.html   # QB predictor app
├── assets/                 # All your assets
├── robots.txt
├── sitemap.xml
└── other files...
```

### 2. ⚙️ Enable GitHub Pages

1. Go to your repository: `https://github.com/David-osb/betyard.net`
2. Click **Settings** tab
3. Scroll to **Pages** section (left sidebar)
4. Under **Source**, select:
   - Source: **Deploy from a branch**
   - Branch: **main** (or **master**)
   - Folder: **/ (root)**
5. Click **Save**

### 3. 🌐 GoDaddy DNS Configuration

In your GoDaddy DNS settings for `betyard.net`:

#### Method 1: A Records (Recommended)
Delete any existing A records and add these **4 A records**:
```
Type: A    Name: @    Value: 185.199.108.153    TTL: 1 Hour
Type: A    Name: @    Value: 185.199.109.153    TTL: 1 Hour  
Type: A    Name: @    Value: 185.199.110.153    TTL: 1 Hour
Type: A    Name: @    Value: 185.199.111.153    TTL: 1 Hour
```

#### For www subdomain:
```
Type: CNAME    Name: www    Value: david-osb.github.io    TTL: 1 Hour
```

#### Method 2: CNAME (Alternative)
If you prefer CNAME (root domain):
```
Type: CNAME    Name: @    Value: david-osb.github.io    TTL: 1 Hour
```

### 4. 🔧 Verification Steps

1. **Check DNS Propagation** (takes 24-48 hours):
   ```powershell
   nslookup betyard.net
   ```

2. **Verify GitHub Pages**:
   - Go to repository Settings > Pages
   - Should show: "Your site is published at https://betyard.net"

3. **Test Domain**:
   - Visit: `https://betyard.net`
   - Visit: `https://www.betyard.net`

### 5. 🔒 SSL Certificate (HTTPS)

GitHub automatically provides SSL. In repository Settings > Pages:
- ✅ Check **"Enforce HTTPS"** (after domain verification)

### 6. 📱 Testing Checklist

- [ ] `betyard.net` loads correctly
- [ ] `www.betyard.net` redirects to main domain
- [ ] HTTPS is working (🔒 in browser)
- [ ] NFL QB Predictor app functions properly
- [ ] Mobile responsiveness works
- [ ] All assets load correctly

## 🛠️ Troubleshooting

### Domain Not Working?
1. **Check DNS**: Use online DNS checkers
2. **Wait**: DNS changes take 24-48 hours
3. **Clear cache**: Browser cache and DNS cache
4. **GitHub Status**: Check if CNAME is recognized in repo

### SSL Certificate Issues?
1. Wait for DNS propagation
2. Disable and re-enable "Enforce HTTPS"
3. Check that CNAME file contains only: `betyard.net`

### Files Not Loading?
1. Ensure all file paths are correct
2. Check case sensitivity (GitHub is case-sensitive)
3. Verify all assets are in the repository

## 🚀 Quick Commands

### Push changes to GitHub:
```powershell
cd "c:\Users\david\OneDrive\Workspace\Python Projects\UI.UX\betyard-deployment"
git add .
git commit -m "Updated betyard.net domain setup"
git push origin main
```

### Check domain status:
```powershell
# Check DNS
nslookup betyard.net

# Check web response
curl -I https://betyard.net
```

## 📞 Support Resources

- **GitHub Pages**: https://docs.github.com/en/pages
- **GoDaddy DNS**: https://www.godaddy.com/help/manage-dns
- **DNS Checker**: https://dnschecker.org/

## 🎯 Final Notes

- DNS changes can take up to 48 hours
- Keep your CNAME file clean (only `betyard.net`)
- Monitor your domain renewal dates
- GitHub Pages is free for public repositories

Your domain `betyard.net` should be live once DNS propagates! 🏆