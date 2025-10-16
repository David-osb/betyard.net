# 🚀 GoDaddy Hosting Setup for BetYard.net

## Step-by-Step Guide to Upload Your NFL QB Predictor

### Method 1: File Manager (Easiest)

#### 🔐 Access Your GoDaddy Account:
1. Go to https://godaddy.com
2. Sign in to your account
3. Go to "My Products" → "Web Hosting"
4. Click "Manage" next to your betyard.net hosting

#### 📁 Upload via File Manager:
1. In cPanel, find "File Manager" 
2. Navigate to `public_html` folder (this is your website root)
3. **DELETE** any existing files (like index.html, coming soon pages)
4. **UPLOAD** all files from your `betyard-deployment` folder:
   - index.html
   - nfl-qb-predictor.html
   - robots.txt
   - sitemap.xml
   - CNAME (optional for GoDaddy)
   - simple_server.py (for backup)

#### 📂 Create Assets Folder:
1. In `public_html`, create new folder called "assets"
2. This is where you'll add favicons and images later

### Method 2: FTP Upload (Advanced)

#### 📡 FTP Connection Details:
- **Host:** Your domain (betyard.net) or FTP hostname from GoDaddy
- **Username:** Your cPanel username
- **Password:** Your cPanel password
- **Port:** 21 (standard FTP) or 22 (SFTP)

#### 🔧 FTP Software Options:
- **FileZilla** (Free) - https://filezilla-project.org/
- **WinSCP** (Windows) - https://winscp.net/
- **Built-in Windows Explorer** (ftp://yourdomain.com)

#### 📤 Upload Process:
1. Connect to your FTP
2. Navigate to `public_html` folder
3. Upload all files from `betyard-deployment` folder
4. Set permissions to 644 for files, 755 for folders

### Method 3: cPanel File Upload

#### 🎛️ cPanel Steps:
1. Log into GoDaddy cPanel
2. Find "File Manager" icon
3. Open `public_html` directory
4. Click "Upload" button
5. Select all files from `betyard-deployment` folder
6. Wait for upload to complete
7. Extract if uploaded as ZIP

### 🔗 Domain Configuration

#### ✅ DNS Settings (Should already be configured):
Your betyard.net domain should point to GoDaddy hosting automatically. If not:
1. Go to GoDaddy Domain settings
2. Ensure nameservers point to GoDaddy hosting
3. A Record should point to your hosting IP

#### 🔒 SSL Certificate:
1. In GoDaddy cPanel, find "SSL/TLS"
2. Enable "Let's Encrypt" free SSL
3. Force HTTPS redirect
4. Your site will be accessible at https://betyard.net

### 📋 File Structure on GoDaddy:
```
public_html/
├── index.html              ← Your BetYard homepage
├── nfl-qb-predictor.html   ← Your NFL app
├── robots.txt              ← SEO file
├── sitemap.xml             ← Search engine map
├── assets/                 ← Images/icons folder
│   └── (future icons)
└── (other files)
```

### 🎯 Testing After Upload:

#### 🌐 Test Your Website:
1. Visit https://betyard.net (homepage)
2. Visit https://betyard.net/nfl-qb-predictor.html (NFL app)
3. Test mobile responsiveness
4. Check all buttons and features work
5. Verify API calls work (click "Fetch Live NFL Data")

### ⚡ Quick Upload Checklist:
- [ ] Access GoDaddy cPanel
- [ ] Open File Manager
- [ ] Navigate to public_html
- [ ] Delete existing files
- [ ] Upload index.html
- [ ] Upload nfl-qb-predictor.html
- [ ] Upload robots.txt
- [ ] Upload sitemap.xml
- [ ] Create assets folder
- [ ] Test https://betyard.net
- [ ] Test NFL predictor functionality

### 🔧 Troubleshooting:

#### 🚨 Common Issues:
1. **404 Error:** Check files are in `public_html`, not a subfolder
2. **Permission Error:** Set file permissions to 644
3. **SSL Issues:** Enable SSL in GoDaddy cPanel
4. **API Errors:** These are normal for CORS, fallback will work

#### 💡 Pro Tips:
- Keep original files as backup
- Test on mobile devices
- Monitor site speed with GoDaddy tools
- Set up Google Analytics for visitor tracking

### 📞 GoDaddy Support:
If you need help:
- GoDaddy Phone: 480-505-8877
- Live Chat in your account
- Help Center: support.godaddy.com

## 🏆 Final Result:
After upload, your betyard.net will have:
- Professional landing page
- Fully functional NFL QB predictor
- Mobile-responsive design
- SEO optimization
- SSL security

**Your professional NFL analysis platform will be live at https://betyard.net!**