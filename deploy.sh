#!/bin/bash
# 🚀 BetYard.net Deployment Script

echo "🏈 BetYard.net Deployment Preparation"
echo "====================================="

# Create deployment folder
echo "📁 Creating deployment folder..."
mkdir -p betyard-deployment

# Copy essential files
echo "📋 Copying files for deployment..."
cp index.html betyard-deployment/
cp nfl-qb-predictor.html betyard-deployment/
cp simple_server.py betyard-deployment/
cp Start-Server.ps1 betyard-deployment/
cp start_server.bat betyard-deployment/

# Create assets folder
mkdir -p betyard-deployment/assets

# Create CNAME file for GitHub Pages
echo "betyard.net" > betyard-deployment/CNAME

# Create robots.txt
cat > betyard-deployment/robots.txt << EOF
User-agent: *
Allow: /

Sitemap: https://betyard.net/sitemap.xml
EOF

# Create basic sitemap.xml
cat > betyard-deployment/sitemap.xml << EOF
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://betyard.net/</loc>
    <lastmod>$(date +%Y-%m-%d)</lastmod>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://betyard.net/nfl-qb-predictor.html</loc>
    <lastmod>$(date +%Y-%m-%d)</lastmod>
    <changefreq>daily</changefreq>
    <priority>0.9</priority>
  </url>
</urlset>
EOF

echo "✅ Deployment files ready in 'betyard-deployment' folder"
echo ""
echo "🌐 Next steps:"
echo "1. Upload 'betyard-deployment' folder contents to your hosting"
echo "2. Configure DNS to point betyard.net to your hosting"
echo "3. Ensure SSL certificate is active"
echo "4. Test all functionality"
echo ""
echo "🎯 Your professional NFL analysis platform is ready!"