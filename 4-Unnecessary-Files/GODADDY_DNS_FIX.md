# 🔧 GoDaddy DNS Configuration for GitHub Pages

## Common GoDaddy DNS Issues & Solutions

### 🚨 "Invalid DNS" Error Solutions:

#### **Issue 1: GoDaddy DNS Format**
GoDaddy uses slightly different terminology:

**Instead of "@ (or betyard.net)" use:**
- **Type**: A Record
- **Host**: @ 
- **Points to**: [IP Address]
- **TTL**: 1 Hour (or 3600 seconds)

#### **Issue 2: Remove Existing Records First**
GoDaddy often has default records that conflict:
1. **Delete** any existing A records pointing to GoDaddy parking
2. **Delete** any CNAME records for @ (root domain)
3. **Then** add the GitHub Pages records

### 🎯 **Correct GoDaddy DNS Settings:**

#### **A Records (Add these 4):**
```
Type: A Record
Host: @
Points to: 185.199.108.153
TTL: 1 Hour

Type: A Record  
Host: @
Points to: 185.199.109.153
TTL: 1 Hour

Type: A Record
Host: @  
Points to: 185.199.110.153
TTL: 1 Hour

Type: A Record
Host: @
Points to: 185.199.111.153  
TTL: 1 Hour
```

#### **CNAME Record:**
```
Type: CNAME
Host: www
Points to: david-osb.github.io
TTL: 1 Hour
```

### 🔍 **Step-by-Step GoDaddy Fix:**

#### **1. Access GoDaddy DNS:**
- Login to GoDaddy account
- Go to "My Products" → "All Products and Services"
- Find your "betyard.net" domain
- Click "DNS" or "Manage DNS"

#### **2. Clean Existing Records:**
- Look for any A records with "@" host
- **DELETE** records pointing to:
  - Parked domain IPs (usually 184.168.x.x)
  - GoDaddy default parking
  - Any conflicting A records

#### **3. Add GitHub Pages Records:**
- Click "Add Record" or "+"
- Select "A" type
- Host: @ (just the @ symbol)
- Points to: 185.199.108.153
- Save and repeat for other 3 IPs

#### **4. Add CNAME:**
- Add Record → CNAME
- Host: www
- Points to: david-osb.github.io
- Save

### 🚨 **Common GoDaddy Mistakes:**

#### **❌ Wrong Format:**
- Using "betyard.net" instead of "@"
- Including "https://" in the IP address
- Missing the www CNAME record

#### **✅ Correct Format:**
- Host field: exactly "@" (no quotes)
- IP addresses: just the numbers, no protocol
- TTL: 1 Hour or 3600

### 🛠️ **Alternative: Use GoDaddy Forwarding**

If DNS keeps failing, use GoDaddy's domain forwarding:

#### **Domain Forwarding Setup:**
1. In GoDaddy Domain settings
2. Look for "Forwarding" section  
3. Forward betyard.net → https://david-osb.github.io
4. Forward www.betyard.net → https://david-osb.github.io
5. Enable "Forward with masking" to keep betyard.net in URL

### 🔄 **Troubleshooting Commands:**

Check if DNS is working:
```bash
nslookup betyard.net
dig betyard.net
```

Should return GitHub's IP addresses.

### 📞 **GoDaddy Support Script:**
If still having issues, call GoDaddy support and say:

*"I need to point my domain betyard.net to GitHub Pages. I need to add 4 A records with @ host pointing to GitHub's IPs: 185.199.108.153, 185.199.109.153, 185.199.110.153, and 185.199.111.153. I also need a CNAME record for www pointing to david-osb.github.io. The DNS manager is showing invalid errors."*

### ⚡ **Quick Alternative: Cloudflare**
If GoDaddy DNS keeps causing issues:
1. Keep domain at GoDaddy
2. Change nameservers to Cloudflare (free)
3. Manage DNS through Cloudflare (much easier)
4. Add GitHub Pages records in Cloudflare

Would you like me to help with any of these solutions?