# 🆓 FREE DEPLOYMENT GUIDE - Render.com

## Why Render? 
✅ **Completely FREE forever**
✅ **No credit card required**  
✅ **Automatic HTTPS**
✅ **GitHub integration**
✅ **Perfect for ML backends**

## 🚀 Deploy to Render (5 Minutes)

### Step 1: Create Account
1. Go to https://render.com
2. Sign up with GitHub (free)
3. No credit card needed!

### Step 2: Deploy Your ML Backend
1. Click **"New"** → **"Web Service"**
2. Connect your GitHub repository
3. Choose your `betyard.net` repo
4. **Root Directory**: `ml-backend`
5. **Build Command**: `pip install -r requirements.txt`
6. **Start Command**: `python app.py`
7. Click **Create Web Service**

### Step 3: Get Your URL
Render gives you a URL like:
```
https://betyard-ml-backend.onrender.com
```

### Step 4: Update Frontend
Edit `assets/js/ml-config.js`:
```javascript
const ML_CONFIG = {
    RENDER: 'https://YOUR-APP-NAME.onrender.com',
    ACTIVE: 'RENDER'  // Change this line
};
```

## 🔧 Render Configuration

Your app will automatically:
- ✅ Install Python dependencies
- ✅ Start your Flask server  
- ✅ Get HTTPS certificate
- ✅ Auto-redeploy on GitHub push

## ⚡ Performance Notes
- **Cold Start**: ~10-15 seconds after 15min inactivity
- **Active Performance**: Excellent (~100-200ms responses)
- **Storage**: Persistent during runtime
- **Monthly Limit**: 750 hours (more than enough!)

## 💰 Cost Breakdown
- **Monthly Cost**: $0.00 forever
- **Bandwidth**: Unlimited
- **Deployments**: Unlimited
- **Custom Domain**: Free

## 🔄 Auto-Deploy Setup
Once connected, every GitHub push automatically deploys!

## 🚨 Only Limitation
Apps "sleep" after 15 minutes of no requests. First request after sleep takes ~10-15 seconds to wake up. Perfect for development and testing!

## 🎯 Next Steps
1. Deploy to Render (free)
2. Test your ML predictions
3. If you need 24/7 always-on, upgrade later ($7/month)