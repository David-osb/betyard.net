# 🚀 Railway Cloud Deployment Guide

## Quick Deploy to Railway

### 1. One-Click Deploy
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

### 2. Manual Setup

1. **Install Railway CLI**
```bash
npm install -g @railway/cli
```

2. **Login to Railway**
```bash
railway login
```

3. **Deploy from this directory**
```bash
cd ml-backend
railway deploy
```

### 3. Environment Variables
Railway will automatically set:
- `PORT` - Server port (set by Railway)
- Python runtime from `runtime.txt`
- Dependencies from `requirements.txt`

### 4. Get Your ML Backend URL
After deployment, Railway will provide a URL like:
```
https://your-ml-backend-production.up.railway.app
```

### 5. Update Frontend
Replace the localhost URL in `assets/js/ml-integration.js`:
```javascript
// Change this:
this.baseURL = 'http://localhost:5000';

// To your Railway URL:
this.baseURL = 'https://your-ml-backend-production.up.railway.app';
```

## 🔧 Deployment Features

✅ **Auto-scaling** - Railway handles traffic spikes  
✅ **HTTPS** - Automatic SSL certificates  
✅ **Health monitoring** - Built-in uptime monitoring  
✅ **Logs** - Real-time application logs  
✅ **Zero downtime** - Rolling deployments  

## 💰 Cost
- **Free tier**: 500 hours/month (perfect for testing)
- **Pro**: $5/month for unlimited usage

## 🚦 Health Check
Your deployed ML backend will be available at:
- Health: `https://your-url/health`
- Predictions: `https://your-url/predict`
- Model Info: `https://your-url/model/info`