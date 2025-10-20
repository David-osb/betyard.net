# 🚀 Deploy Multi-Position ML Backend to Render.com

## ✅ What You're Deploying
- **4 XGBoost Models**: QB, RB, WR, TE predictions
- **Flask REST API**: `/health`, `/predict`, `/model/info` endpoints
- **CORS Enabled**: Works with your betyard.net frontend
- **Model Files**: ~4MB total (all 4 .pkl files)

---

## 📋 Pre-Deployment Checklist

### ✅ Files Ready to Deploy
- [x] `app.py` - Flask server with multi-position support
- [x] `requirements.txt` - Python dependencies
- [x] `runtime.txt` - Python 3.11
- [x] `Procfile` - Render start command
- [x] `qb_model.pkl` - QB XGBoost model
- [x] `rb_model.pkl` - RB XGBoost model  
- [x] `wr_model.pkl` - WR XGBoost model
- [x] `te_model.pkl` - TE XGBoost model

### ⚠️ Important Notes
- **Git LFS**: Model files are ~1MB each, should work fine
- **Cold Start**: First request takes 10-15 seconds after sleep
- **Free Tier**: 750 hours/month (perfect for testing)
- **Auto-Deploy**: Every GitHub push auto-deploys

---

## 🎯 Step-by-Step Deployment

### Step 1: Sign Up for Render (2 minutes)
1. Go to https://render.com
2. Click **"Get Started for Free"**
3. Sign up with GitHub (recommended)
4. **No credit card required** ✅

### Step 2: Create New Web Service (3 minutes)
1. Click **"New +"** → **"Web Service"**
2. Click **"Connect GitHub"** and authorize Render
3. Select your repository: `David-osb/betyard.net`
4. Click **"Connect"**

### Step 3: Configure Service Settings

#### Basic Settings:
- **Name**: `betyard-ml-backend` (or your choice)
- **Region**: `Oregon (US West)` (closest to you)
- **Branch**: `main`
- **Root Directory**: `ml-backend` ⚠️ **CRITICAL!**
- **Runtime**: `Python 3`

#### Build & Deploy Settings:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`

#### Instance Settings:
- **Instance Type**: `Free` (perfect for testing)
- **Auto-Deploy**: `Yes` ✅ (deploy on every push)

### Step 4: Add Environment Variables (Optional)
Click **"Advanced"** → **"Add Environment Variable"**

**Recommended Variables:**
```
PORT = 5000
PYTHON_VERSION = 3.11.0
FLASK_ENV = production
```

### Step 5: Deploy! 🚀
1. Click **"Create Web Service"**
2. Wait 5-10 minutes for first deploy
3. Watch the build logs in real-time

#### Expected Build Output:
```
==> Building...
Collecting flask==3.0.3
Collecting xgboost==2.0.3
Installing collected packages...
==> Build successful!

==> Starting service...
🏈 NFL ML Prediction Backend Starting...
✅ Loading QB model...
✅ Loading RB model...
✅ Loading WR model...
✅ Loading TE model...
🚀 All 4 models loaded successfully!
Flask app is running on port 5000
```

### Step 6: Get Your Deployment URL
After deployment completes, you'll get a URL like:
```
https://betyard-ml-backend.onrender.com
```

**Test it immediately:**
```
https://betyard-ml-backend.onrender.com/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "QB": true,
    "RB": true,
    "WR": true,
    "TE": true
  },
  "timestamp": "2025-10-20T12:34:56.789Z"
}
```

---

## 🔧 Update Frontend Configuration

### Step 7: Update ml-config.js (2 minutes)

Edit `assets/js/ml-config.js`:

```javascript
const ML_CONFIG = {
    // 🚀 PRODUCTION URLs
    RENDER: 'https://betyard-ml-backend.onrender.com', // ⬅️ UPDATE THIS!
    
    // 🔧 Development
    LOCAL: 'http://localhost:5000',
    
    // 🎯 Active Configuration
    ACTIVE: 'RENDER' // ⬅️ CHANGE FROM 'LOCAL' TO 'RENDER'
};
```

**Replace with YOUR actual URL from Render!**

### Step 8: Test Frontend Connection

1. Open browser console on betyard.net
2. Look for ML backend test results:
```
🔍 Testing ML Backend Endpoints...
🎯 Testing ACTIVE provider: RENDER
✅ RENDER: https://betyard-ml-backend.onrender.com - ONLINE
```

### Step 9: Commit and Push Changes

```powershell
git add assets/js/ml-config.js
git commit -m "feat: Update ML backend to Render deployment"
git push origin main
```

This will:
- ✅ Update your frontend to use Render
- ✅ Auto-deploy the backend (if any changes)

---

## 🧪 Testing Your Deployment

### Test 1: Health Check
```powershell
Invoke-WebRequest -Uri https://YOUR-APP.onrender.com/health | ConvertFrom-Json
```

### Test 2: QB Prediction
```powershell
$body = @{
    player_name = "Patrick Mahomes"
    team_code = "KC"
    opponent_code = "LV"
    position = "QB"
} | ConvertTo-Json

Invoke-RestMethod -Uri https://YOUR-APP.onrender.com/predict -Method POST -Body $body -ContentType "application/json"
```

### Test 3: All Position Predictions
```powershell
# Test each position
"QB", "RB", "WR", "TE" | ForEach-Object {
    $body = @{
        player_name = "Test Player"
        team_code = "KC"
        opponent_code = "LV"
        position = $_
    } | ConvertTo-Json
    
    Write-Host "`n🏈 Testing $_..." -ForegroundColor Cyan
    Invoke-RestMethod -Uri https://YOUR-APP.onrender.com/predict -Method POST -Body $body -ContentType "application/json"
}
```

---

## 📊 Monitoring Your Deployment

### Render Dashboard Features:
- **Real-time Logs**: See every request in real-time
- **Metrics**: CPU, Memory, Request volume
- **Deployments**: History of all deploys
- **Events**: Build/deploy/error notifications

### Health Monitoring:
Render automatically pings your `/health` endpoint to check if service is alive.

### Sleep Mode Behavior:
- **Free Tier**: Sleeps after 15 minutes of inactivity
- **Wake Time**: 10-15 seconds on first request
- **User Experience**: First prediction slightly slower, then fast

---

## 🚨 Troubleshooting

### Issue 1: Build Fails with "No module named 'flask'"
**Solution**: Check `requirements.txt` is in `ml-backend` folder

### Issue 2: Model files not found
**Solution**: Ensure all 4 .pkl files are committed to GitHub:
```powershell
git add ml-backend/*.pkl
git commit -m "chore: Add model files"
git push
```

### Issue 3: CORS errors in browser
**Solution**: Check `flask-cors` is installed and app.py has:
```python
CORS(app, resources={r"/*": {"origins": "*"}})
```

### Issue 4: Cold start takes too long
**Solutions**:
1. **Upgrade to paid tier** ($7/month) for always-on
2. **Keep alive ping**: Set up free uptime monitoring (uptimerobot.com)
3. **Warm-up request**: Add health check in frontend on page load

### Issue 5: Model predictions are wrong
**Solution**: Check which model version is deployed:
```powershell
Invoke-RestMethod -Uri https://YOUR-APP.onrender.com/model/info
```

---

## 💰 Cost & Performance

### Free Tier (Current):
- **Cost**: $0/month forever
- **Hours**: 750/month (31 days)
- **Memory**: 512MB
- **CPU**: 0.1 CPU
- **Bandwidth**: Unlimited
- **Deployments**: Unlimited
- **Sleep**: After 15 min inactivity

### Paid Tier (If Needed):
- **Cost**: $7/month
- **Always On**: No sleep mode
- **Memory**: 512MB
- **CPU**: 0.5 CPU
- **Response**: <100ms always

### When to Upgrade:
- ✅ **Stay Free If**: Testing, low traffic, can tolerate cold starts
- 💰 **Upgrade If**: Production traffic, need <100ms always, 24/7 availability

---

## 🔄 Auto-Deploy Workflow

Once connected to GitHub, every push automatically:
1. **Detects changes** in `ml-backend/` folder
2. **Builds** new Docker container
3. **Tests** with health check
4. **Deploys** if successful
5. **Rolls back** if failed

**No manual steps needed!**

---

## 🎯 Next Steps After Deployment

### 1. Update Cache Buster
Update `index.html` to force browser reload:
```javascript
<meta name="timestamp" content="2025-10-20-render-deployment-[timestamp]">
```

### 2. Test All Positions
Test QB, RB, WR, TE predictions on live site

### 3. Monitor Performance
Watch Render logs for:
- Response times
- Error rates  
- Memory usage

### 4. Set Up Uptime Monitoring (Optional)
Use https://uptimerobot.com (free) to:
- Ping your `/health` endpoint every 5 minutes
- Prevent sleep mode
- Get alerts if service goes down

---

## 📚 Useful Render Commands

### View Logs:
Click **"Logs"** tab in Render dashboard

### Manually Trigger Deploy:
Click **"Manual Deploy"** → **"Deploy latest commit"**

### Restart Service:
Click **"Manual Deploy"** → **"Clear build cache & deploy"**

### Check Environment:
Click **"Environment"** tab to see all variables

---

## ✅ Deployment Checklist

Before going live:
- [ ] Render service deployed successfully
- [ ] Health endpoint returns 200 OK
- [ ] All 4 models loaded (QB, RB, WR, TE)
- [ ] ml-config.js updated with Render URL
- [ ] ACTIVE set to 'RENDER' in ml-config.js
- [ ] Tested predictions for all positions
- [ ] Committed and pushed frontend changes
- [ ] Updated cache buster in index.html
- [ ] Tested on live betyard.net site
- [ ] No CORS errors in browser console
- [ ] Response times acceptable (<500ms)

---

## 🎉 Success Criteria

Your deployment is successful when:
1. ✅ `/health` returns healthy status
2. ✅ `/predict` works for QB, RB, WR, TE
3. ✅ Frontend shows predictions on betyard.net
4. ✅ No errors in browser console
5. ✅ No errors in Render logs

---

## 🆘 Need Help?

- **Render Docs**: https://render.com/docs
- **Render Community**: https://community.render.com
- **Your Logs**: Check Render dashboard logs tab
- **Test Local First**: Run `python app.py` locally to verify

---

## 🚀 You're Ready to Deploy!

Follow the steps above and your multi-position ML backend will be live in under 15 minutes!

**Current Status**: All code ready, just need to click "Deploy" on Render! 🎯
