# Deploy Backend Proxy Update to Render

## What Was Changed
Added API proxy endpoints to solve CORS issues with Tank01 NFL API.

## New Endpoints Added
1. `/api/proxy/tank01` - Generic proxy for any Tank01 endpoint
2. `/api/proxy/nfl/schedule` - NFL team schedules
3. `/api/proxy/nfl/roster` - NFL team rosters
4. `/api/proxy/nfl/games` - NFL games by week

## Deployment Steps

### Option 1: Automatic Deployment (Recommended)
If you have auto-deploy enabled on Render:
1. Changes are already pushed to GitHub (commit `58d3431`)
2. Render will automatically detect the changes
3. Wait 2-3 minutes for build and deployment
4. Check https://betyard-ml-backend.onrender.com/health

### Option 2: Manual Deployment
If auto-deploy is not enabled:
1. Go to https://dashboard.render.com
2. Find your `betyard-ml-backend` service
3. Click "Manual Deploy" → "Deploy latest commit"
4. Wait for build to complete

### Option 3: Deploy from Local
```powershell
cd "c:\Users\david\OneDrive\Workspace\Python Projects\UI.UX\ml-backend"
git push render main
```

## Verify Deployment

### Test the Health Endpoint
```powershell
curl https://betyard-ml-backend.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "positions": ["QB", "RB", "WR", "TE"]
}
```

### Test the Proxy Endpoint
```powershell
curl "https://betyard-ml-backend.onrender.com/api/proxy/nfl/roster?teamID=LAC&getStats=false"
```

Should return Chargers roster data without CORS errors.

## Frontend Changes
The frontend is already updated and will automatically use the proxy once backend is deployed:
- ✅ api-proxy-config.js created
- ✅ nfl-schedule-api.js updated to use proxy
- ✅ Script loaded in index.html
- ✅ Changes pushed to betyard.net

## Troubleshooting

### If deployment fails:
1. Check Render logs for errors
2. Verify requirements.txt includes `requests` library
3. Check if environment variables are set correctly

### If proxy doesn't work:
1. Check browser console for proxy errors
2. Verify backend URL in api-proxy-config.js
3. Test backend endpoints directly with curl

### CORS still appearing:
1. Clear browser cache
2. Hard refresh (Ctrl+Shift+R)
3. Check if backend is running on correct URL
4. Verify CORS is enabled in Flask app (`CORS(app)`)

## Expected Result
After deployment:
- ❌ Before: "CORS policy: No 'Access-Control-Allow-Origin' header"
- ✅ After: All Tank01 API calls work through backend proxy
- 🎯 Roster data loads successfully
- 🚀 No CORS errors in browser console

## Render Dashboard
https://dashboard.render.com

## Backend URL
Production: https://betyard-ml-backend.onrender.com
Local: http://localhost:5000
