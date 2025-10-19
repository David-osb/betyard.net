// Quick ML Backend URL Configuration
// Update this after deploying to Railway/Render/Heroku

const ML_CONFIG = {
    // 🚀 PRODUCTION URLs (Update these with your deployed URLs)
    RAILWAY: 'https://betyard-ml-backend-production.up.railway.app',
    RENDER: 'https://betyard-ml-backend.onrender.com', 
    HEROKU: 'https://betyard-ml-backend.herokuapp.com',
    FLY: 'https://betyard-ml-backend.fly.dev',
    
    // 🔧 Development
    LOCAL: 'http://localhost:5000',
    
    // 🎯 Active Configuration - LIVE ML BACKEND! 🔥
    ACTIVE: 'RENDER' // ✅ LIVE: Your deployed ML backend with real XGBoost predictions!
};

// Export for use in ml-integration.js
window.ML_CONFIG = ML_CONFIG;

// Test active endpoint and optionally others
console.log('🔍 Testing ML Backend Endpoints...');

// Always test the active provider
const activeProvider = ML_CONFIG.ACTIVE;
console.log(`🎯 Testing ACTIVE provider: ${activeProvider}`);

try {
    const response = await fetch(`${ML_CONFIG[activeProvider]}/health`, { 
        method: 'GET',
        timeout: 5000 
    });
    const data = await response.json();
    
    if (data.status === 'healthy') {
        console.log(`✅ ${activeProvider}: ${ML_CONFIG[activeProvider]} - ONLINE`);
    } else {
        console.log(`⚠️ ${activeProvider}: ${ML_CONFIG[activeProvider]} - DEGRADED`);
    }
} catch (error) {
    console.log(`❌ ${activeProvider}: ${ML_CONFIG[activeProvider]} - OFFLINE`);
}

// Optional: Test other providers (set to false to reduce console noise)
const TEST_ALL_PROVIDERS = false;

if (TEST_ALL_PROVIDERS) {
    Object.keys(ML_CONFIG).forEach(async (provider) => {
        if (provider === 'ACTIVE' || provider === activeProvider) return;
        
        try {
            const response = await fetch(`${ML_CONFIG[provider]}/health`, { 
                method: 'GET',
                timeout: 5000 
            });
            const data = await response.json();
            
            if (data.status === 'healthy') {
                console.log(`✅ ${provider}: ${ML_CONFIG[provider]} - ONLINE`);
            } else {
                console.log(`⚠️ ${provider}: ${ML_CONFIG[provider]} - DEGRADED`);
            }
        } catch (error) {
            console.log(`❌ ${provider}: ${ML_CONFIG[provider]} - OFFLINE`);
        }
    });
}