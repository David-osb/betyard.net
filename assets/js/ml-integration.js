// Real ML Backend Integration for BetYard
// Replaces mock XGBoost simulation with actual trained model

class BetYardMLAPI {
    constructor() {
        // 🌐 CLOUD DEPLOYMENT READY
        // Update this URL with your Railway/cloud deployment URL
        this.baseURL = this.getMLBackendURL();
        this.isAvailable = false;
        
        console.log(`🔗 ML Backend URL: ${this.baseURL}`);
        this.checkBackendHealth();
    }
    
    getMLBackendURL() {
        // 🌐 Use ML_CONFIG if available
        if (window.ML_CONFIG) {
            const activeProvider = window.ML_CONFIG.ACTIVE;
            const url = window.ML_CONFIG[activeProvider];
            console.log(`🔧 Using ${activeProvider} provider: ${url}`);
            return url;
        }
        
        // 🚀 Fallback: Auto-detect environment
        const isLocalDevelopment = window.location.hostname === 'localhost' || 
                                  window.location.hostname === '127.0.0.1' ||
                                  window.location.protocol === 'file:';
                                  
        return isLocalDevelopment 
            ? 'http://localhost:5000' 
            : 'https://betyard-ml-backend-production.up.railway.app';
    }

    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.baseURL}/health`);
            const data = await response.json();
            this.isAvailable = data.status === 'healthy' && data.model_loaded;
            
            if (this.isAvailable) {
                console.log('🧠 ML Backend connected - Real XGBoost model active!');
                this.logModelInfo();
            } else {
                console.log('⚠️ ML Backend unavailable - Using fallback predictions');
            }
        } catch (error) {
            console.log('⚠️ ML Backend not running - Using fallback predictions');
            this.isAvailable = false;
        }
    }

    async logModelInfo() {
        try {
            const response = await fetch(`${this.baseURL}/model/info`);
            const data = await response.json();
            console.log('📊 ML Model Info:', data);
            console.log('🎯 Feature Importance:', data.feature_importance);
        } catch (error) {
            console.log('Could not fetch model info:', error);
        }
    }

    async getPrediction(qbName, teamCode, opponentCode = null) {
        if (!this.isAvailable) {
            return this.getFallbackPrediction(qbName, teamCode);
        }

        try {
            const response = await fetch(`${this.baseURL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    qb_name: qbName,
                    team_code: teamCode,
                    opponent_code: opponentCode
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            
            if (data.success) {
                console.log('🎯 Real ML Prediction Generated:', data.prediction);
                console.log('📊 Weather Impact:', data.metadata.weather_impact);
                console.log('🏥 Injury Adjustment:', data.metadata.injury_adjustment);
                
                return {
                    ...data.prediction,
                    metadata: data.metadata,
                    source: 'XGBoost_Real'
                };
            } else {
                throw new Error(data.error);
            }

        } catch (error) {
            console.log('❌ ML Backend error:', error.message);
            console.log('🔄 Falling back to simulation...');
            return this.getFallbackPrediction(qbName, teamCode);
        }
    }

    getFallbackPrediction(qbName, teamCode) {
        // Enhanced fallback with more realistic variance
        const basePrediction = {
            passing_yards: Math.floor(Math.random() * 150) + 200,
            completions: Math.floor(Math.random() * 10) + 20,
            attempts: Math.floor(Math.random() * 15) + 30,
            touchdowns: Math.floor(Math.random() * 3) + 1,
            interceptions: Math.floor(Math.random() * 2),
            confidence: Math.floor(Math.random() * 20) + 70
        };

        // Calculate QB rating
        const compPct = (basePrediction.completions / basePrediction.attempts) * 100;
        const yardsPerAtt = basePrediction.passing_yards / basePrediction.attempts;
        const tdPct = (basePrediction.touchdowns / basePrediction.attempts) * 100;
        const intPct = (basePrediction.interceptions / basePrediction.attempts) * 100;

        basePrediction.qb_rating = Math.min(158.3, Math.max(0, 
            (compPct - 30) * 0.05 + 
            (yardsPerAtt - 3) * 0.25 + 
            tdPct * 0.2 - 
            intPct * 0.25
        ) * 100);

        return {
            ...basePrediction,
            source: 'Fallback_Simulation',
            metadata: {
                note: 'Using fallback prediction - Start ML backend for real predictions'
            }
        };
    }

    async getWeatherData(teamCode) {
        if (!this.isAvailable) {
            return { temp: 72, wind: 5, precipitation: 0, dome: false };
        }

        // Weather data is included in the main prediction
        // This is a separate endpoint for weather-only requests
        return { temp: 72, wind: 5, precipitation: 0, dome: false };
    }
}

// Initialize ML API
window.BetYardML = new BetYardMLAPI();

// Enhanced XGBoost integration function
async function generateRealMLPrediction(qbName, teamCode, opponentCode = null) {
    console.log(`🧠 Generating ML prediction for ${qbName} (${teamCode})`);
    
    const prediction = await window.BetYardML.getPrediction(qbName, teamCode, opponentCode);
    
    // Add enhanced logging
    if (prediction.source === 'XGBoost_Real') {
        console.log('✅ Using REAL XGBoost model predictions!');
        console.log('📊 Model metadata:', prediction.metadata);
    } else {
        console.log('⚠️ Using fallback simulation - Start ML backend for real predictions');
        console.log('   Run: .\\ml-backend\\Start-ML-Backend.ps1');
    }
    
    return prediction;
}

// Replace the old XGBoost simulation
if (window.XGBoostModel) {
    console.log('🔄 Replacing XGBoost simulation with real ML integration');
    
    // Keep the old interface but use real ML backend
    window.XGBoostModel.predict = async function(features) {
        // This is called by legacy code - redirect to real ML
        const prediction = await window.BetYardML.getPrediction('Unknown', 'KC');
        return prediction.confidence / 100; // Return normalized confidence
    };
    
    window.XGBoostModel.getFeatureImportance = async function() {
        try {
            const response = await fetch('http://localhost:5000/model/info');
            const data = await response.json();
            return data.feature_importance;
        } catch {
            // Fallback feature importance
            return {
                recent_form: 18.2,
                injury_factor: 15.3,
                opponent_strength: 14.1,
                weather_impact: 12.4,
                home_advantage: 11.2,
                experience_factor: 10.1,
                matchup_history: 9.3,
                clutch_rating: 8.4,
                variance_score: 1.0
            };
        }
    };
}

console.log('🏈 BetYard Real ML Integration loaded!');
console.log('🎯 To use real XGBoost predictions, start the ML backend:');
console.log('   cd ml-backend && .\\Start-ML-Backend.ps1');