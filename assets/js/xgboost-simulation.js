// XGBoost Model Simulation - In production, this would load actual XGBoost.js
window.XGBoostModel = {
    predict: function(features) {
        // Advanced ML prediction simulation with realistic variance
        const weights = {
            recent_form: 0.18,
            injury_factor: 0.15,
            opponent_strength: 0.14,
            weather_impact: 0.12,
            home_advantage: 0.11,
            experience_factor: 0.10,
            matchup_history: 0.09,
            clutch_rating: 0.08,
            variance_score: 0.03
        };
        
        let prediction = 0;
        Object.keys(weights).forEach(key => {
            if (features[key] !== undefined) {
                prediction += features[key] * weights[key];
            }
        });
        
        return Math.max(0.1, Math.min(1.0, prediction));
    },
    
    getFeatureImportance: function() {
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