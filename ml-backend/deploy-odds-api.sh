#!/bin/bash
# Deployment script for Betyard ML Backend with Odds API Integration

echo "🚀 Deploying Betyard ML Backend with Odds API Integration..."

# Stop any existing processes
echo "Stopping existing backend..."
pkill -f "python app.py" || true

# Install/update dependencies
echo "Installing dependencies..."
pip install aiohttp==3.10.5 python-dotenv==1.0.1

# Set environment variables for production
echo "Setting environment variables..."
export ODDS_API_KEY="464b16f6d6ce6f6ae933e55c07cf11c1"
export FLASK_ENV="production"
export PORT="5001"

# Test the application
echo "Testing application..."
python -c "
import app
print('✅ Application imports successfully')
print('✅ Odds API key:', app.ODDS_API_KEY[:8] + '...' + app.ODDS_API_KEY[-4:])
"

echo "🎯 Deployment completed!"
echo "📡 Backend will be available at: https://betyard-ml-backend.onrender.com"
echo ""
echo "📊 New Endpoints Available:"
echo "  GET  /api/odds/compare/nfl - Real-time odds comparison"
echo "  POST /api/odds/value-bets/nfl - Value betting opportunities"  
echo "  GET  /api/odds/arbitrage/nfl - Arbitrage opportunities"
echo "  GET  /api/odds/best-lines/nfl/<team> - Best odds for team"
echo ""
echo "🔧 To test locally:"
echo "  curl https://betyard-ml-backend.onrender.com/api/odds/compare/nfl"