const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS for your frontend
app.use(cors());
app.use(express.json());
app.use(express.static('.')); // Serve static files from current directory

// Tank01 API Configuration
const RAPIDAPI_KEY = process.env.RAPIDAPI_KEY || 'be76a86c9cmsh0d0cecaaefbc722p1efcdbjsn598e66d34cf3';
const RAPIDAPI_HOST = 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com';

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Tank01 NFL Data Endpoint
app.get('/api/nfl/tank01', async (req, res) => {
    try {
        const date = req.query.date || new Date().toISOString().slice(0, 10).replace(/-/g, '');
        
        console.log(`📡 Fetching Tank01 data for date: ${date}`);
        
        const url = `https://${RAPIDAPI_HOST}/getNFLDFS?date=${date}&includeTeamDefense=true`;
        
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'x-rapidapi-host': RAPIDAPI_HOST,
                'x-rapidapi-key': RAPIDAPI_KEY,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`Tank01 API error: ${response.status}`);
        }
        
        const data = await response.json();
        console.log(`✅ Tank01 data fetched successfully`);
        
        res.json({
            success: true,
            data: data,
            timestamp: new Date().toISOString(),
            source: 'tank01'
        });
        
    } catch (error) {
        console.error('❌ Tank01 API Error:', error.message);
        res.status(500).json({
            success: false,
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

// RapidAPI NFL Data Endpoint (Fallback)
app.get('/api/nfl/rapidapi', async (req, res) => {
    try {
        const RAPIDAPI_NFL_HOST = 'nfl-api-data.p.rapidapi.com';
        
        console.log('📡 Fetching RapidAPI NFL team listing...');
        
        const url = `https://${RAPIDAPI_NFL_HOST}/nfl-team-listing/v1/data`;
        
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'X-RapidAPI-Key': RAPIDAPI_KEY,
                'X-RapidAPI-Host': RAPIDAPI_NFL_HOST,
                'Accept': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`RapidAPI error: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ RapidAPI data fetched successfully');
        
        res.json({
            success: true,
            data: data,
            timestamp: new Date().toISOString(),
            source: 'rapidapi'
        });
        
    } catch (error) {
        console.error('❌ RapidAPI Error:', error.message);
        res.status(500).json({
            success: false,
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 NFL Data Server running on http://localhost:${PORT}`);
    console.log(`📡 Tank01 API endpoint: http://localhost:${PORT}/api/nfl/tank01`);
    console.log(`📡 RapidAPI endpoint: http://localhost:${PORT}/api/nfl/rapidapi`);
    console.log(`🏥 Health check: http://localhost:${PORT}/api/health`);
});is it working