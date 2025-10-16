const express = require('express');
const fetch = require('node-fetch');
const app = express();
const PORT = process.env.PORT || 3001;

// CORS middleware
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-rapidapi-key, x-rapidapi-host');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

app.get('/tank01-getNFL', async (req, res) => {
  try {
    const date = req.query.date || new Date().toISOString().slice(0,10).replace(/-/g,'');
    const RAPIDAPI_KEY = process.env.RAPIDAPI_KEY || 'be76a86c9cmsh0d0cecaaefbc722p1efcdbjsn598e66d34cf3';
    const RAPIDAPI_HOST = 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com';
    const upstream = `https://${RAPIDAPI_HOST}/getNFLDFS?date=${date}&includeTeamDefense=true`;

    console.log(`📡 Proxying request to Tank01 API for date: ${date}`);

    const response = await fetch(upstream, {
      method: 'GET',
      headers: {
        'x-rapidapi-host': RAPIDAPI_HOST,
        'x-rapidapi-key': RAPIDAPI_KEY,
        'Content-Type': 'application/json'
      }
    });

    const text = await response.text();
    console.log(`✅ Tank01 API responded with status: ${response.status}`);
    
    res.status(response.status).send(text);
  } catch (err) {
    console.error('❌ Proxy error:', err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Tank01 proxy server running on http://localhost:${PORT}`);
  console.log(`📡 Ready to proxy Tank01 NFL API requests`);
});