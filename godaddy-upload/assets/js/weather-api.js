/**
 * 🌤️ WEATHER API INTEGRATION
 * Real-time weather data for NFL stadium locations
 * Provider: Open-Meteo API (FREE - No API Key Required!)
 * Documentation: https://open-meteo.com/en/docs
 */

class WeatherAPI {
    constructor() {
        // Open-Meteo API configuration (FREE - No API key needed!)
        this.baseUrl = 'https://api.open-meteo.com/v1/forecast';
        this.cache = new Map();
        this.cacheTimeout = 30 * 60 * 1000; // 30 minutes cache
        
        // NFL Stadium Locations (City coordinates for weather)
        this.stadiumLocations = {
            'ARI': { city: 'Glendale', state: 'AZ', lat: 33.5276, lon: -112.2626 },
            'ATL': { city: 'Atlanta', state: 'GA', lat: 33.7573, lon: -84.4009 },
            'BAL': { city: 'Baltimore', state: 'MD', lat: 39.2780, lon: -76.6227 },
            'BUF': { city: 'Orchard Park', state: 'NY', lat: 42.7738, lon: -78.7870 },
            'CAR': { city: 'Charlotte', state: 'NC', lat: 35.2258, lon: -80.8528 },
            'CHI': { city: 'Chicago', state: 'IL', lat: 41.8623, lon: -87.6167 },
            'CIN': { city: 'Cincinnati', state: 'OH', lat: 39.0954, lon: -84.5160 },
            'CLE': { city: 'Cleveland', state: 'OH', lat: 41.5061, lon: -81.6995 },
            'DAL': { city: 'Arlington', state: 'TX', lat: 32.7473, lon: -97.0945 },
            'DEN': { city: 'Denver', state: 'CO', lat: 39.7439, lon: -105.0201 },
            'DET': { city: 'Detroit', state: 'MI', lat: 42.3400, lon: -83.0456 },
            'GB': { city: 'Green Bay', state: 'WI', lat: 44.5013, lon: -88.0622 },
            'HOU': { city: 'Houston', state: 'TX', lat: 29.6847, lon: -95.4107 },
            'IND': { city: 'Indianapolis', state: 'IN', lat: 39.7601, lon: -86.1639 },
            'JAX': { city: 'Jacksonville', state: 'FL', lat: 30.3240, lon: -81.6373 },
            'KC': { city: 'Kansas City', state: 'MO', lat: 39.0489, lon: -94.4839 },
            'LV': { city: 'Las Vegas', state: 'NV', lat: 36.0909, lon: -115.1833 },
            'LAC': { city: 'Inglewood', state: 'CA', lat: 33.9534, lon: -118.3392 },
            'LAR': { city: 'Inglewood', state: 'CA', lat: 33.9534, lon: -118.3392 },
            'MIA': { city: 'Miami Gardens', state: 'FL', lat: 25.9580, lon: -80.2389 },
            'MIN': { city: 'Minneapolis', state: 'MN', lat: 44.9740, lon: -93.2577 },
            'NE': { city: 'Foxborough', state: 'MA', lat: 42.0909, lon: -71.2643 },
            'NO': { city: 'New Orleans', state: 'LA', lat: 29.9511, lon: -90.0812 },
            'NYG': { city: 'East Rutherford', state: 'NJ', lat: 40.8135, lon: -74.0745 },
            'NYJ': { city: 'East Rutherford', state: 'NJ', lat: 40.8135, lon: -74.0745 },
            'PHI': { city: 'Philadelphia', state: 'PA', lat: 39.9008, lon: -75.1675 },
            'PIT': { city: 'Pittsburgh', state: 'PA', lat: 40.4468, lon: -80.0158 },
            'SF': { city: 'Santa Clara', state: 'CA', lat: 37.4030, lon: -121.9697 },
            'SEA': { city: 'Seattle', state: 'WA', lat: 47.5952, lon: -122.3316 },
            'TB': { city: 'Tampa', state: 'FL', lat: 27.9759, lon: -82.5033 },
            'TEN': { city: 'Nashville', state: 'TN', lat: 36.1665, lon: -86.7713 },
            'WSH': { city: 'Landover', state: 'MD', lat: 38.9076, lon: -76.8645 }
        };

        console.log('🌤️ Weather API initialized (Open-Meteo - FREE, no API key needed!)');
    }

    /**
     * Get weather for a specific NFL team's stadium
     */
    async getStadiumWeather(teamCode) {
        const location = this.stadiumLocations[teamCode];
        if (!location) {
            console.warn(`⚠️ No stadium location found for team: ${teamCode}`);
            return null;
        }

        // Check cache
        const cached = this.getCachedWeather(teamCode);
        if (cached) {
            console.log(`✅ Using cached weather for ${location.city}`);
            return cached;
        }

        try {
            console.log(`🌤️ Fetching weather for ${location.city}, ${location.state}...`);
            
            // Open-Meteo API parameters
            const params = new URLSearchParams({
                latitude: location.lat,
                longitude: location.lon,
                current: 'temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m',
                temperature_unit: 'fahrenheit',
                wind_speed_unit: 'mph',
                precipitation_unit: 'inch',
                timezone: 'America/New_York'
            });
            
            const url = `${this.baseUrl}?${params}`;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`Weather API error: ${response.status}`);
            }

            const data = await response.json();
            const weather = this.parseWeatherData(data, location);
            
            // Cache the result
            this.cacheWeather(teamCode, weather);
            
            console.log(`✅ Weather retrieved for ${location.city}:`, weather);
            return weather;

        } catch (error) {
            console.error(`❌ Failed to fetch weather for ${location.city}:`, error);
            return null;
        }
    }

    /**
     * Parse Open-Meteo API response
     */
    parseWeatherData(data, location) {
        const current = data.current;
        const weatherCode = current.weather_code;
        const condition = this.getWeatherCondition(weatherCode);
        
        return {
            city: location.city,
            state: location.state,
            temp: Math.round(current.temperature_2m),
            feelsLike: Math.round(current.apparent_temperature),
            condition: condition.main,
            description: condition.description,
            humidity: Math.round(current.relative_humidity_2m),
            windSpeed: Math.round(current.wind_speed_10m),
            windDirection: this.getWindDirection(current.wind_direction_10m),
            precipitation: current.precipitation,
            cloudCover: current.cloud_cover,
            weatherCode: weatherCode,
            timestamp: new Date(current.time)
        };
    }

    /**
     * Convert WMO weather code to readable condition
     * WMO Weather interpretation codes (WW)
     * https://open-meteo.com/en/docs
     */
    getWeatherCondition(code) {
        const conditions = {
            0: { main: 'Clear', description: 'Clear sky' },
            1: { main: 'Mainly Clear', description: 'Mainly clear' },
            2: { main: 'Partly Cloudy', description: 'Partly cloudy' },
            3: { main: 'Overcast', description: 'Overcast' },
            45: { main: 'Fog', description: 'Foggy' },
            48: { main: 'Fog', description: 'Depositing rime fog' },
            51: { main: 'Drizzle', description: 'Light drizzle' },
            53: { main: 'Drizzle', description: 'Moderate drizzle' },
            55: { main: 'Drizzle', description: 'Dense drizzle' },
            56: { main: 'Drizzle', description: 'Light freezing drizzle' },
            57: { main: 'Drizzle', description: 'Dense freezing drizzle' },
            61: { main: 'Rain', description: 'Slight rain' },
            63: { main: 'Rain', description: 'Moderate rain' },
            65: { main: 'Rain', description: 'Heavy rain' },
            66: { main: 'Rain', description: 'Light freezing rain' },
            67: { main: 'Rain', description: 'Heavy freezing rain' },
            71: { main: 'Snow', description: 'Slight snow' },
            73: { main: 'Snow', description: 'Moderate snow' },
            75: { main: 'Snow', description: 'Heavy snow' },
            77: { main: 'Snow', description: 'Snow grains' },
            80: { main: 'Rain', description: 'Slight rain showers' },
            81: { main: 'Rain', description: 'Moderate rain showers' },
            82: { main: 'Rain', description: 'Violent rain showers' },
            85: { main: 'Snow', description: 'Slight snow showers' },
            86: { main: 'Snow', description: 'Heavy snow showers' },
            95: { main: 'Thunderstorm', description: 'Thunderstorm' },
            96: { main: 'Thunderstorm', description: 'Thunderstorm with slight hail' },
            99: { main: 'Thunderstorm', description: 'Thunderstorm with heavy hail' }
        };
        
        return conditions[code] || { main: 'Unknown', description: 'Weather data unavailable' };
    }

    /**
     * Convert wind degree to cardinal direction
     */
    getWindDirection(deg) {
        const directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
        const index = Math.round(deg / 22.5) % 16;
        return directions[index];
    }

    /**
     * Cache weather data
     */
    cacheWeather(teamCode, weather) {
        this.cache.set(teamCode, {
            data: weather,
            timestamp: Date.now()
        });
    }

    /**
     * Get cached weather if still valid
     */
    getCachedWeather(teamCode) {
        const cached = this.cache.get(teamCode);
        if (!cached) return null;

        const age = Date.now() - cached.timestamp;
        if (age > this.cacheTimeout) {
            this.cache.delete(teamCode);
            return null;
        }

        return cached.data;
    }

    /**
     * Get weather for both teams in a game
     */
    async getGameWeather(awayTeam, homeTeam) {
        // Weather is based on the home team's stadium
        return await this.getStadiumWeather(homeTeam);
    }

    /**
     * Format weather for display
     */
    formatWeatherDisplay(weather) {
        if (!weather) {
            return 'Weather data unavailable';
        }

        return `
            <div class="weather-info">
                <div class="weather-main">
                    <span class="weather-temp">${weather.temp}°F</span>
                    <span class="weather-condition">${weather.description}</span>
                </div>
                <div class="weather-details">
                    <div>Feels like: ${weather.feelsLike}°F</div>
                    <div>Wind: ${weather.windSpeed} mph ${weather.windDirection}</div>
                    <div>Humidity: ${weather.humidity}%</div>
                    <div>Visibility: ${weather.visibility} mi</div>
                </div>
                <div class="weather-location">${weather.city}, ${weather.state}</div>
            </div>
        `;
    }

    /**
     * Get weather condition emoji
     */
    getWeatherEmoji(condition) {
        const emojiMap = {
            'Clear': '☀️',
            'Clouds': '☁️',
            'Rain': '🌧️',
            'Drizzle': '🌦️',
            'Thunderstorm': '⛈️',
            'Snow': '🌨️',
            'Mist': '🌫️',
            'Fog': '🌫️',
            'Haze': '🌫️'
        };
        return emojiMap[condition] || '🌤️';
    }
}

// Create global instance
window.WeatherAPI = new WeatherAPI();

console.log('✅ Weather API module loaded');
