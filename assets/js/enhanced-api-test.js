/**
 * 🧪 ENHANCED NFL API TESTING SUITE
 * Comprehensive testing of all Tank01 endpoints with real API key
 * Author: GitHub Copilot
 * Version: 1.0.0
 */

class EnhancedAPITester {
    constructor() {
        this.apiKey = 'be76a86c9cmsh0d0cecaaefbc722p1efcdbjsn598e66d34cf3';
        this.baseUrl = 'https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com';
        this.testResults = [];
        
        this.endpoints = [
            // VERIFIED WORKING ENDPOINTS
            { name: 'Daily Schedule', endpoint: '/getNFLGamesForDate', params: { gameDate: '2025-10-19' } },
            { name: 'Team Roster', endpoint: '/getNFLTeamRoster', params: { teamID: 'PHI', getStats: 'false' } },
            { name: 'Game Info', endpoint: '/getNFLGameInfo', params: { gameID: 'sample_game_id' } },
            { name: 'Team Schedule', endpoint: '/getNFLTeamSchedule', params: { teamID: 'BAL', season: '2025' } },
            { name: 'Box Score', endpoint: '/getNFLBoxScore', params: { gameID: 'sample_box_score' } },
            
            // HIGH-VALUE ENDPOINT (5.2MB of data!)
            { name: 'Player List', endpoint: '/getNFLPlayerList', params: { playerStats: 'true', getStats: 'true' } },
            
            // Additional verified variations (removed non-working /getNFLScores)
            { name: 'Team Roster (KC)', endpoint: '/getNFLTeamRoster', params: { teamID: 'KC', getStats: 'true' } },
            { name: 'Team Roster (LV)', endpoint: '/getNFLTeamRoster', params: { teamID: 'LV', getStats: 'false' } },
            { name: 'Team Roster (CIN)', endpoint: '/getNFLTeamRoster', params: { teamID: 'CIN', getStats: 'false' } },
            { name: 'Player List (Minimal)', endpoint: '/getNFLPlayerList', params: { playerStats: 'false', getStats: 'false' } },
            
            // Test additional working endpoints discovered
            { name: 'Team Schedule (PHI)', endpoint: '/getNFLTeamSchedule', params: { teamID: 'PHI', season: '2025' } },
            { name: 'Team Schedule (KC)', endpoint: '/getNFLTeamSchedule', params: { teamID: 'KC', season: '2025' } }
        ];
        
        this.init();
    }
    
    init() {
        console.log('🧪 Enhanced API Tester: Starting comprehensive Tank01 endpoint testing...');
        console.log('📊 Tests will run in background - check console for results');
        // No visual interface - console only
        this.runAllTests();
    }
    
    async runAllTests() {
        console.log('🔄 Running comprehensive API tests...');
        
        let successCount = 0;
        let totalTests = this.endpoints.length;
        
        for (const test of this.endpoints) {
            const result = await this.testEndpoint(test);
            this.testResults.push(result);
            
            const statusIcon = result.success ? '✅' : (result.rateLimited ? '⚠️' : '❌');
            
            // Log to console only
            console.log(`${statusIcon} ${test.name}: ${result.message}${result.dataSize ? ` (${result.dataSize} bytes)` : ''}`);
            
            if (result.success) successCount++;
            
            // Add delay to respect rate limits
            await this.delay(1000);
        }
        
        // Console summary only
        const successRate = Math.round((successCount / totalTests) * 100);
        console.log('\n📊 Test Summary:');
        console.log(`Success Rate: ${successRate}% (${successCount}/${totalTests})`);
        console.log(`API Key Status: ${successCount > 0 ? '🟢 Active' : '🔴 Issues Detected'}`);
        console.log(`Tank01 Integration: ${successCount > 5 ? '🟢 Excellent' : successCount > 2 ? '🟡 Good' : '🔴 Limited'}\n`);
        
        console.log('🧪 API Testing Complete:', this.testResults);
        this.generateAPIReport();
    }
    
    async testEndpoint(test) {
        try {
            const url = this.buildURL(test.endpoint, test.params);
            const startTime = Date.now();
            
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'X-RapidAPI-Key': this.apiKey,
                    'X-RapidAPI-Host': 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com'
                }
            });
            
            const duration = Date.now() - startTime;
            
            if (response.status === 429) {
                return {
                    name: test.name,
                    success: false,
                    rateLimited: true,
                    message: `Rate limited (429) - ${duration}ms`,
                    status: response.status
                };
            }
            
            if (response.status === 403) {
                return {
                    name: test.name,
                    success: false,
                    rateLimited: true,
                    message: `Forbidden (403) - API quota or permissions - ${duration}ms`,
                    status: response.status
                };
            }
            
            if (response.ok) {
                const data = await response.json();
                const dataSize = JSON.stringify(data).length;
                
                return {
                    name: test.name,
                    success: true,
                    rateLimited: false,
                    message: `Success (${response.status}) - ${duration}ms`,
                    status: response.status,
                    duration,
                    dataSize,
                    data: data
                };
            } else {
                return {
                    name: test.name,
                    success: false,
                    rateLimited: false,
                    message: `HTTP ${response.status} - ${duration}ms`,
                    status: response.status,
                    duration
                };
            }
            
        } catch (error) {
            return {
                name: test.name,
                success: false,
                rateLimited: false,
                message: `Network error: ${error.message}`,
                error: error.message
            };
        }
    }
    
    buildURL(endpoint, params) {
        const url = new URL(endpoint, this.baseUrl);
        Object.keys(params).forEach(key => {
            url.searchParams.append(key, params[key]);
        });
        return url.toString();
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    generateAPIReport() {
        const report = {
            timestamp: new Date().toISOString(),
            apiKey: this.apiKey.substring(0, 8) + '...',
            totalTests: this.testResults.length,
            successfulTests: this.testResults.filter(r => r.success).length,
            rateLimitedTests: this.testResults.filter(r => r.rateLimited).length,
            averageResponseTime: this.calculateAverageResponseTime(),
            workingEndpoints: this.testResults.filter(r => r.success).map(r => r.name),
            recommendations: this.generateRecommendations()
        };
        
        console.log('📋 Comprehensive API Report:', report);
        
        // Store report for later access
        window.nflAPIReport = report;
        
        return report;
    }
    
    calculateAverageResponseTime() {
        const successfulTests = this.testResults.filter(r => r.success && r.duration);
        if (successfulTests.length === 0) return 0;
        
        const totalTime = successfulTests.reduce((sum, test) => sum + test.duration, 0);
        return Math.round(totalTime / successfulTests.length);
    }
    
    generateRecommendations() {
        const recommendations = [];
        const successCount = this.testResults.filter(r => r.success).length;
        const rateLimitCount = this.testResults.filter(r => r.rateLimited).length;
        
        if (successCount === 0) {
            recommendations.push('🔴 API key appears to be invalid or quota exhausted');
        } else if (successCount < 3) {
            recommendations.push('🟡 Limited API access - consider upgrading plan');
        } else {
            recommendations.push('🟢 Good API access - maximize with intelligent caching');
        }
        
        if (rateLimitCount > 5) {
            recommendations.push('⚠️ High rate limiting - implement request queuing');
        }
        
        recommendations.push('💡 Use comprehensive caching to minimize API calls');
        recommendations.push('🎯 Focus on priority endpoints for game-day data');
        
        return recommendations;
    }
}

// Auto-start when page loads
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        window.enhancedAPITester = new EnhancedAPITester();
    }, 2000); // Wait 2 seconds for other systems to initialize
});

// Export for manual testing
window.testNFLAPI = () => {
    if (window.enhancedAPITester) {
        window.enhancedAPITester.runAllTests();
    } else {
        window.enhancedAPITester = new EnhancedAPITester();
    }
};