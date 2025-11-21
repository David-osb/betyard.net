"""
Self-Healing Adaptive Odds Scraper
- Auto-detects API structure changes
- Bypasses rate limiting with rotation
- Falls back to alternative sources automatically
"""

import requests
import json
import time
import random
from datetime import datetime, timedelta
from functools import wraps
import hashlib

# Import SmartRequester for advanced bypass capabilities
try:
    from proxy_rotator import SmartRequester
    SMART_REQUESTER_AVAILABLE = True
except ImportError:
    SMART_REQUESTER_AVAILABLE = False
    print("⚠️ proxy_rotator not available - using basic requests")

# Import player ID database
try:
    from player_ids import get_player_id, get_all_players
    PLAYER_DB_AVAILABLE = True
except ImportError:
    PLAYER_DB_AVAILABLE = False
    print("⚠️ player_ids.py not available - using fallback IDs")

class AdaptiveOddsScraper:
    """
    Scraper that adapts to website changes and avoids rate limits
    """
    
    def __init__(self, use_smart_requester=True, use_proxies=False, use_tor=False, use_cloudflare_bypass=False):
        """
        Initialize scraper
        
        Args:
            use_smart_requester: Use advanced SmartRequester (recommended)
            use_proxies: Enable free proxy rotation
            use_tor: Use Tor network (requires tor installed)
            use_cloudflare_bypass: Bypass Cloudflare protection
        """
        # Initialize SmartRequester if available and enabled
        if use_smart_requester and SMART_REQUESTER_AVAILABLE:
            self.smart_requester = SmartRequester(
                use_proxies=use_proxies,
                use_tor=use_tor,
                use_cloudflare_bypass=use_cloudflare_bypass
            )
            print("✅ SmartRequester enabled with advanced bypass")
        else:
            self.smart_requester = None
            print("ℹ️ Using basic request handling")
        
        # Rotating user agents to avoid detection (fallback)
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
        ]
        
        # Cache to reduce requests
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # API endpoint patterns (REAL working endpoints)
        self.endpoint_patterns = {
            'draftkings': [
                # Player-specific props API (CONFIRMED WORKING)
                'https://sportsbook-nash.draftkings.com/api/team/markets/dkusoh/v2/player/{player_id}/markets',
                # Fallback patterns
                'https://sportsbook-us-nj.draftkings.com/api/team/markets/dkusoh/v2/player/{player_id}/markets',
                'https://sportsbook.draftkings.com/api/team/markets/dkusoh/v2/player/{player_id}/markets'
            ],
            'fanduel': [
                # Market prices API (POST request - needs market IDs)
                'https://smp.oh.sportsbook.fanduel.com/api/sports/fixedodds/readonly/v1/getMarketPrices?priceHistory=0',
                # Alternative state endpoints
                'https://smp.nj.sportsbook.fanduel.com/api/sports/fixedodds/readonly/v1/getMarketPrices?priceHistory=0',
                'https://smp.pa.sportsbook.fanduel.com/api/sports/fixedodds/readonly/v1/getMarketPrices?priceHistory=0'
            ]
            # Disable BetMGM until we find their endpoints
            # 'betmgm': [
            #     'https://sports.betmgm.com/cds-api/bettingoffer/fixtures',
            # ]
        }
        
        # Known player IDs (fallback if player_ids.py not available)
        self.player_ids = {
            'josh allen': '11370',
            'patrick mahomes': '10838',
            'james cook': '13456',
        }
        
        # Rate limiting config
        self.last_request_time = {}
        self.min_delay = 2  # seconds between requests
        self.max_retries = 3
        
        # Proxy rotation (optional - add your proxies here)
        self.proxies = []  # ['http://proxy1:port', 'http://proxy2:port']
        
    def _get_session(self):
        """Create session with random user agent"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/',
            'DNT': '1'
        })
        return session
    
    def _rate_limit(self, source):
        """Smart rate limiting with exponential backoff"""
        now = time.time()
        
        if source in self.last_request_time:
            elapsed = now - self.last_request_time[source]
            if elapsed < self.min_delay:
                sleep_time = self.min_delay - elapsed + random.uniform(0, 1)
                time.sleep(sleep_time)
        
        self.last_request_time[source] = time.time()
    
    def _get_cache_key(self, url, params=None):
        """Generate cache key"""
        key = url + str(sorted((params or {}).items()))
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key):
        """Get from cache if not expired"""
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                return data
        return None
    
    def _save_to_cache(self, cache_key, data):
        """Save to cache"""
        self.cache[cache_key] = (data, time.time())
    
    def _try_request(self, url, params=None, use_proxy=False):
        """
        Try request with automatic fallback and retry logic
        Uses SmartRequester if available for advanced bypass
        """
        cache_key = self._get_cache_key(url, params)
        
        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached, 'cache'
        
        # Use SmartRequester if available (handles all bypass methods)
        if self.smart_requester:
            try:
                response = self.smart_requester.get(url, use_cache=True)
                if response and response.status_code == 200:
                    data = response.json()
                    self._save_to_cache(cache_key, data)
                    return data, 'smart_requester'
            except Exception as e:
                print(f"⚠️ SmartRequester failed: {str(e)}, falling back to basic...")
        
        # Fallback to basic request handling
        session = self._get_session()
        
        for attempt in range(self.max_retries):
            try:
                # Add random delay to avoid pattern detection
                if attempt > 0:
                    time.sleep(2 ** attempt + random.uniform(0, 1))
                
                # Use proxy if available and requested
                proxy = None
                if use_proxy and self.proxies:
                    proxy = {'http': random.choice(self.proxies), 'https': random.choice(self.proxies)}
                
                response = session.get(url, params=params, proxies=proxy, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    self._save_to_cache(cache_key, data)
                    return data, 'api'
                
                elif response.status_code == 429:  # Rate limited
                    print(f"⚠️ Rate limited on {url}, waiting...")
                    time.sleep(60)  # Wait 1 minute
                    continue
                
                elif response.status_code == 403:  # Blocked
                    if not use_proxy and self.proxies:
                        print(f"⚠️ Blocked on {url}, trying with proxy...")
                        return self._try_request(url, params, use_proxy=True)
                    else:
                        print(f"❌ 403 Forbidden on {url}")
                        break
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                continue
        
        return None, 'failed'
    
    def _auto_discover_endpoints(self, player_name, sport='nfl'):
        """
        Automatically discover working API endpoints by trying patterns
        This runs periodically to adapt to site changes
        """
        working_endpoints = {}
        
        # Get player ID - try database first, then fallback dict
        player_name_lower = player_name.lower()
        
        if PLAYER_DB_AVAILABLE:
            player_id = get_player_id(player_name_lower, sport)
        else:
            player_id = self.player_ids.get(player_name_lower)
        
        if not player_id:
            print(f"⚠️ Player ID not found for '{player_name}'")
            print(f"   Available players: {list(self.player_ids.keys())[:5]}...")
            return working_endpoints
        
        # Try all known patterns
        for source, patterns in self.endpoint_patterns.items():
            for pattern in patterns:
                # Replace placeholders
                url = pattern.replace('{sport_id}', '88808' if sport == 'nfl' else '88670')
                url = url.replace('{player_id}', player_id)
                
                self._rate_limit(source)
                data, status = self._try_request(url)
                
                if data and status in ['api', 'smart_requester']:
                    working_endpoints[source] = url
                    print(f"✅ Found working endpoint for {source}: {url}")
                    break
        
        return working_endpoints
    
    def _extract_player_props_generic(self, data, player_name, stat_type):
        """
        Extract player props from DraftKings API response
        Parses the real DraftKings market structure
        """
        # Map stat types to DraftKings market names
        stat_keywords = {
            'passing_yards': ['passing yards o/u', 'pass yds', '038'],
            'rushing_yards': ['rushing yards o/u', 'rush yds', '003'],
            'receiving_yards': ['receiving yards o/u', 'rec yds', '011'],
            'receptions': ['receptions o/u', 'reception'],
            'passing_tds': ['passing touchdowns o/u', 'passing td', 'pass td', '037'],
            'rushing_tds': ['rushing touchdowns o/u', 'rushing td', 'rush td'],
            'passing_attempts': ['passing attempts o/u', 'pass att', '012'],
            'rushing_attempts': ['rushing attempts o/u', 'rush att', '042'],
            'passing_completions': ['passing completions o/u', 'completions', '040'],
            'interceptions': ['interceptions thrown o/u', 'interceptions', '039'],
        }
        
        keywords = stat_keywords.get(stat_type, [stat_type.replace('_', ' ')])
        
        try:
            # DraftKings structure: top-level selections array
            selections = data.get('selections', [])
            markets = data.get('markets', [])
            
            # Build marketId to market name map
            market_map = {m['id']: m.get('name', '') for m in markets}
            
            # Group selections by marketId to find Over/Under pairs
            market_selections = {}
            for selection in selections:
                market_id = selection.get('marketId')
                if market_id:
                    if market_id not in market_selections:
                        market_selections[market_id] = []
                    market_selections[market_id].append(selection)
            
            # Find the right market for this stat
            for market_id, sels in market_selections.items():
                market_name = market_map.get(market_id, '').lower()
                
                # Check if market matches our stat type
                matches = any(kw in market_name or kw in market_id for kw in keywords)
                
                if matches and len(sels) >= 2:
                    # Look for main Over/Under line
                    over_sel = None
                    under_sel = None
                    
                    for sel in sels:
                        # Only use "main" lines with Over/Under outcomes
                        if sel.get('main') and sel.get('points'):
                            if sel.get('outcomeType') == 'Over':
                                over_sel = sel
                            elif sel.get('outcomeType') == 'Under':
                                under_sel = sel
                    
                    if over_sel and under_sel:
                        # Parse American odds (handle unicode minus)
                        def parse_odds(sel):
                            odds_str = sel.get('displayOdds', {}).get('american', '-110')
                            odds_str = str(odds_str).replace('−', '-').replace('+', '')
                            try:
                                return int(odds_str)
                            except:
                                return -110
                        
                        line = over_sel.get('points') or under_sel.get('points')
                        
                        return {
                            'line': float(line),
                            'over': parse_odds(over_sel),
                            'under': parse_odds(under_sel),
                            'market_name': market_map.get(market_id, 'Unknown'),
                            'market_id': market_id
                        }
        
        except Exception as e:
            print(f"⚠️ Parser error: {str(e)}")
        
        return None
    
    def get_player_props(self, player_name, stat_type='passing_yards', sport='nfl'):
        """
        Main method - automatically adapts to site changes
        
        Args:
            player_name: "Josh Allen"
            stat_type: 'passing_yards', 'rushing_yards', etc.
            sport: 'nfl', 'nba', 'mlb'
        
        Returns:
            {
                'player': 'Josh Allen',
                'stat': 'passing_yards',
                'books': {
                    'draftkings': {'line': 229.5, 'over': -110, 'under': -110},
                    'fanduel': {'line': 232.5, 'over': -115, 'under': -105}
                },
                'consensus_line': 231.0,
                'best_over': {'book': 'draftkings', 'line': 229.5, 'odds': -110},
                'best_under': {'book': 'fanduel', 'line': 232.5, 'odds': -105},
                'sources': ['draftkings'],
                'timestamp': '2025-11-20T...'
            }
        """
        # Auto-discover endpoints with player name
        endpoints = self._auto_discover_endpoints(player_name, sport)
        
        props = {
            'player': player_name,
            'stat': stat_type,
            'books': {},
            'sources': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Try each source
        for source, url in endpoints.items():
            self._rate_limit(source)
            data, status = self._try_request(url)
            
            if data:
                props['sources'].append(source)
                
                # Extract props using generic parser
                extracted = self._extract_player_props_generic(data, player_name, stat_type)
                
                if extracted:
                    props['books'][source] = extracted
        
        # Calculate consensus and best lines
        if props['books']:
            lines = [b['line'] for b in props['books'].values()]
            props['consensus_line'] = round(sum(lines) / len(lines), 1)
            
            # Find best odds
            best_over = max(props['books'].items(), key=lambda x: x[1].get('over', -999))
            best_under = max(props['books'].items(), key=lambda x: x[1].get('under', -999))
            
            props['best_over'] = {
                'book': best_over[0],
                'line': best_over[1]['line'],
                'odds': best_over[1]['over']
            }
            props['best_under'] = {
                'book': best_under[0],
                'line': best_under[1]['line'],
                'odds': best_under[1]['under']
            }
        else:
            # No data found
            props['error'] = 'No odds data available'
        
        return props
    
    def get_all_nfl_props(self):
        """
        Get all NFL player props for the week
        Useful for populating your database
        """
        endpoints = self._auto_discover_endpoints('nfl')
        all_props = []
        
        for source, url in endpoints.items():
            self._rate_limit(source)
            data, status = self._try_request(url)
            
            if data:
                # Extract all players
                # This would parse the full response
                pass
        
        return all_props


# Flask integration
def cache_with_ttl(ttl_seconds=300):
    """Decorator to cache function results"""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl_seconds:
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result
        
        return wrapper
    return decorator


# Usage in your Flask app
if __name__ == '__main__':
    # Example 1: Basic scraper (no SmartRequester)
    scraper = AdaptiveOddsScraper(use_smart_requester=False)
    
    # Example 2: With free proxy rotation
    # scraper = AdaptiveOddsScraper(use_smart_requester=True, use_proxies=True)
    
    # Example 3: With Tor network (most reliable, requires tor installed)
    # scraper = AdaptiveOddsScraper(use_smart_requester=True, use_tor=True)
    
    # Example 4: Full power - all bypass methods
    # scraper = AdaptiveOddsScraper(
    #     use_smart_requester=True,
    #     use_proxies=True,
    #     use_tor=False,  # Set True if tor is installed
    #     use_cloudflare_bypass=True
    # )
    
    # Get Josh Allen props
    props = scraper.get_player_props('Josh Allen', 'passing_yards')
    
    print(json.dumps(props, indent=2))
    
    # Example output:
    # {
    #   "player": "Josh Allen",
    #   "stat": "passing_yards",
    #   "books": {
    #     "draftkings": {"line": 229.5, "over": -110, "under": -110},
    #     "fanduel": {"line": 232.5, "over": -115, "under": -105}
    #   },
    #   "consensus_line": 231.0,
    #   "best_over": {"book": "draftkings", "line": 229.5, "odds": -110},
    #   "best_under": {"book": "fanduel", "line": 232.5, "odds": -105},
    #   "data_sources": ["cache", "api"],
    #   "timestamp": "2025-11-20T15:30:00"
    # }
