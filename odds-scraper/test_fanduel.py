from fanduel_scraper import FanDuelScraper

scraper = FanDuelScraper()

print("Testing FanDuel API...")
print("="*60)

# Test 1: Get NFL events
print("\n1. Fetching NFL events...")
events = scraper.get_nfl_events()
print(f"   Found {len(events)} NFL events")

if events:
    print(f"\n   First event:")
    print(f"   - Name: {events[0].get('name')}")
    print(f"   - Home: {events[0].get('home_team')}")
    print(f"   - Away: {events[0].get('away_team')}")
    print(f"   - ID: {events[0].get('id')}")
else:
    print("   ❌ No events found or API blocked")

print("\n" + "="*60)
