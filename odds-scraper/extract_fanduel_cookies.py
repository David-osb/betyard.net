"""
FanDuel Cookie Extractor
Run this once to save your browser cookies for automated scraping
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pickle
import time
import os

def extract_fanduel_cookies():
    """
    Opens Chrome to FanDuel, waits for you to log in, then saves cookies
    """
    print("="*60)
    print("FanDuel Cookie Extractor")
    print("="*60)
    
    # Setup Chrome with normal profile
    chrome_options = Options()
    chrome_options.add_argument('--start-maximized')
    
    # Optional: Use your existing Chrome profile (keeps you logged in)
    # Uncomment these lines to use your real Chrome profile:
    # user_data_dir = os.path.expanduser('~\\AppData\\Local\\Google\\Chrome\\User Data')
    # chrome_options.add_argument(f'user-data-dir={user_data_dir}')
    # chrome_options.add_argument('profile-directory=Default')
    
    print("\n1. Opening Chrome...")
    driver = webdriver.Chrome(options=chrome_options)
    
    print("2. Navigating to FanDuel Sportsbook...")
    driver.get('https://sportsbook.fanduel.com/navigation/nfl')
    
    print("\n" + "="*60)
    print("PLEASE LOG IN TO FANDUEL IN THE BROWSER WINDOW")
    print("="*60)
    print("\nSteps:")
    print("1. Click 'Login' button")
    print("2. Enter your email/password")
    print("3. Complete any 2FA if required")
    print("4. Wait for the NFL page to fully load")
    print("\nOnce logged in and page is loaded, press ENTER here...")
    print("="*60)
    
    input("\n>>> Press ENTER after logging in: ")
    
    # Give a moment for any final loads
    time.sleep(2)
    
    print("\n3. Extracting cookies...")
    cookies = driver.get_cookies()
    
    # Save cookies to file
    cookie_file = 'fanduel_cookies.pkl'
    with open(cookie_file, 'wb') as f:
        pickle.dump(cookies, f)
    
    print(f"✅ Saved {len(cookies)} cookies to {cookie_file}")
    
    # Also save as JSON for inspection
    import json
    with open('fanduel_cookies.json', 'w') as f:
        json.dump(cookies, f, indent=2)
    
    print(f"✅ Also saved as fanduel_cookies.json (for inspection)")
    
    # Test if cookies work
    print("\n4. Testing cookies...")
    print("Attempting to fetch NFL events with saved cookies...")
    
    driver.quit()
    
    # Test with requests
    import requests
    session = requests.Session()
    
    # Load cookies into session
    for cookie in cookies:
        session.cookies.set(cookie['name'], cookie['value'])
    
    # Copy headers from browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://sportsbook.fanduel.com/'
    }
    session.headers.update(headers)
    
    # Test API call
    test_url = 'https://sportsbook.fanduel.com/api/content-managed-page'
    params = {
        'page': 'SPORT',
        'eventTypeId': '6423',
        '_ak': 'FhMFpcPWXMeyZxOx',
        'timezone': 'America/New_York'
    }
    
    try:
        response = session.get(test_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'attachments' in data:
                print("✅ SUCCESS! Cookies are working!")
                print(f"✅ API returned valid data")
                
                # Count events
                events = data.get('attachments', {}).get('events', {})
                print(f"✅ Found {len(events)} NFL events")
            else:
                print("⚠️  Got response but unexpected format")
                print(f"Status: {response.status_code}")
        else:
            print(f"⚠️  Got status code: {response.status_code}")
            print("Cookies may need refresh - try running again")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("You may need to run this script again")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nYour FanDuel cookies are saved and ready to use.")
    print("\nNext steps:")
    print("1. Run your scraper scripts - they'll use these cookies automatically")
    print("2. Cookies last 30-90 days typically")
    print("3. If scraper stops working, just run this script again")
    print("\n" + "="*60)

if __name__ == '__main__':
    extract_fanduel_cookies()
