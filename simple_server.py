#!/usr/bin/env python3
"""
Simple HTTP Server for testing NFL QB Predictor
This solves CORS issues by serving files over HTTP instead of file://
"""

import http.server
import socketserver
import webbrowser
import os
import sys

# Configuration
PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow API requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def main():
    try:
        # Change to the HTML directory
        os.chdir(DIRECTORY)
        
        with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
            server_url = f"http://localhost:{PORT}/UI.roughdraft2.html"
            
            print("🏈 NFL QB Predictor Server Starting...")
            print(f"📡 Server: http://localhost:{PORT}")
            print(f"🎯 NFL App: {server_url}")
            print("✅ CORS enabled for API access")
            print("🚀 Opening browser...")
            print("\n🛑 Press Ctrl+C to stop server\n")
            
            # Open browser automatically
            webbrowser.open(server_url)
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()