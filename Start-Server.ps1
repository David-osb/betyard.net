# Simple PowerShell HTTP Server for NFL QB Predictor
# This solves CORS issues by serving over HTTP instead of file://

param(
    [int]$Port = 8000
)

$Directory = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Url = "http://localhost:$Port/"

Write-Host "🏈 NFL QB Predictor Server Starting..." -ForegroundColor Green
Write-Host "📡 Server: $Url" -ForegroundColor Cyan
Write-Host "🎯 NFL App: ${Url}UI.roughdraft2.html" -ForegroundColor Yellow
Write-Host "✅ CORS headers enabled for API access" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 Opening browser automatically..." -ForegroundColor Magenta
Write-Host "🛑 Press Ctrl+C to stop server" -ForegroundColor Red
Write-Host ""

# Start simple HTTP server using .NET HttpListener
Add-Type -AssemblyName System.Net.Http

try {
    $listener = New-Object System.Net.HttpListener
    $listener.Prefixes.Add($Url)
    $listener.Start()
    
    # Open browser
    Start-Process "${Url}UI.roughdraft2.html"
    
    Write-Host "✅ Server running at $Url" -ForegroundColor Green
    Write-Host "📂 Serving from: $Directory" -ForegroundColor Gray
    
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $request = $context.Request
        $response = $context.Response
        
        # Add CORS headers
        $response.Headers.Add("Access-Control-Allow-Origin", "*")
        $response.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        $response.Headers.Add("Access-Control-Allow-Headers", "*")
        
        $requestedFile = $request.Url.LocalPath.TrimStart('/')
        if ($requestedFile -eq '') { $requestedFile = 'UI.roughdraft2.html' }
        
        $filePath = Join-Path $Directory $requestedFile
        
        if (Test-Path $filePath) {
            $content = Get-Content $filePath -Raw -Encoding UTF8
            $buffer = [System.Text.Encoding]::UTF8.GetBytes($content)
            
            # Set content type
            if ($requestedFile.EndsWith('.html')) {
                $response.ContentType = "text/html; charset=utf-8"
            } elseif ($requestedFile.EndsWith('.js')) {
                $response.ContentType = "application/javascript"
            } elseif ($requestedFile.EndsWith('.css')) {
                $response.ContentType = "text/css"
            }
            
            $response.ContentLength64 = $buffer.Length
            $response.OutputStream.Write($buffer, 0, $buffer.Length)
        } else {
            $response.StatusCode = 404
            $buffer = [System.Text.Encoding]::UTF8.GetBytes("File not found: $requestedFile")
            $response.ContentLength64 = $buffer.Length
            $response.OutputStream.Write($buffer, 0, $buffer.Length)
        }
        
        $response.OutputStream.Close()
    }
} catch {
    Write-Host "❌ Server error: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    if ($listener) {
        $listener.Stop()
        Write-Host "🛑 Server stopped" -ForegroundColor Yellow
    }
}