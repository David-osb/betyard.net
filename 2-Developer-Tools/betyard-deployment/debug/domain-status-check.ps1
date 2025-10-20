#!/usr/bin/env pwsh
# BetYard Domain Status Checker
# This script helps monitor your domain transition from GoDaddy to GitHub Pages

Write-Host "🌐 BetYard.net Domain Status Check" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Check DNS Resolution
Write-Host "`n📡 DNS Resolution:" -ForegroundColor Yellow
try {
    $dnsResult = Resolve-DnsName betyard.net -Type A -ErrorAction Stop
    Write-Host "Current IP Address: $($dnsResult.IPAddress)" -ForegroundColor Green
    
    # Check if pointing to GitHub Pages
    $githubIPs = @('185.199.108.153', '185.199.109.153', '185.199.110.153', '185.199.111.153')
    if ($dnsResult.IPAddress -in $githubIPs) {
        Write-Host "✅ Domain is pointing to GitHub Pages!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Domain is NOT pointing to GitHub Pages yet" -ForegroundColor Yellow
        Write-Host "   This is normal if you just changed DNS settings" -ForegroundColor Gray
    }
} catch {
    Write-Host "❌ DNS Resolution failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Check Website Response
Write-Host "`n🌐 Website Status:" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "https://betyard.net" -Method Head -TimeoutSec 10 -ErrorAction Stop
    Write-Host "HTTP Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "Server: $($response.Headers.Server)" -ForegroundColor Green
    
    # Check if it's GitHub Pages
    if ($response.Headers.Server -like "*github*" -or $response.Headers["x-github-request-id"]) {
        Write-Host "✅ Website is served by GitHub Pages!" -ForegroundColor Green
    } else {
        Write-Host "ℹ️  Website is served by: $($response.Headers.Server)" -ForegroundColor Cyan
    }
} catch {
    Write-Host "❌ Website check failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Check CNAME
Write-Host "`n📄 CNAME Configuration:" -ForegroundColor Yellow
$cnamePath = "betyard-deployment\CNAME"
if (Test-Path $cnamePath) {
    $cnameContent = Get-Content $cnamePath -Raw
    Write-Host "CNAME file content: '$($cnameContent.Trim())'" -ForegroundColor Green
    
    if ($cnameContent.Trim() -eq "betyard.net") {
        Write-Host "✅ CNAME file is correctly configured" -ForegroundColor Green
    } else {
        Write-Host "⚠️  CNAME file may need adjustment" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ CNAME file not found in betyard-deployment folder" -ForegroundColor Red
}

# Instructions
Write-Host "`n📋 Next Steps:" -ForegroundColor Magenta
Write-Host "1. If DNS is not pointing to GitHub yet:" -ForegroundColor White
Write-Host "   → Update GoDaddy DNS with GitHub IP addresses" -ForegroundColor Gray
Write-Host "2. If GitHub Pages is not enabled:" -ForegroundColor White
Write-Host "   → Visit: https://github.com/David-osb/betyard.net/settings/pages" -ForegroundColor Gray
Write-Host "3. Wait 24-48 hours for full DNS propagation" -ForegroundColor White

Write-Host "`n🔄 Rerun this script to check progress:" -ForegroundColor Cyan
Write-Host "   .\domain-status-check.ps1" -ForegroundColor Gray

Write-Host "`n" -ForegroundColor White