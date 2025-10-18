#!/usr/bin/env pwsh
# Enhanced JavaScript Syntax Checker
Write-Host "🔧 Enhanced JavaScript Syntax Analysis" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

$htmlFile = "nfl-qb-predictor.html"

if (Test-Path $htmlFile) {
    $content = Get-Content $htmlFile -Raw
    
    # Extract JavaScript content between script tags
    $scriptPattern = '(?s)<script[^>]*>(.*?)</script>'
    $scripts = [regex]::Matches($content, $scriptPattern)
    
    Write-Host "`n📜 Found $($scripts.Count) script blocks" -ForegroundColor Yellow
    
    foreach ($i in 0..($scripts.Count - 1)) {
        $scriptContent = $scripts[$i].Groups[1].Value
        Write-Host "`n🔍 Analyzing Script Block $($i + 1):" -ForegroundColor Green
        
        # Count brackets in this script block
        $openBraces = ($scriptContent.ToCharArray() | Where-Object { $_ -eq '{' }).Count
        $closeBraces = ($scriptContent.ToCharArray() | Where-Object { $_ -eq '}' }).Count
        $openBrackets = ($scriptContent.ToCharArray() | Where-Object { $_ -eq '[' }).Count
        $closeBrackets = ($scriptContent.ToCharArray() | Where-Object { $_ -eq ']' }).Count
        $openParens = ($scriptContent.ToCharArray() | Where-Object { $_ -eq '(' }).Count
        $closeParens = ($scriptContent.ToCharArray() | Where-Object { $_ -eq ')' }).Count
        
        Write-Host "   Curly Braces { }: $openBraces vs $closeBraces" -ForegroundColor White
        Write-Host "   Square Brackets [ ]: $openBrackets vs $closeBrackets" -ForegroundColor White
        Write-Host "   Parentheses ( ): $openParens vs $closeParens" -ForegroundColor White
        
        if ($openBraces -ne $closeBraces) {
            Write-Host "   ❌ MISMATCH: Curly braces ($($openBraces - $closeBraces) difference)" -ForegroundColor Red
        }
        if ($openBrackets -ne $closeBrackets) {
            Write-Host "   ❌ MISMATCH: Square brackets ($($openBrackets - $closeBrackets) difference)" -ForegroundColor Red
        }
        if ($openParens -ne $closeParens) {
            Write-Host "   ❌ MISMATCH: Parentheses ($($openParens - $closeParens) difference)" -ForegroundColor Red
        }
        
        # Check for unterminated strings
        $singleQuotes = ($scriptContent.ToCharArray() | Where-Object { $_ -eq "'" }).Count
        $doubleQuotes = ($scriptContent.ToCharArray() | Where-Object { $_ -eq '"' }).Count
        $backticks = ($scriptContent.ToCharArray() | Where-Object { $_ -eq '`' }).Count
        
        Write-Host "   String Quotes - Single: $singleQuotes, Double: $doubleQuotes, Backticks: $backticks" -ForegroundColor Cyan
        
        if ($singleQuotes % 2 -ne 0) {
            Write-Host "   ⚠️  Odd number of single quotes (possible unterminated string)" -ForegroundColor Yellow
        }
        if ($doubleQuotes % 2 -ne 0) {
            Write-Host "   ⚠️  Odd number of double quotes (possible unterminated string)" -ForegroundColor Yellow
        }
        if ($backticks % 2 -ne 0) {
            Write-Host "   ⚠️  Odd number of backticks (possible unterminated template literal)" -ForegroundColor Yellow
        }
    }
    
    # Check for specific problematic patterns
    Write-Host "`n🔍 Checking for Common JavaScript Issues:" -ForegroundColor Green
    
    # Check for missing semicolons before closing braces
    if ($content -match '\w\s*\}' -and $content -notmatch ';\s*\}') {
        Write-Host "   ⚠️  Possible missing semicolon before closing brace" -ForegroundColor Yellow
    }
    
    # Check for missing commas in object literals
    if ($content -match '\w\s*\n\s*\w.*:' -and $content -notmatch ',\s*\n\s*\w.*:') {
        Write-Host "   ⚠️  Possible missing comma in object literal" -ForegroundColor Yellow
    }
    
    # Check for function definition issues
    $functionCount = ([regex]::Matches($content, 'function\s+\w+\s*\(')).Count
    $functionEndCount = ([regex]::Matches($content, '\}\s*(?=\s*(?:function|\s*</script>|\s*$))')).Count
    
    Write-Host "   Functions declared: $functionCount" -ForegroundColor Cyan
    Write-Host "   Function endings found: $functionEndCount" -ForegroundColor Cyan
    
} else {
    Write-Host "❌ File not found: $htmlFile" -ForegroundColor Red
}

Write-Host "`n💡 Recommendation:" -ForegroundColor Magenta
Write-Host "Use browser developer tools (F12) to see exact line numbers of any syntax errors." -ForegroundColor White
Write-Host "`n" -ForegroundColor White