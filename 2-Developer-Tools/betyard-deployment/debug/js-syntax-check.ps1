#!/usr/bin/env pwsh
# JavaScript Syntax Checker for NFL QB Predictor
Write-Host "🔍 JavaScript Syntax Check for NFL QB Predictor" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

$htmlFile = "nfl-qb-predictor.html"

if (Test-Path $htmlFile) {
    Write-Host "`n📄 Checking: $htmlFile" -ForegroundColor Yellow
    
    # Extract JavaScript content and check for common syntax errors
    $content = Get-Content $htmlFile -Raw
    
    # Check for missing closing braces/brackets
    $openBraces = ($content.ToCharArray() | Where-Object { $_ -eq '{' }).Count
    $closeBraces = ($content.ToCharArray() | Where-Object { $_ -eq '}' }).Count
    $openBrackets = ($content.ToCharArray() | Where-Object { $_ -eq '[' }).Count
    $closeBrackets = ($content.ToCharArray() | Where-Object { $_ -eq ']' }).Count
    $openParens = ($content.ToCharArray() | Where-Object { $_ -eq '(' }).Count
    $closeParens = ($content.ToCharArray() | Where-Object { $_ -eq ')' }).Count
    
    Write-Host "`n🔧 Bracket Analysis:" -ForegroundColor Green
    Write-Host "   Curly Braces  { }: $openBraces open, $closeBraces close" -ForegroundColor White
    Write-Host "   Square Brackets [ ]: $openBrackets open, $closeBrackets close" -ForegroundColor White
    Write-Host "   Parentheses ( ): $openParens open, $closeParens close" -ForegroundColor White
    
    $syntaxIssues = @()
    
    if ($openBraces -ne $closeBraces) {
        $syntaxIssues += "❌ Mismatched curly braces: $openBraces vs $closeBraces"
    } else {
        Write-Host "   ✅ Curly braces match" -ForegroundColor Green
    }
    
    if ($openBrackets -ne $closeBrackets) {
        $syntaxIssues += "❌ Mismatched square brackets: $openBrackets vs $closeBrackets"
    } else {
        Write-Host "   ✅ Square brackets match" -ForegroundColor Green
    }
    
    if ($openParens -ne $closeParens) {
        $syntaxIssues += "❌ Mismatched parentheses: $openParens vs $closeParens"
    } else {
        Write-Host "   ✅ Parentheses match" -ForegroundColor Green
    }
    
    # Check for specific function definitions
    Write-Host "`n🔍 Function Check:" -ForegroundColor Green
    
    $functions = @(
        'toggleDarkTheme',
        'closeInjuryReportOverlay',
        'closeRecentStatsOverlay',
        'closePracticeReportOverlay',
        'closeBettingAnalysisOverlay'
    )
    
    foreach ($func in $functions) {
        if ($content -match "function $func\s*\(") {
            Write-Host "   ✅ $func function found" -ForegroundColor Green
        } else {
            $syntaxIssues += "❌ Missing function: $func"
        }
    }
    
    # Check for common syntax issues
    Write-Host "`n🔍 Common Issues Check:" -ForegroundColor Green
    
    if ($content -match "getElementById\s*\(\s*['\`"]theme-text['\`"]\s*\)") {
        $syntaxIssues += "❌ Reference to non-existent 'theme-text' element"
    } else {
        Write-Host "   ✅ No references to non-existent 'theme-text' element" -ForegroundColor Green
    }
    
    if ($content -match "</script>\s*</script>") {
        $syntaxIssues += "❌ Duplicate closing script tags found"
    } else {
        Write-Host "   ✅ No duplicate script tags" -ForegroundColor Green
    }
    
    # Summary
    Write-Host "`n📋 Summary:" -ForegroundColor Magenta
    if ($syntaxIssues.Count -eq 0) {
        Write-Host "   🎉 No syntax issues detected!" -ForegroundColor Green
        Write-Host "   ✅ File should load without JavaScript errors" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Found $($syntaxIssues.Count) potential issues:" -ForegroundColor Yellow
        foreach ($issue in $syntaxIssues) {
            Write-Host "      $issue" -ForegroundColor Red
        }
    }
    
} else {
    Write-Host "❌ File not found: $htmlFile" -ForegroundColor Red
}

Write-Host "`n🌐 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Open the file in browser and check console (F12)" -ForegroundColor White
Write-Host "2. Test the dark theme toggle button" -ForegroundColor White
Write-Host "3. Verify all interactive elements work" -ForegroundColor White

Write-Host "`n" -ForegroundColor White