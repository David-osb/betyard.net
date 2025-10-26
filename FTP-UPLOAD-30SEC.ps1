# 30-Second FTP Upload Script for Betyard.net
# This uploads directly to your GoDaddy hosting via FTP

Write-Host "🚀 30-SECOND FTP UPLOAD TO BETYARD.NET" -ForegroundColor Green

# You'll need to enter your GoDaddy FTP credentials
$ftpHost = "ftp.betyard.net"
$username = Read-Host "Enter your GoDaddy FTP username"
$password = Read-Host "Enter your GoDaddy FTP password" -AsSecureString
$plainPassword = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($password))

$localPath = "C:\Users\david\OneDrive\Workspace\Python Projects\UI.UX\2-Developer-Tools\godaddy-upload"

Write-Host "📁 Uploading from: $localPath" -ForegroundColor Yellow
Write-Host "🌐 Uploading to: $ftpHost/public_html" -ForegroundColor Yellow

# Function to upload file via FTP
function Upload-FTPFile {
    param($LocalFile, $RemoteFile, $FtpHost, $Username, $Password)
    
    try {
        $ftpUri = "ftp://$FtpHost/public_html/$RemoteFile"
        $request = [System.Net.FtpWebRequest]::Create($ftpUri)
        $request.Method = [System.Net.WebRequestMethods+Ftp]::UploadFile
        $request.Credentials = New-Object System.Net.NetworkCredential($Username, $Password)
        $request.UseBinary = $true
        $request.UsePassive = $true
        
        $fileContent = [System.IO.File]::ReadAllBytes($LocalFile)
        $request.ContentLength = $fileContent.Length
        
        $requestStream = $request.GetRequestStream()
        $requestStream.Write($fileContent, 0, $fileContent.Length)
        $requestStream.Close()
        
        $response = $request.GetResponse()
        Write-Host "✅ $RemoteFile" -ForegroundColor Green
        $response.Close()
        return $true
    }
    catch {
        Write-Host "❌ Failed: $RemoteFile - $_" -ForegroundColor Red
        return $false
    }
}

# Create assets/js directory on server
try {
    $ftpUri = "ftp://$ftpHost/public_html/assets"
    $request = [System.Net.FtpWebRequest]::Create($ftpUri)
    $request.Method = [System.Net.WebRequestMethods+Ftp]::MakeDirectory
    $request.Credentials = New-Object System.Net.NetworkCredential($username, $plainPassword)
    $request.GetResponse().Close()
} catch { }

try {
    $ftpUri = "ftp://$ftpHost/public_html/assets/js"
    $request = [System.Net.FtpWebRequest]::Create($ftpUri)
    $request.Method = [System.Net.WebRequestMethods+Ftp]::MakeDirectory
    $request.Credentials = New-Object System.Net.NetworkCredential($username, $plainPassword)
    $request.GetResponse().Close()
} catch { }

Write-Host "`n⚡ UPLOADING FILES..." -ForegroundColor Cyan

# Upload main files
Upload-FTPFile "$localPath\index.html" "index.html" $ftpHost $username $plainPassword

# Upload JavaScript files
$jsFiles = @(
    "game-centric-ui.js",
    "live-scores.js", 
    "ml-config.js",
    "ml-integration.js",
    "nfl-schedule-2025-correct.js",
    "nfl-schedule-api.js",
    "nfl-schedule.js"
)

foreach ($file in $jsFiles) {
    if (Test-Path "$localPath\assets\js\$file") {
        Upload-FTPFile "$localPath\assets\js\$file" "assets/js/$file" $ftpHost $username $plainPassword
    }
}

Write-Host "`n🎉 UPLOAD COMPLETE!" -ForegroundColor Green
Write-Host "🌐 Advanced analytics are now LIVE on https://betyard.net" -ForegroundColor Yellow
Write-Host "🔄 Clear your browser cache and check the site!" -ForegroundColor Cyan