# GoDaddy FTP Upload Script for Betyard.net
# This script uploads files directly to your GoDaddy hosting

Write-Host "🚀 Starting direct upload to betyard.net..." -ForegroundColor Green

# FTP Connection Details (you'll need to update these with your actual credentials)
$ftpServer = "ftp://betyard.net"
$username = "your_godaddy_username"
$password = "your_godaddy_password"

# Source and target paths
$localPath = "C:\Users\david\OneDrive\Workspace\Python Projects\UI.UX\2-Developer-Tools\godaddy-upload"
$remotePath = "public_html"

Write-Host "📁 Uploading from: $localPath" -ForegroundColor Yellow
Write-Host "🌐 Uploading to: $ftpServer/$remotePath" -ForegroundColor Yellow

# Function to upload file via FTP
function Upload-File {
    param(
        [string]$LocalFile,
        [string]$RemoteFile,
        [string]$FtpUri,
        [string]$Username,
        [string]$Password
    )
    
    try {
        $request = [System.Net.FtpWebRequest]::Create($FtpUri + "/" + $RemoteFile)
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
        Write-Host "✅ Uploaded: $RemoteFile" -ForegroundColor Green
        $response.Close()
        return $true
    }
    catch {
        Write-Host "❌ Failed to upload $RemoteFile : $_" -ForegroundColor Red
        return $false
    }
}

# Upload index.html
Write-Host "📄 Uploading index.html..." -ForegroundColor Cyan
Upload-File -LocalFile "$localPath\index.html" -RemoteFile "index.html" -FtpUri $ftpServer -Username $username -Password $password

# Upload CNAME
Write-Host "🌐 Uploading CNAME..." -ForegroundColor Cyan
Upload-File -LocalFile "$localPath\CNAME" -RemoteFile "CNAME" -FtpUri $ftpServer -Username $username -Password $password

# Upload all JavaScript files
Write-Host "📦 Uploading JavaScript files..." -ForegroundColor Cyan
$jsFiles = Get-ChildItem "$localPath\assets\js\*.js" -File
foreach ($file in $jsFiles) {
    $remotePath = "assets/js/$($file.Name)"
    Upload-File -LocalFile $file.FullName -RemoteFile $remotePath -FtpUri $ftpServer -Username $username -Password $password
}

Write-Host "🎉 Upload completed! Advanced analytics should now be live on betyard.net" -ForegroundColor Green
Write-Host "🔄 Please check https://betyard.net in 1-2 minutes for the new features" -ForegroundColor Yellow