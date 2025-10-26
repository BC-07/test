# Script to remove ResumeProcessor class from utils.py

$filePath = "utils.py"
$outputPath = "utils_cleaned.py"

# Read all lines
$lines = Get-Content $filePath

# Find the start and end of ResumeProcessor class
$startLine = -1
$endLine = -1
$inClass = $false

for ($i = 0; $i -lt $lines.Count; $i++) {
    $line = $lines[$i]
    
    # Find start of ResumeProcessor class
    if ($line -match "^class ResumeProcessor:") {
        $startLine = $i
        $inClass = $true
        Write-Host "Found ResumeProcessor class start at line $($i + 1)"
        continue
    }
    
    # Find end of ResumeProcessor class (next class definition)
    if ($inClass -and $line -match "^class PersonalDataSheetProcessor:") {
        $endLine = $i - 1
        $inClass = $false
        Write-Host "Found ResumeProcessor class end at line $($i)"
        break
    }
}

if ($startLine -ge 0 -and $endLine -ge 0) {
    Write-Host "Removing lines $($startLine + 1) to $($endLine + 1)"
    
    # Create new content without ResumeProcessor class
    $newLines = @()
    
    # Add lines before ResumeProcessor
    for ($i = 0; $i -lt $startLine; $i++) {
        $newLines += $lines[$i]
    }
    
    # Skip ResumeProcessor class lines
    # Add lines after ResumeProcessor
    for ($i = $endLine + 1; $i -lt $lines.Count; $i++) {
        $newLines += $lines[$i]
    }
    
    # Write cleaned file
    $newLines | Out-File -FilePath $outputPath -Encoding UTF8
    
    Write-Host "ResumeProcessor class removed successfully!"
    Write-Host "Original file had $($lines.Count) lines"
    Write-Host "Cleaned file has $($newLines.Count) lines"
    Write-Host "Removed $($lines.Count - $newLines.Count) lines"
} else {
    Write-Host "Could not find ResumeProcessor class boundaries"
}