# Script to remove resume-related methods from app.py

$filePath = "app.py"
$outputPath = "app_cleaned.py"

# Read all lines
$lines = Get-Content $filePath

# Find and remove upload_resumes method
$newLines = @()
$skipLines = $false
$methodDepth = 0

for ($i = 0; $i -lt $lines.Count; $i++) {
    $line = $lines[$i]
    
    # Check if we're starting to skip the upload_resumes method
    if ($line -match "def upload_resumes\(") {
        $skipLines = $true
        $methodDepth = 1
        Write-Host "Found upload_resumes method at line $($i + 1) - skipping"
        continue
    }
    
    # Check if we're starting to skip the _process_resume_for_job method  
    if ($line -match "def _process_resume_for_job\(") {
        $skipLines = $true
        $methodDepth = 1
        Write-Host "Found _process_resume_for_job method at line $($i + 1) - skipping"
        continue
    }
    
    # If we're skipping, check for the end of the method
    if ($skipLines) {
        # Count indentation to find method end
        if ($line -match "^\s*def \w+\(" -and -not ($line -match "def upload_resumes\(|def _process_resume_for_job\(")) {
            # Found next method, stop skipping
            $skipLines = $false
            $methodDepth = 0
            Write-Host "End of method found at line $($i + 1) - resuming"
            $newLines += $line
        }
        # Skip this line (part of the method we're removing)
        continue
    }
    
    # Keep this line
    $newLines += $line
}

# Write cleaned file
$newLines | Out-File -FilePath $outputPath -Encoding UTF8

Write-Host "Resume methods removed successfully!"
Write-Host "Original file had $($lines.Count) lines"
Write-Host "Cleaned file has $($newLines.Count) lines"
Write-Host "Removed $($lines.Count - $newLines.Count) lines"