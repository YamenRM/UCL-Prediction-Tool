$project_root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $project_root

set-location "$project_root/src"
python data_scraping.py
if ($LASTEXITCODE -ne 0){
    Write-Error "Fberf Scraping ran into a problem" 
    "$(Get-Date)[error]: <error while scraping>" | Out-File -Append "$project_root/log.txt"
    exit 1
}

Write-Host "The Scrape is Complete"
"$(Get-Date)[ok]: <scrape complete>" | Out-File -Append "$project_root/log.txt"

git add $project_root/data/processed/matches_combined.csv
git add $project_root/data/processed/fixtures_to_predict.csv
$stageChanged = git status --porcelain
if (!($stageChanged | Select-String "data/processed/matches_combined.csv" -quiet) -and !($stageChanged | select-string "data/processed/fixtures_to_predict.csv" -quiet)){
    Write-Host "No new data found for the model. Exiting"
    "$(Get-Date)[ok]: <no new data found>" | Out-File -Append "$project_root/log.txt"
    exit 0
}
else {
    Write-Host "Refreshing the data for the model"
    git commit -m "Update processed files"
    git push
    if ($LASTEXITCODE -ne 0){
        Write-Error "Git push failed. Please check your credentials and try again."
        "$(Get-Date)[error]: <git push failed>" | Out-File -Append "$project_root/log.txt"
        exit 1
    }
    "$(Get-Date)[ok]: <git push successful>" | Out-File -Append "$project_root/log.txt"

}

