$envFile = "c:\Workspaces\SKN22-4th-2Team\.env"
Get-Content $envFile | ForEach-Object {
  $line = $_.Trim()
  if (-not $line -or $line.StartsWith('#') -or -not $line.Contains('=')) { continue }
  $parts = $line.Split('=',2)
  $k = $parts[0].Trim()
  $v = $parts[1].Trim().Trim('"')
  Set-Item -Path "Env:$k" -Value $v
}
Set-Location "c:\Workspaces\SKN22-4th-2Team"
& "C:\Users\Playdata\miniconda3\python.exe" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8010
