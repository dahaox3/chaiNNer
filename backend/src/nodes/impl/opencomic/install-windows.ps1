param([Parameter(Mandatory)][string]$Destination)
$ErrorActionPreference = 'Stop'
$target = [IO.Directory]::CreateDirectory($Destination).FullName
if (Get-ChildItem -LiteralPath $target -Force | Select-Object -First 1) {
    $manifest = Join-Path $target 'package.json'
    if (!(Test-Path -LiteralPath $manifest) -or (Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json).name -ne 'chainner-opencomic-runtime') {
        throw 'Choose an empty folder or the existing OpenComic runtime folder.'
    }
}
$nodePath = (Get-Command node.exe -ErrorAction Stop).Source
$nodeMajor = [int](& $nodePath -p 'process.versions.node.split(".")[0]')
if ($nodeMajor -lt 22) { throw 'Install Node.js 22 or newer to prepare the runtime.' }
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'package.json'),(Join-Path $PSScriptRoot 'package-lock.json') -Destination $target
Push-Location -LiteralPath $target
try {
    & npm.cmd ci --omit=dev --ignore-scripts --no-audit --no-fund --cache (Join-Path (Split-Path $target) 'opencomic-npm-cache')
    if ($LASTEXITCODE) { throw 'OpenComic runtime installation failed.' }
    & $nodePath (Join-Path $PSScriptRoot 'patch-daemon-tta.cjs') $target
    if ($LASTEXITCODE) { throw 'Failed to prepare OpenComic daemon TTA support.' }
    & $nodePath -e "require('sharp'); require('opencomic-ai-bin')"
    if ($LASTEXITCODE) { throw 'The installed dependencies require a newer Node.js runtime.' }
    Copy-Item -LiteralPath $nodePath -Destination (Join-Path $target 'node.exe')
    $version = & $nodePath -p 'process.version'
    Invoke-WebRequest "https://raw.githubusercontent.com/nodejs/node/$version/LICENSE" -OutFile (Join-Path $target 'NODE-LICENSE.txt')
    Invoke-WebRequest 'https://raw.githubusercontent.com/ollm/upscayl-ncnn/d4deb721ee31a313acbb810b1bbd4f437868da22/LICENSE' -OutFile (Join-Path $target 'UPSCAYL-LICENSE.txt')
} finally {
    Pop-Location
}
