param(
    [string]$ZBrushDirectory,
    [switch]$InstallPython
)
$ErrorActionPreference = 'Stop'
$version = '1.4.3'
$workFolder = 'C:\LQR'
$privatePython = Join-Path $workFolder 'python'
$pluginName = 'LocalQuadRetopo.txt'
$compiled2022 = 'LocalQuadRetopo_2022.zsc'
$compiled2025 = 'LocalQuadRetopo_2025.zsc'
$files = @('lqr_validate.py', 'lqr_run_pipeline.py', 'lqr_recover.py', 'lqr_local.py', 'lqr_mrgb.py',
           'recovery_geometry.py', 'patch_remesh.py', 'core_patch.py', 'LqrLaunch.exe')
# Python 3.12.10 is the last 3.12 release with Windows binaries.
$pythonUrls = @(
    'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip',
    'https://registry.npmmirror.com/-/binary/python/3.12.10/python-3.12.10-embed-amd64.zip'
)
$getPipUrls = @('https://bootstrap.pypa.io/get-pip.py')
# Empty entry = the default package index; the others are mirrors tried in turn.
$pipIndexes = @('', 'https://pypi.tuna.tsinghua.edu.cn/simple', 'https://mirrors.aliyun.com/pypi/simple/')
$pythonCheck = '-c "import numpy, scipy, sys; from scipy.optimize import milp; print(sys.executable)"'
$selfTest = '-c "import sys, lqr_run_pipeline, lqr_recover, lqr_local, lqr_mrgb, core_patch, patch_remesh, recovery_geometry, lqr_validate; sys.exit(lqr_validate.main([''--self-test'']))"'

function Say([string]$text, [string]$color = 'Gray') {
    Write-Host $text -ForegroundColor $color
}

# Runs a program without showing its output. Never throws.
function Invoke-Quiet([string]$exe, [string]$arguments) {
    try {
        $info = New-Object System.Diagnostics.ProcessStartInfo
        $info.FileName = $exe
        $info.Arguments = $arguments
        $info.WorkingDirectory = $workFolder
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $process = [System.Diagnostics.Process]::Start($info)
        $errors = $process.StandardError.ReadToEndAsync()
        $output = $process.StandardOutput.ReadToEnd()
        $process.WaitForExit()
        return @{ Code = $process.ExitCode; Output = $output.Trim(); Errors = $errors.Result.Trim() }
    } catch {
        return @{ Code = -1; Output = ''; Errors = $_.Exception.Message }
    }
}

# Returns the interpreter path if it can run the plugin, otherwise $null.
function Test-Python([string]$exe) {
    if (-not $exe) { return $null }
    $exe = $exe.Trim().Trim('"')
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { return $null }
    # The WindowsApps entry is only a shortcut to the Microsoft Store.
    if ($exe -like '*\WindowsApps\*') { return $null }
    $result = Invoke-Quiet $exe $pythonCheck
    if ($result.Code -ne 0 -or -not $result.Output) { return $null }
    return (($result.Output -split "`r?`n")[-1]).Trim()
}

function Find-Python {
    $candidates = New-Object System.Collections.Generic.List[string]
    $candidates.Add((Join-Path $privatePython 'python.exe'))
    $saved = Join-Path $workFolder 'lqr_python_path.txt'
    if (Test-Path -LiteralPath $saved -PathType Leaf) {
        $candidates.Add(([IO.File]::ReadAllText($saved)).Trim())
    }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $result = Invoke-Quiet $launcher.Source '-3 -c "import sys; print(sys.executable)"'
        if ($result.Code -eq 0 -and $result.Output) { $candidates.Add($result.Output) }
    }
    foreach ($command in @(Get-Command python.exe -All -ErrorAction SilentlyContinue)) {
        $candidates.Add($command.Source)
    }
    foreach ($candidate in $candidates) {
        $found = Test-Python $candidate
        if ($found) { return $found }
    }
    return $null
}

function Get-Download([string[]]$urls, [string]$destination, [string]$label) {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    foreach ($url in $urls) {
        if (Test-Path -LiteralPath $destination) { Remove-Item -LiteralPath $destination -Force }
        Say "      $url"
        $ok = $false
        try {
            Start-BitsTransfer -Source $url -Destination $destination -DisplayName $label -Description 'Downloading ...' -ErrorAction Stop
            $ok = $true
        } catch {
            try {
                (New-Object System.Net.WebClient).DownloadFile($url, $destination)
                $ok = $true
            } catch {
                Say "      This address failed; trying the next one. ($($_.Exception.Message))" 'Yellow'
            }
        }
        if ($ok -and (Test-Path -LiteralPath $destination) -and (Get-Item -LiteralPath $destination).Length -gt 0) {
            return
        }
    }
    throw "$label could not be downloaded. Check the network connection and run INSTALL.bat again."
}

# Installs a private Python with NumPy and SciPy into C:\LQR\python.
# Sets $script:pythonPath. Output of pip goes straight to the window.
function Install-PrivatePython {
    Say ''
    Say 'No usable Python was found. Installing one now.' 'Cyan'
    Say "  Location: $privatePython" 'Cyan'
    Say '  It is used by this plug-in only; any Python already on this computer is left alone.' 'Cyan'
    Say '  About 60 MB is downloaded, on the first installation only.' 'Cyan'
    Say ''
    $temp = Join-Path $env:TEMP ('lqr_setup_' + [Guid]::NewGuid().ToString('N'))
    New-Item -Path $temp -ItemType Directory | Out-Null
    # pip writes warnings to its error stream; Windows PowerShell 5.1 can turn
    # those into fatal errors under 'Stop'. Exit codes are checked instead.
    $previous = $ErrorActionPreference
    try {
        Say '[1/3] Downloading Python (about 11 MB) ...' 'White'
        $zip = Join-Path $temp 'python.zip'
        Get-Download $pythonUrls $zip 'Python 3.12'
        if (Test-Path -LiteralPath $privatePython) {
            Remove-Item -LiteralPath $privatePython -Recurse -Force
        }
        Expand-Archive -LiteralPath $zip -DestinationPath $privatePython
        $exe = Join-Path $privatePython 'python.exe'
        $signature = Get-AuthenticodeSignature -FilePath $exe
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notlike '*Python Software Foundation*') {
            Remove-Item -LiteralPath $privatePython -Recurse -Force
            throw 'The downloaded Python failed its digital signature check and was deleted. Run INSTALL.bat again.'
        }
        # Turn on site-packages, and let the plugin's own modules be found.
        $pth = Get-ChildItem -LiteralPath $privatePython -Filter '*._pth' | Select-Object -First 1
        $lines = @([IO.File]::ReadAllLines($pth.FullName) | ForEach-Object {
            if ($_.Trim() -eq '#import site') { 'import site' } else { $_ }
        })
        $lines += $workFolder
        [IO.File]::WriteAllLines($pth.FullName, $lines)

        Say '[2/3] Installing pip ...' 'White'
        $getPip = Join-Path $temp 'get-pip.py'
        Get-Download $getPipUrls $getPip 'pip'
        $ErrorActionPreference = 'Continue'
        & $exe $getPip --no-warn-script-location --disable-pip-version-check
        $ErrorActionPreference = $previous
        if ($LASTEXITCODE -ne 0) {
            throw 'pip could not be installed. Check the network connection and run INSTALL.bat again.'
        }

        Say '[3/3] Downloading NumPy and SciPy (about 50 MB), please wait ...' 'White'
        $done = $false
        foreach ($index in $pipIndexes) {
            $arguments = @('-m', 'pip', 'install', '--no-warn-script-location', '--disable-pip-version-check',
                           'numpy', 'scipy>=1.9')
            if ($index) {
                Say "      Retrying with mirror: $index" 'Yellow'
                $arguments += @('-i', $index)
            }
            $ErrorActionPreference = 'Continue'
            & $exe @arguments
            $ErrorActionPreference = $previous
            if ($LASTEXITCODE -eq 0) {
                $done = $true
                break
            }
        }
        if (-not $done) {
            throw 'NumPy / SciPy could not be downloaded. Check the network connection and run INSTALL.bat again.'
        }
    } finally {
        $ErrorActionPreference = $previous
        Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue
    }
    $script:pythonPath = Test-Python (Join-Path $privatePython 'python.exe')
    if (-not $script:pythonPath) {
        throw 'The Python that was just installed does not run. Please send a screenshot of this window to the plug-in author.'
    }
    Say 'Python is installed.' 'Green'
}

function Find-ZBrush {
    $places = New-Object System.Collections.Generic.List[string]
    $saved = Join-Path $workFolder 'lqr_zbrush_dirs.txt'
    if (Test-Path -LiteralPath $saved -PathType Leaf) {
        foreach ($line in [IO.File]::ReadAllLines($saved)) { $places.Add($line) }
    }
    $keys = @('HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
              'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*',
              'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*')
    foreach ($key in $keys) {
        foreach ($entry in @(Get-ItemProperty -Path $key -ErrorAction SilentlyContinue)) {
            if ($entry.DisplayName -like '*ZBrush*' -and $entry.InstallLocation) {
                $places.Add([string]$entry.InstallLocation)
            }
        }
    }
    foreach ($base in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if (-not $base -or -not (Test-Path -LiteralPath $base)) { continue }
        $folders = @(Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue |
                     Where-Object { $_.Name -like '*ZBrush*' -or $_.Name -like 'Pixologic*' -or $_.Name -like 'Maxon*' })
        foreach ($folder in $folders) {
            $places.Add($folder.FullName)
            foreach ($child in @(Get-ChildItem -LiteralPath $folder.FullName -Directory -ErrorAction SilentlyContinue |
                                 Where-Object { $_.Name -like '*ZBrush*' })) {
                $places.Add($child.FullName)
            }
        }
    }
    $result = @()
    foreach ($place in $places) {
        $dir = ([string]$place).Trim().Trim('"').TrimEnd('\')
        if ($dir -and (Test-Path -LiteralPath (Join-Path $dir 'ZBrush.exe') -PathType Leaf)) {
            $full = (Resolve-Path -LiteralPath $dir).Path
            if ($result -notcontains $full) { $result += $full }
        }
    }
    return $result
}

function Get-ZBrushYear([string]$dir) {
    $info = (Get-Item -LiteralPath (Join-Path $dir 'ZBrush.exe')).VersionInfo
    foreach ($text in @($info.ProductVersion, $info.FileVersion, $info.ProductName, $dir)) {
        if ($text -and ([string]$text -match '(20[1-9][0-9])')) { return [int]$Matches[1] }
    }
    return 0
}

function Select-ZBrush {
    if ($ZBrushDirectory) {
        $choice = $ZBrushDirectory
    } else {
        # Always ask: a ZBrush outside Program Files may not be found, and
        # installing silently into the only one found could be the wrong one.
        $found = @(Find-ZBrush)
        if ($found.Count -gt 0) {
            Say 'These ZBrush installations were found. Choose one to install into:' 'White'
            for ($i = 0; $i -lt $found.Count; $i++) {
                Say ('  {0}. ZBrush {1}    {2}' -f ($i + 1), (Get-ZBrushYear $found[$i]), $found[$i]) 'White'
            }
            Say '  0. Not listed; type the folder path' 'White'
            $answer = ([string](Read-Host 'Type a number and press Enter (Enter alone picks 1)')).Trim()
            if (-not $answer) { $answer = '1' }
            if ($answer -match '^\d+$' -and [int]$answer -ge 1 -and [int]$answer -le $found.Count) {
                $choice = $found[[int]$answer - 1]
            } elseif ($answer -eq '0') {
                $choice = Read-Host 'Type the folder that contains ZBrush.exe and press Enter'
            } else {
                $choice = $answer
            }
        } else {
            Say 'No ZBrush installation was found automatically.' 'Yellow'
            $choice = Read-Host 'Type the folder that contains ZBrush.exe and press Enter'
        }
    }
    $choice = ([string]$choice).Trim().Trim('"').TrimEnd('\')
    if ($choice -like '*\ZBrush.exe') { $choice = Split-Path -Parent $choice }
    if (-not $choice -or -not (Test-Path -LiteralPath (Join-Path $choice 'ZBrush.exe') -PathType Leaf)) {
        throw "ZBrush.exe was not found in $choice."
    }
    return (Resolve-Path -LiteralPath $choice).Path
}

function Copy-Checked([string]$source, [string]$destination, [string]$backupFolder) {
    if (Test-Path -LiteralPath $destination) {
        if ((Get-FileHash -LiteralPath $source).Hash -eq (Get-FileHash -LiteralPath $destination).Hash) {
            return
        }
        Copy-Item -LiteralPath $destination -Destination $backupFolder
    }
    Copy-Item -LiteralPath $source -Destination $destination
    if ((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $destination).Hash) {
        throw "The copied file did not match the original: $destination"
    }
    # Files from a downloaded zip carry an internet mark; without removing it
    # Windows may stop LqrLaunch.exe with a security prompt.
    Unblock-File -LiteralPath $destination -ErrorAction SilentlyContinue
}

try {
    Say "Local Quad Retopo $version installer" 'Cyan'
    Say ''
    foreach ($name in @($pluginName) + $files) {
        if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $name) -PathType Leaf)) {
            throw "The package is incomplete; $name is missing. Extract the whole zip file again."
        }
    }
    New-Item -Path $workFolder -ItemType Directory -Force | Out-Null

    # ---- ZBrush ------------------------------------------------------------
    $zbrush = Select-ZBrush
    $exe = Join-Path $zbrush 'ZBrush.exe'
    $year = Get-ZBrushYear $zbrush
    if ($year -eq 0) {
        Say 'The version of this ZBrush could not be identified; treating it as ZBrush 2025.' 'Yellow'
        $year = 2025
    }
    Say "Installing into: ZBrush $year    $zbrush" 'White'
    if ($year -lt 2022) {
        Say "This plug-in is tested on ZBrush 2022 and 2025 only; it may not work in ZBrush $year." 'Yellow'
        $go = [string](Read-Host 'Install anyway? Type Y to continue, anything else to cancel')
        if ($go.Trim() -notmatch '^[Yy]') { throw 'Installation cancelled.' }
    }
    if (Get-Process -Name ZBrush -ErrorAction SilentlyContinue | Where-Object { -not $_.Path -or $_.Path -eq $exe }) {
        throw 'ZBrush is running. Save your work, quit ZBrush, then run INSTALL.bat again.'
    }

    # ---- Python ------------------------------------------------------------
    Say ''
    Say 'Checking for Python ...' 'White'
    $script:pythonPath = $null
    if (-not $InstallPython) { $script:pythonPath = Find-Python }
    if ($script:pythonPath) {
        Say "Using the Python already installed: $script:pythonPath" 'Green'
    } else {
        Install-PrivatePython
    }
    $python = $script:pythonPath

    # ---- Files -------------------------------------------------------------
    Say ''
    Say 'Copying the plug-in files ...' 'White'
    $pluginFolder = Join-Path $zbrush 'ZStartup\ZPlugs64'
    New-Item -Path $pluginFolder -ItemType Directory -Force | Out-Null
    $backupFolder = Join-Path $workFolder ('backup_143_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
    New-Item -Path $backupFolder -ItemType Directory | Out-Null
    foreach ($name in $files) {
        Copy-Checked (Join-Path $PSScriptRoot $name) (Join-Path $workFolder $name) $backupFolder
    }
    $pythonConfig = Join-Path $workFolder 'lqr_python_path.txt'
    if (Test-Path -LiteralPath $pythonConfig -PathType Leaf) {
        Copy-Item -LiteralPath $pythonConfig -Destination $backupFolder
    }
    [IO.File]::WriteAllText($pythonConfig, $python)

    # The panel script declares the interface version it was written for.
    # ZBrush 2022.0.5 ships its own scripts with [IConfig,2021] and
    # [IConfig,4.8]; 2021 is the newest value known to work before 2025.
    if ($year -ge 2025) { $config = 2025 } else { $config = [Math]::Min($year, 2021) }
    $pluginText = [IO.File]::ReadAllText((Join-Path $PSScriptRoot $pluginName))
    if ($config -ne 2025) {
        if (-not $pluginText.Contains('[IConfig, 2025]')) { throw 'The plug-in file is damaged. Extract the whole zip file again.' }
        $pluginText = $pluginText.Replace('[IConfig, 2025]', "[IConfig, $config]")
    }
    $destination = Join-Path $pluginFolder $pluginName
    $legacySource = Join-Path $pluginFolder 'LocalQuadRetopo.v113.bak.txt'
    if (Test-Path -LiteralPath $legacySource -PathType Leaf) {
        Move-Item -LiteralPath $legacySource -Destination $backupFolder
    }
    if (Test-Path -LiteralPath $destination) {
        Copy-Item -LiteralPath $destination -Destination $backupFolder
    }
    [IO.File]::WriteAllText($destination, $pluginText, [Text.Encoding]::ASCII)
    (Get-Item -LiteralPath $destination).LastWriteTimeUtc = [DateTime]::UtcNow
    if ([IO.File]::ReadAllText($destination) -ne $pluginText) {
        throw "The copied file did not match the original: $destination"
    }

    # ZBrush only loads compiled .zsc files from ZPlugs64 at startup; a .txt
    # there is ignored unless a .zsc of the same name already exists. So each
    # version gets a copy compiled by that version, dated after the .txt so
    # ZBrush does not try to recompile it.
    if ($year -ge 2025) { $compiledSource = $compiled2025 } else { $compiledSource = $compiled2022 }
    $compiledDestination = Join-Path $pluginFolder 'LocalQuadRetopo.zsc'
    if (Test-Path -LiteralPath $compiledDestination) {
        Copy-Item -LiteralPath $compiledDestination -Destination $backupFolder
        Remove-Item -LiteralPath $compiledDestination -Force
    }
    $script:manualLoad = $null
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot $compiledSource) -PathType Leaf) {
        Copy-Checked (Join-Path $PSScriptRoot $compiledSource) $compiledDestination $backupFolder
        (Get-Item -LiteralPath $compiledDestination).LastWriteTimeUtc = [DateTime]::UtcNow.AddSeconds(5)
    } else {
        $script:manualLoad = $destination
    }

    $savedDirs = Join-Path $workFolder 'lqr_zbrush_dirs.txt'
    $known = @()
    if (Test-Path -LiteralPath $savedDirs -PathType Leaf) { $known = @([IO.File]::ReadAllLines($savedDirs)) }
    if ($known -notcontains $zbrush) { [IO.File]::WriteAllLines($savedDirs, @($known + $zbrush)) }

    # ---- Check -------------------------------------------------------------
    Say ''
    Say 'Checking the installation ...' 'White'
    $test = Invoke-Quiet $python $selfTest
    if ($test.Code -ne 0) {
        Say $test.Output
        Say $test.Errors
        throw 'The self-test after installation failed. Please send a screenshot of this window to the plug-in author.'
    }

    Say ''
    Say "Installed: Local Quad Retopo $version" 'Green'
    Say "  ZBrush: ZBrush $year    $zbrush" 'Green'
    Say "  Python: $python" 'Green'
    Say "  Backup of old files: $backupFolder" 'Green'
    Say ''
    Say 'Start ZBrush and open Local Quad Retopo in the ZPlugin menu.' 'White'
    if ($script:manualLoad) {
        Say ''
        Say "This package has no compiled panel for ZBrush $year, so load it once by hand:" 'Yellow'
        Say '  Start ZBrush, choose Zscript > Load and pick the file below. It loads by itself after that.' 'Yellow'
        Say "    $script:manualLoad" 'Yellow'
    }
} catch {
    Say ''
    Say "Installation did not finish: $($_.Exception.Message)" 'Red'
    exit 1
}
