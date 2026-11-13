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
$files = @('lqr_validate.py', 'lqr_run_pipeline.py', 'lqr_recover.py', 'recovery_geometry.py',
           'patch_remesh.py', 'core_patch.py', 'LqrLaunch.exe')
# Python 3.12.10 is the last 3.12 release with Windows binaries.
$pythonUrls = @(
    'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip',
    'https://registry.npmmirror.com/-/binary/python/3.12.10/python-3.12.10-embed-amd64.zip'
)
$getPipUrls = @('https://bootstrap.pypa.io/get-pip.py')
# Empty entry = the default package index; the others are mirrors tried in turn.
$pipIndexes = @('', 'https://pypi.tuna.tsinghua.edu.cn/simple', 'https://mirrors.aliyun.com/pypi/simple/')
$pythonCheck = '-c "import numpy, scipy, sys; from scipy.optimize import milp; print(sys.executable)"'
$selfTest = '-c "import sys, lqr_run_pipeline, lqr_recover, core_patch, patch_remesh, recovery_geometry, lqr_validate; sys.exit(lqr_validate.main([''--self-test'']))"'

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
            Start-BitsTransfer -Source $url -Destination $destination -DisplayName $label -Description '下载中 ...' -ErrorAction Stop
            $ok = $true
        } catch {
            try {
                (New-Object System.Net.WebClient).DownloadFile($url, $destination)
                $ok = $true
            } catch {
                Say "      这个地址下载失败，换下一个地址。($($_.Exception.Message))" 'Yellow'
            }
        }
        if ($ok -and (Test-Path -LiteralPath $destination) -and (Get-Item -LiteralPath $destination).Length -gt 0) {
            return
        }
    }
    throw "$label 下载失败。请检查网络连接后重新运行 INSTALL.bat。"
}

# Installs a private Python with NumPy and SciPy into C:\LQR\python.
# Sets $script:pythonPath. Output of pip goes straight to the window.
function Install-PrivatePython {
    Say ''
    Say '没有找到可用的 Python 运行环境，现在自动安装。' 'Cyan'
    Say "  安装位置：$privatePython" 'Cyan'
    Say '  只给本插件使用，不会改动电脑上已有的 Python。' 'Cyan'
    Say '  需要联网下载约 60 MB，只在第一次安装时进行。' 'Cyan'
    Say ''
    $temp = Join-Path $env:TEMP ('lqr_setup_' + [Guid]::NewGuid().ToString('N'))
    New-Item -Path $temp -ItemType Directory | Out-Null
    # pip writes warnings to its error stream; Windows PowerShell 5.1 can turn
    # those into fatal errors under 'Stop'. Exit codes are checked instead.
    $previous = $ErrorActionPreference
    try {
        Say '[1/3] Python 下载中（约 11 MB）...' 'White'
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
            throw '下载的 Python 没有通过数字签名校验，已删除。请重新运行 INSTALL.bat。'
        }
        # Turn on site-packages, and let the plugin's own modules be found.
        $pth = Get-ChildItem -LiteralPath $privatePython -Filter '*._pth' | Select-Object -First 1
        $lines = @([IO.File]::ReadAllLines($pth.FullName) | ForEach-Object {
            if ($_.Trim() -eq '#import site') { 'import site' } else { $_ }
        })
        $lines += $workFolder
        [IO.File]::WriteAllLines($pth.FullName, $lines)

        Say '[2/3] pip 安装中 ...' 'White'
        $getPip = Join-Path $temp 'get-pip.py'
        Get-Download $getPipUrls $getPip 'pip'
        $ErrorActionPreference = 'Continue'
        & $exe $getPip --no-warn-script-location --disable-pip-version-check
        $ErrorActionPreference = $previous
        if ($LASTEXITCODE -ne 0) {
            throw 'pip 安装失败。请检查网络连接后重新运行 INSTALL.bat。'
        }

        Say '[3/3] NumPy 和 SciPy 下载中（约 50 MB），请耐心等待 ...' 'White'
        $done = $false
        foreach ($index in $pipIndexes) {
            $arguments = @('-m', 'pip', 'install', '--no-warn-script-location', '--disable-pip-version-check',
                           'numpy', 'scipy>=1.9')
            if ($index) {
                Say "      换用镜像重试：$index" 'Yellow'
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
            throw 'NumPy / SciPy 下载失败。请检查网络连接后重新运行 INSTALL.bat。'
        }
    } finally {
        $ErrorActionPreference = $previous
        Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue
    }
    $script:pythonPath = Test-Python (Join-Path $privatePython 'python.exe')
    if (-not $script:pythonPath) {
        throw '自动安装的 Python 无法正常运行。请把这个窗口截图发给插件作者。'
    }
    Say 'Python 运行环境安装完成。' 'Green'
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
            Say '找到以下 ZBrush，请选择要安装到哪一个：' 'White'
            for ($i = 0; $i -lt $found.Count; $i++) {
                Say ('  {0}. ZBrush {1}    {2}' -f ($i + 1), (Get-ZBrushYear $found[$i]), $found[$i]) 'White'
            }
            Say '  0. 不在上面，手动输入路径' 'White'
            $answer = ([string](Read-Host '输入编号后回车（直接回车选 1）')).Trim()
            if (-not $answer) { $answer = '1' }
            if ($answer -match '^\d+$' -and [int]$answer -ge 1 -and [int]$answer -le $found.Count) {
                $choice = $found[[int]$answer - 1]
            } elseif ($answer -eq '0') {
                $choice = Read-Host '请输入 ZBrush.exe 所在的文件夹路径后回车'
            } else {
                $choice = $answer
            }
        } else {
            Say '没有自动找到 ZBrush。' 'Yellow'
            $choice = Read-Host '请输入 ZBrush.exe 所在的文件夹路径后回车'
        }
    }
    $choice = ([string]$choice).Trim().Trim('"').TrimEnd('\')
    if ($choice -like '*\ZBrush.exe') { $choice = Split-Path -Parent $choice }
    if (-not $choice -or -not (Test-Path -LiteralPath (Join-Path $choice 'ZBrush.exe') -PathType Leaf)) {
        throw "在 $choice 里没有找到 ZBrush.exe。"
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
        throw "文件复制校验失败：$destination"
    }
    # Files from a downloaded zip carry an internet mark; without removing it
    # Windows may stop LqrLaunch.exe with a security prompt.
    Unblock-File -LiteralPath $destination -ErrorAction SilentlyContinue
}

try {
    Say "局部四边重拓扑 $version 安装程序" 'Cyan'
    Say ''
    foreach ($name in @($pluginName) + $files) {
        if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $name) -PathType Leaf)) {
            throw "安装包不完整，缺少文件：$name。请重新解压整个压缩包。"
        }
    }
    New-Item -Path $workFolder -ItemType Directory -Force | Out-Null

    # ---- ZBrush ------------------------------------------------------------
    $zbrush = Select-ZBrush
    $exe = Join-Path $zbrush 'ZBrush.exe'
    $year = Get-ZBrushYear $zbrush
    if ($year -eq 0) {
        Say '无法识别这个 ZBrush 的版本，按 ZBrush 2025 处理。' 'Yellow'
        $year = 2025
    }
    Say "安装到：ZBrush $year    $zbrush" 'White'
    if ($year -lt 2022) {
        Say "本插件只在 ZBrush 2022 和 2025 上测试过，ZBrush $year 可能无法使用。" 'Yellow'
        $go = [string](Read-Host '仍要继续安装吗？输入 Y 继续，其他键取消')
        if ($go.Trim() -notmatch '^[Yy]') { throw '已取消安装。' }
    }
    if (Get-Process -Name ZBrush -ErrorAction SilentlyContinue | Where-Object { -not $_.Path -or $_.Path -eq $exe }) {
        throw 'ZBrush 正在运行。请先保存工程并正常退出 ZBrush，再重新运行 INSTALL.bat。'
    }

    # ---- Python ------------------------------------------------------------
    Say ''
    Say '正在检查 Python 运行环境 ...' 'White'
    $script:pythonPath = $null
    if (-not $InstallPython) { $script:pythonPath = Find-Python }
    if ($script:pythonPath) {
        Say "使用已有的 Python：$script:pythonPath" 'Green'
    } else {
        Install-PrivatePython
    }
    $python = $script:pythonPath

    # ---- Files -------------------------------------------------------------
    Say ''
    Say '正在复制插件文件 ...' 'White'
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
        if (-not $pluginText.Contains('[IConfig, 2025]')) { throw '插件文件内容异常，请重新解压整个压缩包。' }
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
        throw "文件复制校验失败：$destination"
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
    Say '正在检查安装结果 ...' 'White'
    $test = Invoke-Quiet $python $selfTest
    if ($test.Code -ne 0) {
        Say $test.Output
        Say $test.Errors
        throw '安装后自检没有通过。请把这个窗口截图发给插件作者。'
    }

    Say ''
    Say "安装完成：局部四边重拓扑 $version" 'Green'
    Say "  ZBrush ：ZBrush $year    $zbrush" 'Green'
    Say "  Python ：$python" 'Green'
    Say "  旧文件备份：$backupFolder" 'Green'
    Say ''
    Say '启动 ZBrush，在 Z插件 菜单里找到「局部四边重拓扑」即可使用。' 'White'
    if ($script:manualLoad) {
        Say ''
        Say "这个安装包没有附带 ZBrush $year 用的已编译插件文件，需要手动加载一次：" 'Yellow'
        Say '  启动 ZBrush，点 Z脚本 > 加载，选择下面这个文件。以后启动会自动加载。' 'Yellow'
        Say "    $script:manualLoad" 'Yellow'
    }
} catch {
    Say ''
    Say "安装没有完成：$($_.Exception.Message)" 'Red'
    exit 1
}
