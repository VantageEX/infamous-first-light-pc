# SPDX-License-Identifier: GPL-2.0-or-later
"""Updates from the GitHub releases: one HTTPS GET to the releases API when the launcher starts (no
data of the player is sent). A packaged launcher in a folder it can write to downloads the new
package (this system's: the Windows .zip or the Linux .tar.gz), checks its SHA-256 against the one
GitHub publishes, unpacks it beside itself and quits; a helper (PowerShell on Windows, bash on
Linux) then swaps the package's files (the launcher, _internal/, port/, ...) for the new ones,
puts the old ones back if a move fails, and starts the new launcher. data/ is never touched.
Elsewhere (a source tree, Program Files) the launcher only points to the release page."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from PySide6.QtCore import Property, QObject, Signal, Slot

from .runner import desktop_env, open_external
from .settings import FROZEN, REPO, writable

LATEST_URL = 'https://api.github.com/repos/VantageEX/infamous-first-light-pc/releases/latest'
RELEASE_FILE = REPO / 'RELEASE'      # the package's tag (dev/package.sh, dev/package_linux.sh)
WINDOWS = os.name == 'nt'
EXE_NAME = 'inFAMOUS First Light' + ('.exe' if WINDOWS else '')
PROBE = 'port/out/tlg-probe' + ('.exe' if WINDOWS else '')
STAGE = REPO / '.update'             # the download and the unpacked package, beside the old one
BACKUP = REPO / '.update-backup'     # the replaced files while the helper runs
CHUNK = 1 << 20


class UpdateError(Exception):
    pass


def parse_version(tag):
    """(0, 3, 1) for 'v0.3.1'; None when the tag is not a release version."""
    parts = tag.strip().lstrip('vV').split('.')
    try:
        return tuple(int(part) for part in parts) if len(parts) == 3 else None
    except ValueError:
        return None


def installed_tag():
    try:
        return RELEASE_FILE.read_text(encoding='utf-8').strip()
    except OSError:
        return os.environ.get('FL_RELEASE', '')   # FL_RELEASE: tests and source trees


def is_package(name, windows=WINDOWS):
    """This system's package among a release's assets: the Windows .zip or the Linux .tar.gz,
    not a test build."""
    if '-test' in name:
        return False
    return name.endswith('.zip') if windows else name.endswith('.tar.gz') and '-linux' in name


def fetch_latest(opener=urllib.request.urlopen):
    """{'tag', 'page', 'asset', 'url', 'size', 'sha256'} of the latest release."""
    request = urllib.request.Request(LATEST_URL, headers={
        'Accept': 'application/vnd.github+json', 'User-Agent': 'inFAMOUS-First-Light-PC-launcher'})
    try:
        with opener(request, timeout=15) as response:
            release = json.load(response)
    except (OSError, ValueError) as error:
        raise UpdateError(f'GitHub: {error}') from error
    tag = str(release.get('tag_name', ''))
    package = next((a for a in release.get('assets', []) if is_package(str(a.get('name', '')))), None)
    digest = str((package or {}).get('digest') or '')
    return {'tag': tag, 'page': str(release.get('html_url', '')),
            'asset': package and package['name'], 'url': package and package['browser_download_url'],
            'size': package and int(package['size']),
            'sha256': digest[7:].lower() if digest.startswith('sha256:') else ''}


def download(release, dest, progress, opener=urllib.request.urlopen):
    """The package into `dest`, checked against the release's size and SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    done = 0
    request = urllib.request.Request(release['url'], headers={'User-Agent': 'inFAMOUS-First-Light-PC-launcher'})
    try:
        with opener(request, timeout=30) as response, open(dest, 'wb') as out:
            while chunk := response.read(CHUNK):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                progress(done, release['size'])
    except OSError as error:
        raise UpdateError(f'download: {error}') from error
    if done != release['size']:
        raise UpdateError(f'download: {done} of {release["size"]} bytes')
    if release['sha256'] and digest.hexdigest() != release['sha256']:
        raise UpdateError('download: the SHA-256 does not match the release')


def unpack(archive, folder, tag):
    """The package's folder inside `folder`, unpacked from `archive` and checked: its own exe,
    launcher files and core, its RELEASE the tag, nothing outside it and no data/ folder.
    The archive's names may use either slash (Compress-Archive writes backslashes). A .tar.gz
    (Linux: it keeps the programs executable) goes through tarfile's 'data' filter, which refuses
    absolute names, '..' and links that lead outside."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    root = folder.resolve()
    if str(archive).endswith('.tar.gz'):
        try:
            with tarfile.open(archive, 'r:gz') as package:
                package.extractall(root, filter='data')
        except (tarfile.TarError, OSError) as error:
            raise UpdateError(f'archive: {error}') from error
        return check_payload(folder, tag)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            parts = PurePosixPath(member.filename.replace('\\', '/')).parts
            if not parts or any(part in ('', '.', '..') or ':' in part for part in parts):
                raise UpdateError(f'archive: bad name {member.filename!r}')
            target = root.joinpath(*parts)
            if member.is_dir() or member.filename.endswith(('/', '\\')):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member) as source, open(target, 'wb') as out:
                shutil.copyfileobj(source, out, CHUNK)
    return check_payload(folder, tag)


def check_payload(folder, tag):
    tops = [entry for entry in folder.iterdir()]
    payload = tops[0] if len(tops) == 1 and tops[0].is_dir() else folder
    for needed in (EXE_NAME, '_internal', PROBE, 'RELEASE'):
        if not (payload / needed).exists():
            raise UpdateError(f'archive: no {needed}')
    if (payload / 'data').exists():
        raise UpdateError('archive: holds a data folder')
    if (payload / 'RELEASE').read_text(encoding='utf-8').strip() != tag:
        raise UpdateError('archive: its RELEASE is not the release tag')
    return payload


# The helper: waits for the launcher, refuses while anything runs from the folder (the game), moves
# each top-level item of the new package in (the old one into .update-backup first), puts the old
# items back when a move fails, and starts the launcher again either way. One line per outcome in
# the log, which the next launcher reads.
HELPER = r'''param([int]$WaitPid, [string]$Payload, [string]$Target, [string]$Backup, [string]$Exe, [string]$Log)
$ErrorActionPreference = 'Stop'
function Note($text) {
    # Never fails: a log that cannot be written must not stop the install or the restart.
    try {
        $dir = Split-Path -Parent $Log
        if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        Add-Content -LiteralPath $Log -Value ("{0} {1}" -f (Get-Date -Format s), $text) -Encoding UTF8
    } catch {}
}
Note "started"
try { Wait-Process -Id $WaitPid -Timeout 120 -ErrorAction SilentlyContinue } catch {}
$busy = $null
for ($i = 0; $i -lt 60; $i++) {
    $busy = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($Target + '\', 'OrdinalIgnoreCase') }
    if (-not $busy) { break }
    Start-Sleep -Seconds 1
}
if ($busy) {
    Note ("failed: still running from the folder: " + (($busy | ForEach-Object { $_.ProcessName }) -join ', '))
    Start-Process -FilePath (Join-Path $Target $Exe) -WorkingDirectory $Target
    exit 1
}
if (Test-Path -LiteralPath $Backup) { Remove-Item -LiteralPath $Backup -Recurse -Force }
New-Item -ItemType Directory -Path $Backup | Out-Null
$moved = @()
try {
    foreach ($name in (Get-ChildItem -LiteralPath $Payload -Force | ForEach-Object { $_.Name })) {
        $dest = Join-Path $Target $name
        if (Test-Path -LiteralPath $dest) { Move-Item -LiteralPath $dest -Destination (Join-Path $Backup $name) }
        $moved += $name
        Move-Item -LiteralPath (Join-Path $Payload $name) -Destination $dest
    }
    Note "installed"
} catch {
    Note ("failed: " + $_.Exception.Message)
    foreach ($name in $moved) {
        $dest = Join-Path $Target $name
        $old = Join-Path $Backup $name
        if (Test-Path -LiteralPath $old) {
            if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Recurse -Force }
            Move-Item -LiteralPath $old -Destination $dest
        }
    }
}
Start-Process -FilePath (Join-Path $Target $Exe) -WorkingDirectory $Target
'''

# The same steps in bash (Linux): "busy" is a process whose program lies in the folder.
HELPER_SH = r'''#!/bin/bash
wait_pid=$1 payload=$2 target=$3 backup=$4 exe=$5 log=$6
note() { { mkdir -p "$(dirname "$log")" && printf '%s %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$1" >> "$log"; } 2>/dev/null || true; }
restart() { (cd "$target" && exec "./$exe") </dev/null >/dev/null 2>&1 & }
note started
for _ in $(seq 1200); do kill -0 "$wait_pid" 2>/dev/null || break; sleep 0.1; done
busy=
for _ in $(seq 60); do
    busy=
    for link in /proc/[0-9]*/exe; do
        path=$(readlink "$link" 2>/dev/null) || continue
        [[ $path == "$target"/* ]] && busy="$busy $(basename "$path")"
    done
    [[ -z $busy ]] && break
    sleep 1
done
if [[ -n $busy ]]; then note "failed: still running from the folder:$busy"; restart; exit 1; fi
rm -rf -- "$backup" && mkdir -p -- "$backup" || { note "failed: cannot create $backup"; restart; exit 1; }
moved=()
failed=
shopt -s dotglob nullglob
for item in "$payload"/*; do
    name=$(basename "$item")
    if [[ -e $target/$name ]] && ! mv -- "$target/$name" "$backup/$name"; then failed="cannot move $name aside"; break; fi
    moved+=("$name")
    if ! mv -- "$item" "$target/$name"; then failed="cannot move $name in"; break; fi
done
if [[ -n $failed ]]; then
    note "failed: $failed"
    for name in "${moved[@]}"; do
        if [[ -e $backup/$name ]]; then rm -rf -- "${target:?}/$name"; mv -- "$backup/$name" "$target/$name"; fi
    done
else
    note "installed"
fi
restart
'''


CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def note(log, text):
    """One line in the update log, as the helpers write theirs."""
    import datetime
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, 'a', encoding='utf-8') as out:
            out.write(f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')} {text}\n")
    except OSError:
        pass


def start_helper(payload, log):
    """Starts the helper (it outlives this process) and returns."""
    STAGE.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    note(log, 'requested')   # the helper's own lines follow: started, then the outcome
    if not WINDOWS:
        script = STAGE / 'install.sh'
        script.write_text(HELPER_SH, encoding='utf-8')
        # The system's environment, not PyInstaller's; the new launcher starts as a fresh one.
        env = {**desktop_env(), 'PYINSTALLER_RESET_ENVIRONMENT': '1'}
        subprocess.Popen(['/bin/bash', str(script), str(os.getpid()), str(payload), str(REPO), str(BACKUP),
                          EXE_NAME, str(log)], cwd=str(REPO), env=env, start_new_session=True, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    script = STAGE / 'install.ps1'
    script.write_text(HELPER, encoding='utf-8-sig')   # with a BOM: PowerShell 5.1 reads it as UTF-8
    # A hidden console of its own, not DETACHED_PROCESS: with no console at all powershell.exe
    # ends before it runs the script (v0.3.2 / v0.3.3: no update ever applied, issue #2). Out of
    # the launcher's job when allowed, so closing the launcher cannot end it.
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    args = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script),
            '-WaitPid', str(os.getpid()), '-Payload', str(payload), '-Target', str(REPO),
            '-Backup', str(BACKUP), '-Exe', EXE_NAME, '-Log', str(log)]
    for extra in (CREATE_BREAKAWAY_FROM_JOB, 0):
        try:
            subprocess.Popen(args, cwd=str(REPO), creationflags=flags | extra, close_fds=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except PermissionError:
            if not extra:
                raise


def finish(log):
    """After a start: what the last install did ('installed', 'failed: ...' or '' when none ran
    since), with its leftovers removed."""
    outcome = ''
    if STAGE.exists() or BACKUP.exists():
        try:
            outcome = log.read_text(encoding='utf-8-sig').strip().splitlines()[-1].split(' ', 1)[1]
        except (OSError, IndexError):
            outcome = ''
        # The launcher's own 'requested' or the helper's 'started' last: it did not finish.
        if outcome == 'requested':
            outcome = 'failed: the update helper did not start'
        elif outcome == 'started':
            outcome = 'failed: the update helper stopped before it finished'
        for leftover in (STAGE, BACKUP):
            shutil.rmtree(leftover, ignore_errors=True)
    return outcome


class Updater(QObject):
    """What the launcher's update card sees: state is '' (nothing to show), 'available',
    'downloading', 'installing' or 'error'."""
    changed = Signal()
    notice = Signal(str, str, str)
    _result = Signal(str, object)     # from the worker threads, delivered in the GUI thread

    def __init__(self, texts, data_dir, running, parent=None):
        super().__init__(parent)
        self.t = texts          # a callable returning the current Texts (the language can change)
        self.log = Path(data_dir) / 'logs' / 'update.log'
        self.running = running  # a callable: is the game running
        self.release = None
        self._state = ''
        self._progress = 0
        self._error = ''
        self._result.connect(self._on_result)
        # The last install's outcome, told once the interface is up (check()).
        self._outcome = finish(self.log) if FROZEN else ''

    @Property(str, notify=changed)
    def state(self):
        return self._state

    @Property(int, notify=changed)
    def progress(self):
        return self._progress

    @Property(str, notify=changed)
    def version(self):
        return self.release['tag'] if self.release else ''

    @Property(str, notify=changed)
    def error(self):
        return self._error

    @Property(str, notify=changed)
    def title(self):
        t = self.t()
        return {'available': t('update_available', version=self.version),
                'downloading': t('update_downloading', percent=self._progress),
                'installing': t('update_installing'),
                'error': t('update_failed_title')}.get(self._state, '')

    @Property(str, notify=changed)
    def detail(self):
        if self._state == 'error':
            return self._error
        if self._state == 'available':
            return self.t()('update_hint' if self.canInstall else 'update_hint_manual')
        return ''

    @Property(bool, notify=changed)
    def canInstall(self):
        """In place only from a package that knows its release, in a folder it can write to."""
        return bool(FROZEN and installed_tag() and self.release and self.release['url'] and writable(REPO))

    def _set(self, state, progress=None, error=None):
        self._state = state
        if progress is not None:
            self._progress = progress
        if error is not None:
            self._error = error
        self.changed.emit()

    @Property(str, constant=True)
    def installed(self):
        """This package's release tag ('' for a source tree or a package before v0.3.2)."""
        return installed_tag()

    @Slot()
    def check(self):
        """Once the interface is up: the last install's outcome, then a newer release than this
        package's shows the card."""
        outcome, self._outcome = self._outcome, ''
        if outcome == 'installed':
            self.notice.emit(self.t()('update_done_title'), self.t()('update_done', version=installed_tag()), 'done')
        elif outcome.startswith('failed'):
            self.notice.emit(self.t()('update_failed_title'), outcome, 'warning')
        if parse_version(installed_tag()) is None or os.environ.get('FL_UPDATE_CHECK') == '0':
            return
        self._check(manual=False)

    @Slot()
    def checkNow(self):
        """The Log page's button: the same look, and its result told in any case (up to date,
        offline, a build that cannot update itself)."""
        if self._state in ('downloading', 'installing'):
            return
        self._check(manual=True)

    def _check(self, manual):
        mine = parse_version(installed_tag())

        def work():
            try:
                release = fetch_latest()
            except UpdateError as error:
                self._result.emit('check_failed_manual' if manual else 'check_failed', str(error))
                return
            # A build without a release tag (a source tree) cannot update itself: any release is
            # offered, as a page to download it from.
            newer = mine is None or (parse_version(release['tag']) or ()) > mine
            self._result.emit('checked' if newer or not manual else 'up_to_date', release if newer else None)
        threading.Thread(target=work, daemon=True).start()

    @Slot()
    def install(self):
        if self._state in ('downloading', 'installing') or not self.canInstall:
            self.openPage()
            return
        if self.running():
            self.notice.emit(self.t()('update_failed_title'), self.t()('update_close_game'), 'warning')
            return
        release = dict(self.release)
        self._set('downloading', 0, '')

        def work():
            try:
                archive = STAGE / release['asset']
                last = [-1]

                def progress(done, total):
                    percent = done * 100 // max(1, total)
                    if percent != last[0]:
                        last[0] = percent
                        self._result.emit('progress', percent)
                download(release, archive, progress)
                self._result.emit('unpacking', None)
                payload = unpack(archive, STAGE / 'package', release['tag'])
                archive.unlink(missing_ok=True)
                self._result.emit('ready', payload)
            except (UpdateError, OSError, zipfile.BadZipFile, tarfile.TarError) as error:
                shutil.rmtree(STAGE, ignore_errors=True)
                self._result.emit('failed', str(error))
        threading.Thread(target=work, daemon=True).start()

    @Slot()
    def openPage(self):
        if self.release and self.release['page']:
            open_external(self.release['page'])

    @Slot()
    def dismiss(self):
        self._set('')

    def _on_result(self, kind, value):
        if kind == 'checked':
            self.release = value
            self._set('available' if value else '')
        elif kind == 'progress':
            self._set('downloading', value)
        elif kind == 'unpacking':
            self._set('installing', 100)
        elif kind == 'ready':
            if self.running():
                shutil.rmtree(STAGE, ignore_errors=True)
                self._set('error', error=self.t()('update_close_game'))
                return
            try:
                start_helper(value, self.log)
            except OSError as error:
                shutil.rmtree(STAGE, ignore_errors=True)
                self._set('error', error=str(error))
                return
            from PySide6.QtGui import QGuiApplication
            QGuiApplication.quit()
        elif kind == 'failed':
            self._set('error', error=value)
        elif kind == 'up_to_date':
            self.notice.emit(self.t()('update_none_title'), self.t()('update_none', version=installed_tag()), 'done')
        elif kind == 'check_failed_manual':
            self.notice.emit(self.t()('update_check_failed'), value, 'warning')
        # 'check_failed': offline or rate-limited; the card stays hidden
