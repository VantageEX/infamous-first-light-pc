# SPDX-License-Identifier: GPL-2.0-or-later
"""Starts the core (port/run.py prepares the game image and runs tlg-probe) and keeps its log."""
import codecs
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from .game import LANGUAGES
from .settings import FROZEN, PORT

KEEP_LOGS = 20


def system_language_key(locale_name, available):
    """The LANGUAGES key for a locale such as "ru_RU" or "pt_BR", limited to `available` (the
    game's own languages, empty when unknown); English when nothing fits."""
    language, _, region = locale_name.replace('-', '_').partition('_')
    region = region.upper()
    candidates = {
        ('pt', 'BR'): ['pt_BR', 'pt_PT'], ('pt', 'PT'): ['pt_PT', 'pt_BR'],
        ('es', 'ES'): ['es', 'es_LA'], ('fr', 'CA'): ['fr_CA', 'fr'], ('en', 'GB'): ['en_GB', 'en'],
        ('zh', 'TW'): ['zh_TW', 'zh_CN'], ('zh', 'HK'): ['zh_TW', 'zh_CN'],
    }.get((language, region))
    if candidates is None:
        sibling = {'es': ['es_LA', 'es'], 'fr': ['fr', 'fr_CA'], 'pt': ['pt_BR', 'pt_PT'],
                   'zh': ['zh_CN', 'zh_TW'], 'nb': ['no'], 'nn': ['no']}
        candidates = sibling.get(language, [language])
    for key in candidates + ['en', 'en_GB']:
        if key in LANGUAGES and (not available or key in available):
            return key
    return 'en'


class GameRunner(QObject):
    line = Signal(str)
    started = Signal()
    finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.log_file = None

    @property
    def running(self):
        return self.process is not None and self.process.state() != QProcess.NotRunning

    def start(self, game_dir, data_dir, language_id, extra_env=None, ui_language='en'):
        if self.running:
            return
        data_dir = Path(data_dir)
        logs = data_dir / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        for old in sorted(logs.glob('run-*.log'))[:-KEEP_LOGS]:
            old.unlink(missing_ok=True)
        self.log_file = open(logs / time.strftime('run-%Y%m%d-%H%M%S.log'), 'w', encoding='utf-8')

        env = QProcessEnvironment.systemEnvironment()
        env.insert('TLG_GAME_DIR', str(game_dir))
        env.insert('TLG_DATA_DIR', str(data_dir))
        env.insert('TLG_USER_DIR', str(data_dir / 'user'))
        env.insert('TLG_LANGUAGE', str(language_id))
        # The core's menus and dialogs speak the launcher's language (ru or en).
        env.insert('TLG_UI_LANG', ui_language)
        env.insert('ISS_UI_LANG', ui_language)
        env.insert('PYTHONIOENCODING', 'utf-8')
        env.insert('PYTHONUNBUFFERED', '1')
        for key, value in (extra_env or {}).items():
            env.insert(key, value)

        self._decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self._partial = ''
        self.process = QProcess(self)
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(PORT))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        if FROZEN:
            # The packaged launcher runs Python scripts itself (__main__.py --script).
            self.process.start(sys.executable, ['--script', str(PORT / 'run.py')])
        else:
            # python.exe even when the launcher runs under pythonw.exe: run.py and its scripts
            # write the log to stdout, which pythonw would drop.
            interpreter = Path(sys.executable)
            if interpreter.name.lower() == 'pythonw.exe':
                interpreter = interpreter.with_name('python.exe')
            self.process.start(str(interpreter), [str(PORT / 'run.py')])
        self.started.emit()

    def stop(self):
        if self.running:
            # run.py waits on tlg-probe: killing run.py alone would leave the game running.
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(self.process.processId()), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                for pid in descendants(self.process.processId()):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
            self.process.kill()

    def _read(self, final=False):
        # Output arrives in arbitrary chunks: a UTF-8 character or a line may be cut between two
        # of them, so the decoder and the unfinished line carry over to the next chunk.
        text = self._decoder.decode(bytes(self.process.readAllStandardOutput()), final)
        if self.log_file:
            self.log_file.write(text)
            self.log_file.flush()
        rows = (self._partial + text).split('\n')
        self._partial = rows.pop()
        if final and self._partial:
            rows.append(self._partial)
            self._partial = ''
        for row in rows:
            self.line.emit(row.rstrip('\r'))

    def _finished(self, code, _status):
        self._read(final=True)
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self.finished.emit(code)

    def log_folder(self, data_dir):
        return Path(data_dir) / 'logs'


def descendants(pid):
    """Linux: every process below pid (from /proc), the deepest first."""
    children = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # The command name is in parentheses and may hold spaces: the fields follow the last ')'.
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
        except (OSError, IndexError):
            continue
        children.setdefault(int(fields[1]), []).append(int(entry.name))
    found, queue = [], [pid]
    while queue:
        for child in children.get(queue.pop(), []):
            found.append(child)
            queue.append(child)
    return found[::-1]


def desktop_env():
    """The environment for programs opened for the player (file manager, browser). The packaged
    Linux launcher runs with PyInstaller's LD_LIBRARY_PATH (its _internal folder): a program
    started with it loads the launcher's libraries instead of its own and quits."""
    env = dict(os.environ)
    if FROZEN:
        original = env.pop('LD_LIBRARY_PATH_ORIG', None)
        if original:
            env['LD_LIBRARY_PATH'] = original
        else:
            env.pop('LD_LIBRARY_PATH', None)
    return env


def open_external(target):
    """A folder or a URL in the system's own program."""
    if os.name == 'nt':
        os.startfile(str(target))
        return
    try:
        subprocess.Popen(['xdg-open', str(target)], env=desktop_env(), stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        pass


def open_folder(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    open_external(path)
