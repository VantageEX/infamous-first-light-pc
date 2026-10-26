# SPDX-License-Identifier: GPL-2.0-or-later
"""What the QML interface (qml/Main.qml) sees of the launcher: the game, the settings and the runner."""
from pathlib import Path

from PySide6.QtCore import Property, QLocale, QObject, QUrl, Signal, Slot
from PySide6.QtGui import QTextCursor

import threading

from . import game, profiling, report, saves, symbols, trophies
from .i18n import STRINGS, UI_LANGUAGES, Texts
from .runner import GameRunner, open_folder, system_language_key
from .settings import (ASPECTS, CORE_DEFAULTS, DISPLAY_MODES, FPS_LIMITS, INI_SWITCHES, PORT, RENDER_RESOLUTIONS, TEST_BUILD, VERSION_FILE,
                       Settings, parse_env)

LOG_LINES = 5000
JOBS_PREFIX = 'Shader jobs: '
CORE_SWITCHES = ('vsync', 'fsr1', 'rcas', 'show_fps', 'async_shaders')


def _url(path):
    return QUrl.fromLocalFile(str(path)).toString() if path else ''


# What an exit code says: Windows exception codes (the process died of one) and the C runtime's.
EXIT_MEANINGS = {
    3: 'abort: a failed check or an unhandled C++ exception',
    0xC0000005: 'access violation (STATUS_ACCESS_VIOLATION)',
    0xC000001D: 'illegal instruction: the CPU lacks one the game uses',
    0xC00000FD: 'stack overflow',
    0xC0000374: 'heap corruption',
    0xC0000409: 'fail-fast: a security check or abort (STATUS_STACK_BUFFER_OVERRUN)',
    0xC000013A: 'closed with Ctrl+C or the console window',
    0x80000003: 'breakpoint: a failed assert',
    0xE06D7363: 'unhandled C++ exception',
}


def _screen_size():
    """'<width>x<height>' of the primary screen in pixels (the "auto" picture shape), '' unknown."""
    from PySide6.QtGui import QGuiApplication
    screen = QGuiApplication.primaryScreen()
    if not screen:
        return ''
    size, ratio = screen.size(), screen.devicePixelRatio()
    return f'{round(size.width() * ratio)}x{round(size.height() * ratio)}'


def _log_tail(path, size=4 << 20):
    """The end of a run log (a crash stack is in its last lines; a long session's log is large)."""
    with open(path, 'rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - size))
        return f.read().decode('utf-8', 'replace')


def exit_meaning(code):
    """'(...)' explaining a non-zero exit code, '' for 0 or an unknown one."""
    meaning = EXIT_MEANINGS.get(code & 0xFFFFFFFF) if code else None
    if not meaning and code and code & 0xFFFFFFFF >= 0x80000000:
        meaning = f'Windows exception 0x{code & 0xFFFFFFFF:08X}'
    return f'({meaning})' if meaning else ''


class Launcher(QObject):
    textsChanged = Signal()
    gameChanged = Signal()
    settingsChanged = Signal()
    runningChanged = Signal()
    statusChanged = Signal()
    savesChanged = Signal()          # the game folder, the language or a play session changed the saves
    notice = Signal(str, str, str)   # title, text, kind ('warning' or 'done'): a toast
    showLog = Signal()               # the game failed: open the log page
    profileChanged = Signal()        # the profiling page's choices or routes changed
    sessionPacked = Signal(str)      # a session's zip is ready (from the packing thread)

    def __init__(self, icon_font=('', False), parent=None):
        super().__init__(parent)
        self.settings = Settings()
        self.info = None
        self.game_error = ''
        self._status = ''
        self._log_doc = None
        self._log_buffer = []
        ui = self.settings.launcher['ui_language']
        self.t = Texts(QLocale.system().name()[:2] if ui == 'auto' else ui)
        # The advanced box starts on when it holds something, as the old window's checkable group did.
        self._advanced = bool(self.settings.launcher['extra_env'].strip())
        # app.py set the fonts up (fonts.setup) before this.
        self._icon_font, self._icon_fluent = icon_font
        self.runner = GameRunner(self)
        self.runner.line.connect(self._log_line)
        self.runner.finished.connect(self._game_finished)
        self._record_route = False   # profiling: the next start records a pad route (once)
        self._session = None         # profiling: the folder the running game records into
        self.sessionPacked.connect(self._session_packed)
        self._load_game(self.settings.launcher['game_dir'], quiet=True)

    # ---- texts --------------------------------------------------------------------------------
    @Property('QVariantMap', notify=textsChanged)
    def tr(self):
        return {key: self.t(key) for key in STRINGS}

    @Property(str, notify=textsChanged)
    def uiLanguage(self):
        return self.settings.launcher['ui_language']

    @Property('QVariantList', notify=textsChanged)
    def uiLanguages(self):
        return [{'value': 'auto', 'text': self.t('auto')}] + [
            {'value': key, 'text': name} for key, name in UI_LANGUAGES.items()]

    @Slot(str)
    def setUiLanguage(self, value):
        self.settings.launcher['ui_language'] = value
        self.settings.save()
        self.t = Texts(QLocale.system().name()[:2] if value == 'auto' else value)
        self.textsChanged.emit()
        self.gameChanged.emit()
        self.settingsChanged.emit()
        self.savesChanged.emit()

    @Property(str, constant=True)
    def iconFont(self):
        return self._icon_font

    @Property(bool, constant=True)
    def iconFluent(self):
        """The icon font is Fluent UI System Icons (theme.js iconFluent), not Segoe's."""
        return self._icon_fluent

    # ---- game folder --------------------------------------------------------------------------
    @Property(bool, notify=gameChanged)
    def hasGame(self):
        return self.info is not None

    @Property(str, notify=gameChanged)
    def gameFolder(self):
        return str(self.info.folder) if self.info else self.settings.launcher['game_dir']

    @Property(str, notify=gameChanged)
    def gameFolderUrl(self):
        folder = self.gameFolder
        return _url(folder) if folder and Path(folder).is_dir() else ''

    @Property(str, notify=gameChanged)
    def gameError(self):
        return self.game_error

    @Property(str, notify=gameChanged)
    def title(self):
        return (self.info.title if self.info else '') or 'inFAMOUS First Light'

    @Property(str, notify=gameChanged)
    def titleId(self):
        return self.info.title_id if self.info else ''

    @Property(str, notify=gameChanged)
    def versionText(self):
        return self.t('version_text', version=self.info.version or '?') if self.info else ''

    @Property(str, notify=gameChanged)
    def untested(self):
        if not self.info or self.info.known:
            return ''
        return self.t('game_untested', title_id=self.info.title_id or '?')

    @Property(str, notify=gameChanged)
    def iconUrl(self):
        return _url(self.info.icon) if self.info else ''

    @Property(str, notify=gameChanged)
    def backgroundUrl(self):
        return _url(self.info.background) if self.info else ''

    @Property('QVariantMap', notify=savesChanged)
    def lastSave(self):
        """The newest save of this game for the Continue card: mission, percent (-1 unknown),
        when it was saved, autosave or not, how many saves; empty when there is none."""
        found = saves.list_saves(self.settings.folder / 'user', self.info.title_id) if self.info else []
        if not found:
            return {}
        newest = found[0]
        line = [f"{newest['percent']}%"] if newest['percent'] is not None else []
        line.append(saves.when(newest['mtime'], self.t))
        if newest['auto']:
            line.append(self.t('save_auto'))
        return {'mission': newest['mission'],
                'percent': newest['percent'] if newest['percent'] is not None else -1,
                'detail': ' · '.join(line), 'count': self.t('save_count', count=len(found))}

    @Property('QVariantMap', notify=savesChanged)
    def trophies(self):
        """The trophies unlocked on this PC (trophies.py), newest first, for the trophy card:
        total in the game's pack, and each one's name, grade, icon (a URL) and date; empty without
        a game or without a trophy pack."""
        if not self.info:
            return {}
        data = self.settings.folder
        found = trophies.summary(self.info.folder, data / 'user', data / 'cache' / 'trophies', self.info.title_id)
        if not found['total']:
            return {}
        for row in found['unlocked']:
            row['icon'] = _url(row['icon'])
        return found

    @Slot(str)
    def chooseFolder(self, url):
        folder = QUrl(url).toLocalFile() if '://' in url or url.startswith('file:') else url
        if not folder:
            return
        if self.info:
            # A wrong pick does not throw away the game that works: its error comes as a notice.
            _, error = self._inspect(folder)
            if error:
                self.notice.emit(self.t('game_folder'), error, 'warning')
                return
        self._load_game(folder)

    def _inspect(self, folder):
        """(GameInfo, '') or (None, the error text)."""
        try:
            return game.inspect(folder), ''
        except ValueError as error:
            reason = str(error)
            return None, self.t(reason) if reason in ('no_eboot', 'no_sfo') else self.t('bad_sfo', error=reason)
        except OSError as error:
            return None, self.t('bad_sfo', error=error)

    def _load_game(self, folder, quiet=False):
        self.info, self.game_error = self._inspect(folder) if folder else (None, '')
        if self.info and not quiet:
            self.settings.launcher['game_dir'] = str(self.info.folder)
            self.settings.save()
        self.gameChanged.emit()
        self.settingsChanged.emit()
        self.savesChanged.emit()

    # ---- settings -----------------------------------------------------------------------------
    @Property('QVariantMap', notify=settingsChanged)
    def core(self):
        return dict(self.settings.core)

    @Slot(str, str)
    def setCore(self, key, value):
        if key not in CORE_DEFAULTS or not value:
            return
        self.settings.core[key] = value
        self.settings.save()
        self.settingsChanged.emit()

    @Property('QVariantList', notify=textsChanged)
    def renderers(self):
        """The renderers this build can start: renderer v2 / v3 only when their probe is there."""
        options = [{'value': '1', 'text': self.t('renderer_classic')}]
        if game.renderer_v2_probe(PORT):
            options.append({'value': '2', 'text': self.t('renderer_v2')})
        if game.renderer_v3_probe(PORT):
            options.append({'value': '3', 'text': self.t('renderer_v3')})
        return options

    @Property(str, notify=settingsChanged)
    def renderer(self):
        value = self.settings.launcher.get('renderer', '1')
        if value == '2' and game.renderer_v2_probe(PORT) or value == '3' and game.renderer_v3_probe(PORT):
            return value
        return '1'

    @Property(bool, notify=settingsChanged)
    def experimentalRenderer(self):
        """Renderer v3 is picked: the launcher shows its experimental notice."""
        return self.renderer == '3'

    @Slot(str)
    def setRenderer(self, value):
        if value in ('1', '2', '3') and value != self.settings.launcher.get('renderer'):
            self.settings.launcher['renderer'] = value
            self.settings.save()
            self.settingsChanged.emit()

    @Property('QVariantList', notify=textsChanged)
    def displayModes(self):
        return [{'value': mode, 'text': self.t(mode)} for mode in DISPLAY_MODES]

    @Property('QVariantList', notify=textsChanged)
    def fpsLimits(self):
        return [{'value': v, 'text': self.t('unlimited') if v == '0' else v} for v in FPS_LIMITS]

    @Property('QVariantList', notify=textsChanged)
    def renderResolutions(self):
        options = [{'value': v, 'text': self.t('res_' + v)} for v in RENDER_RESOLUTIONS]
        current = self.settings.core['render_resolution']
        if current not in RENDER_RESOLUTIONS:   # chosen elsewhere (tlg.ini): shown as it is
            options.append({'value': current, 'text': current})
        return options

    @Property('QVariantList', notify=textsChanged)
    def aspects(self):
        return [{'value': v, 'text': self.t('aspect_' + v.replace(':', '_'))} for v in ASPECTS]

    @Property('QVariantList', notify=gameChanged)
    def gameLanguages(self):
        keys = self.info.languages if self.info and self.info.languages else list(game.LANGUAGES)
        return [{'value': 'auto', 'text': self.t('language_auto')}] + [
            {'value': key, 'text': game.LANGUAGE_NAMES[key]} for key in keys]

    @Property(str, notify=settingsChanged)
    def gameLanguage(self):
        return self.settings.launcher['game_language']

    @Slot(str)
    def setGameLanguage(self, value):
        if value:
            self.settings.launcher['game_language'] = value
            self.settings.save()
            self.settingsChanged.emit()

    @Property(bool, notify=settingsChanged)
    def advanced(self):
        return self._advanced

    @Slot(bool)
    def setAdvanced(self, on):
        self._advanced = on
        self.settingsChanged.emit()

    # ---- profiling (profiling.py, port/docs/profiling.md) ------------------------------------
    @Property(bool, notify=profileChanged)
    def profileRecord(self):
        return self.settings.launcher['profile_record'] == '1'

    @Slot(bool)
    def setProfileRecord(self, on):
        self.settings.launcher['profile_record'] = '1' if on else '0'
        self.settings.save()
        self.profileChanged.emit()

    @Property(bool, notify=profileChanged)
    def profileRecordRoute(self):
        return self._record_route

    @Slot(bool)
    def setProfileRecordRoute(self, on):
        self._record_route = on
        self.profileChanged.emit()

    @Property('QVariantList', notify=profileChanged)
    def profileRoutes(self):
        return [{'value': '', 'text': self.t('profile_no_route')}] + [
            {'value': name, 'text': name} for name in profiling.list_routes(self.settings.folder)]

    @Property(str, notify=profileChanged)
    def profileRoute(self):
        return self.settings.launcher['profile_route']

    @Slot(str)
    def setProfileRoute(self, value):
        self.settings.launcher['profile_route'] = value
        self.settings.save()
        self.profileChanged.emit()

    @Property('QVariantList', notify=textsChanged)
    def profileDurations(self):
        return [{'value': v, 'text': self.t('profile_until_closed') if v == '0' else self.t('profile_seconds', n=v)}
                for v in profiling.SECONDS]

    @Property(str, notify=profileChanged)
    def profileSeconds(self):
        return self.settings.launcher['profile_seconds']

    @Slot(str)
    def setProfileSeconds(self, value):
        self.settings.launcher['profile_seconds'] = value
        self.settings.save()
        self.profileChanged.emit()

    @Property(bool, notify=settingsChanged)
    def profileBuild(self):
        """The profiling build of renderer v2 is here (the Tracy capture); without it a session
        still records the sensors, the timeline and the game's code."""
        return profiling.profile_build(game.renderer_v2_probe(PORT)) is not None

    @Slot()
    def openSessions(self):
        folder = profiling.sessions_dir(self.settings.folder)
        folder.mkdir(parents=True, exist_ok=True)
        open_folder(folder)

    def _profile_env(self):
        """The profiling page's variables for this start (the session folder is kept for packing)."""
        l = self.settings.launcher
        env, self._session = profiling.start_env(self.settings.folder, l['profile_record'] == '1', l['profile_route'],
                                                 l['profile_seconds'], self._record_route)
        if self._record_route:  # once: the next start plays normally again
            self._record_route = False
            self.profileChanged.emit()
        return env

    def _pack_session(self):
        session, self._session = self._session, None
        if session is None:
            return
        self._log_line(self.t('profile_packing'))
        log = self.runner.log_path

        def work():
            try:
                archive = profiling.pack(session, log)
                self.sessionPacked.emit(str(archive))
            except OSError as error:
                self.sessionPacked.emit('!' + str(error))
        threading.Thread(target=work, daemon=True).start()

    def _session_packed(self, result):
        if result.startswith('!'):
            self._log_line(self.t('profile_pack_failed', error=result[1:]))
            return
        text = self.t('profile_packed', path=result, size=profiling.size_text(result))
        self._log_line(text)
        self.notice.emit(self.t('profiling'), text, 'done')
        self.profileChanged.emit()  # a recorded route is a new choice

    @Property(str, constant=True)
    def extraEnv(self):
        """The advanced box as stored; QML sets it once and then only writes."""
        return self.settings.launcher['extra_env']

    @Slot(str)
    def setExtraEnv(self, text):
        if text != self.settings.launcher['extra_env']:
            self.settings.launcher['extra_env'] = text
            self.settings.save()

    def _extra_env(self):
        """The advanced variables (when the box is on); the ones run.py reads from tlg.ini go there.
        The game's own core switches come from run.py (port/scripts/game_defaults.py)."""
        extra = {}
        if self.info:
            extra.update(game.renderer_env(self.renderer, self.info.folder, PORT))
        if TEST_BUILD:
            extra.update(game.TEST_BUILD_ENV)
        if self._advanced:
            extra.update(parse_env(self.settings.launcher['extra_env']))
        for variable, key in INI_SWITCHES.items():
            value = extra.pop(variable, CORE_DEFAULTS[key])
            self.settings.core[key] = '0' if value == '0' else '1'
        return extra

    def _language_id(self):
        key = self.settings.launcher['game_language']
        available = self.info.languages if self.info else []
        if key == 'auto' or key not in game.LANGUAGES:
            key = system_language_key(QLocale.system().name(), available)
        return game.LANGUAGES[key]

    # ---- running ------------------------------------------------------------------------------
    @Property(bool, notify=runningChanged)
    def running(self):
        return self.runner.running

    @Property(str, notify=statusChanged)
    def status(self):
        return self._status

    def _set_status(self, text):
        self._status = text
        self.statusChanged.emit()

    @Slot()
    def play(self):
        if self.runner.running:
            self.runner.stop()
            return
        if not self.info:
            return
        extra = self._extra_env()
        extra.update(self._profile_env())
        self.settings.save()
        self._clear_log()
        self._log_line(self.t('starting'))
        if self.settings.core.get('aspect') == 'auto':
            extra['TLG_SCREEN_SIZE'] = _screen_size()
        for key, value in extra.items():
            self._log_line(f'{key}={value}')
        version = VERSION_FILE.read_text(encoding='utf-8').strip() if VERSION_FILE.is_file() else ''
        user_env = parse_env(self.settings.launcher['extra_env']) if self._advanced else {}
        header = report.session_header(version, self.info, self.settings.core, self.renderer, user_env)
        self.runner.start(self.info.folder, self.settings.folder, self._language_id(), extra,
                          ui_language=self.t.language, header=header)
        self._set_status(self.t('running') if self.runner.running else '')
        self.runningChanged.emit()

    def _game_finished(self, code):
        message = self.t('exited', code=code)
        meaning = exit_meaning(code)
        if meaning:
            message += f' {meaning}'
        self._log_line(message)
        self.runner.append_log(f'Launcher: {message}')
        if code:
            # The crash stack's frames with function names (symbols.py), in the log and the view.
            try:
                names = symbols.symbolize(_log_tail(self.runner.log_path), PORT / 'out')
            except (OSError, AttributeError, TypeError):
                names = ''
            if names:
                block = 'Crash stack with names (the nearest exported function):\n' + names
                self.runner.append_log(block)
                for line in block.splitlines():
                    self._log_line(line)
        self._set_status(message)
        self.runningChanged.emit()
        self.savesChanged.emit()
        self._pack_session()
        self.profileChanged.emit()  # a route recorded in this run
        if code:
            self.showLog.emit()

    @Slot()
    def openLogs(self):
        open_folder(self.settings.folder / 'logs')

    @Property(bool, constant=True)
    def testBuild(self):
        return TEST_BUILD

    @Slot()
    def createReport(self):
        """data/reports/report-*.zip for the developer; opens the folder and names the archive."""
        self.settings.save()
        try:
            path = report.create(self.settings.folder, self.settings.launcher, self.info, VERSION_FILE)
        except OSError as error:
            self.notice.emit(self.t('create_report'), self.t('report_failed', error=error), 'warning')
            return
        open_folder(path.parent)
        self.notice.emit(self.t('report_done'), path.name, 'done')

    @Slot()
    def openSaves(self):
        open_folder(self.settings.folder / 'user')

    # ---- log ----------------------------------------------------------------------------------
    @Slot(QObject)
    def attachLog(self, quick_document):
        """The log page's TextArea (its QQuickTextDocument): the lines go straight into its
        document, at most LOG_LINES. The text item may swap the document, so it is looked up each time."""
        self._log_doc = quick_document
        for row in self._log_buffer:
            self._append(row)
        self._log_buffer = []

    def _document(self):
        document = self._log_doc.textDocument()
        if document.maximumBlockCount() != LOG_LINES:
            document.setMaximumBlockCount(LOG_LINES)
        return document

    def _append(self, text):
        document = self._document()
        cursor = QTextCursor(document)
        cursor.movePosition(QTextCursor.End)
        if not document.isEmpty():
            cursor.insertBlock()
        cursor.insertText(text)

    def _log_line(self, text):
        # run.py's first-start job file (build_job_file): "Shader jobs: <phase> <n>%" for the index
        # phase then the jobs phase, shown as one progress in the status line, until its last line
        # ("... in <n> s" or "not built ...").
        if self.runner.running and text.startswith(JOBS_PREFIX):
            body = text[len(JOBS_PREFIX):]
            phase, _, percent = body.rpartition(' ')
            if phase in ('index', 'jobs') and percent.endswith('%') and percent[:-1].isdigit():
                done = int(percent[:-1]) // 2 + (50 if phase == 'jobs' else 0)
                self._set_status(self.t('preparing_shaders', percent=done))
            elif body.endswith(' s') or body.startswith('not built'):
                self._set_status(self.t('running'))
        if self._log_doc is None:
            self._log_buffer = (self._log_buffer + [text])[-LOG_LINES:]
        else:
            self._append(text)

    def _clear_log(self):
        self._log_buffer = []
        if self._log_doc is not None:
            self._document().clear()

    def save(self):
        self.settings.save()
