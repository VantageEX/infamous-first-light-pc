# SPDX-License-Identifier: GPL-2.0-or-later
"""What the QML interface (qml/Main.qml) sees of the launcher: the game, the settings and the runner."""
from pathlib import Path

from PySide6.QtCore import Property, QLocale, QObject, QUrl, Signal, Slot
from PySide6.QtGui import QFontDatabase, QGuiApplication, QIcon, QTextCursor

from . import game, report
from .i18n import STRINGS, UI_LANGUAGES, Texts
from .runner import GameRunner, open_folder, system_language_key
from .settings import (CORE_DEFAULTS, DISPLAY_MODES, FPS_LIMITS, INI_SWITCHES, TEST_BUILD, VERSION_FILE,
                       Settings, parse_env)

LOG_LINES = 5000
CORE_SWITCHES = ('vsync', 'fsr1', 'rcas', 'show_fps', 'async_shaders')


def _url(path):
    return QUrl.fromLocalFile(str(path)).toString() if path else ''


class Launcher(QObject):
    textsChanged = Signal()
    gameChanged = Signal()
    settingsChanged = Signal()
    runningChanged = Signal()
    statusChanged = Signal()
    notice = Signal(str, str, str)   # title, text, kind ('warning' or 'done'): a toast
    showLog = Signal()               # the game failed: open the log page

    def __init__(self, parent=None):
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
        families = QFontDatabase.families()
        self._icon_font = next((f for f in ('Segoe Fluent Icons', 'Segoe MDL2 Assets') if f in families), '')
        self.runner = GameRunner(self)
        self.runner.line.connect(self._log_line)
        self.runner.finished.connect(self._game_finished)
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

    @Property(str, constant=True)
    def iconFont(self):
        return self._icon_font

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
        if self.info and self.info.icon:
            QGuiApplication.setWindowIcon(QIcon(str(self.info.icon)))
        self.gameChanged.emit()
        self.settingsChanged.emit()

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
    def displayModes(self):
        return [{'value': mode, 'text': self.t(mode)} for mode in DISPLAY_MODES]

    @Property('QVariantList', notify=textsChanged)
    def fpsLimits(self):
        return [{'value': v, 'text': self.t('unlimited') if v == '0' else v} for v in FPS_LIMITS]

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
        """The advanced variables (when the box is on); the ones run.py reads from tlg.ini go there."""
        extra = dict(game.FIRST_LIGHT_ENV)
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
        self.settings.save()
        self._clear_log()
        self._log_line(self.t('starting'))
        for key, value in extra.items():
            self._log_line(f'{key}={value}')
        self.runner.start(self.info.folder, self.settings.folder, self._language_id(), extra,
                          ui_language=self.t.language)
        self._set_status(self.t('running') if self.runner.running else '')
        self.runningChanged.emit()

    def _game_finished(self, code):
        message = self.t('exited', code=code)
        self._log_line(message)
        self._set_status(message)
        self.runningChanged.emit()
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
