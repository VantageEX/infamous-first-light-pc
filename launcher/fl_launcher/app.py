# SPDX-License-Identifier: GPL-2.0-or-later
"""The launcher window."""
import sys

from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPlainTextEdit,
                               QPushButton, QToolButton, QVBoxLayout, QWidget)

from . import game
from .i18n import UI_LANGUAGES, Texts
from .runner import GameRunner, open_folder, system_language_key
from .settings import CORE_DEFAULTS, DISPLAY_MODES, FPS_LIMITS, INI_SWITCHES, Settings, parse_env

STYLE = """
QWidget { background: #0d0b14; color: #e9e4f5; font-size: 10pt; }
QGroupBox { border: 1px solid #2e2540; border-radius: 8px; margin-top: 14px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #ff4fa8; font-weight: 600; }
QLineEdit, QComboBox, QPlainTextEdit { background: #171320; border: 1px solid #2e2540; border-radius: 6px; padding: 5px; }
QComboBox QAbstractItemView { background: #171320; selection-background-color: #5b2a86; }
QPushButton { background: #221b30; border: 1px solid #3a2f52; border-radius: 6px; padding: 6px 12px; }
QPushButton:hover { border-color: #ff4fa8; }
QPushButton#play { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff2d95, stop:1 #8a3ffc);
                   border: none; color: white; font-size: 15pt; font-weight: 700; padding: 12px; letter-spacing: 2px; }
QPushButton#play:disabled { background: #2a2338; color: #6f6684; }
QPushButton#play[running="true"] { background: #3a2f52; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #4a3d66; border-radius: 4px; background: #171320; }
QCheckBox::indicator:checked { background: #ff2d95; border-color: #ff2d95; }
QLabel#title { font-size: 20pt; font-weight: 700; color: #ffffff; }
QLabel#status { color: #b9b0cc; }
QLabel#warning { color: #ffb347; }
QLabel#credits { color: #6f6684; font-size: 8pt; }
QToolButton { border: none; color: #b9b0cc; }
"""


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.info = None
        ui = self.settings.launcher['ui_language']
        self.t = Texts(QLocale.system().name()[:2] if ui == 'auto' else ui)
        self.runner = GameRunner(self)
        self.runner.line.connect(self._log_line)
        self.runner.finished.connect(self._game_finished)
        self._build()
        self._load_game(self.settings.launcher['game_dir'], quiet=True)
        self.retranslate()

    # ---- layout -------------------------------------------------------------------------------
    def _build(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 12)
        layout.setSpacing(10)

        header = QHBoxLayout()
        self.icon = QLabel()
        self.icon.setFixedSize(96, 96)
        self.icon.setScaledContents(True)
        header.addWidget(self.icon)
        titles = QVBoxLayout()
        self.title = QLabel('inFAMOUS First Light')
        self.title.setObjectName('title')
        self.subtitle = QLabel()
        self.subtitle.setObjectName('status')
        titles.addWidget(self.title)
        titles.addWidget(self.subtitle)
        titles.addStretch()
        header.addLayout(titles, 1)
        layout.addLayout(header)

        self.game_box = QGroupBox()
        game_layout = QVBoxLayout(self.game_box)
        row = QHBoxLayout()
        self.folder = QLineEdit()
        self.folder.setReadOnly(True)
        self.browse = QPushButton()
        self.browse.clicked.connect(self._browse)
        row.addWidget(self.folder, 1)
        row.addWidget(self.browse)
        game_layout.addLayout(row)
        self.status = QLabel()
        self.status.setObjectName('status')
        self.status.setWordWrap(True)
        self.warning = QLabel()
        self.warning.setObjectName('warning')
        self.warning.setWordWrap(True)
        game_layout.addWidget(self.status)
        game_layout.addWidget(self.warning)
        lang_form = QFormLayout()
        self.game_language = QComboBox()
        self.game_language.currentIndexChanged.connect(self._game_language_changed)
        self.ui_language = QComboBox()
        self.ui_language.addItem('Auto', 'auto')
        for key, name in UI_LANGUAGES.items():
            self.ui_language.addItem(name, key)
        self.ui_language.setCurrentIndex(max(0, self.ui_language.findData(self.settings.launcher['ui_language'])))
        self.ui_language.currentIndexChanged.connect(self._ui_language_changed)
        self.game_language_label, self.ui_language_label = QLabel(), QLabel()
        lang_form.addRow(self.game_language_label, self.game_language)
        lang_form.addRow(self.ui_language_label, self.ui_language)
        game_layout.addLayout(lang_form)
        layout.addWidget(self.game_box)

        self.display_box = QGroupBox()
        form = QFormLayout(self.display_box)
        self.display_mode = QComboBox()
        self.fps_limit = QComboBox()
        self.display_mode_label, self.fps_limit_label = QLabel(), QLabel()
        form.addRow(self.display_mode_label, self.display_mode)
        form.addRow(self.fps_limit_label, self.fps_limit)
        self.checks = {}
        for key in ('vsync', 'fsr1', 'rcas', 'show_fps', 'async_shaders'):
            box = QCheckBox()
            box.setChecked(self.settings.core[key] == '1')
            box.toggled.connect(lambda on, k=key: self._set_core(k, '1' if on else '0'))
            self.checks[key] = box
            form.addRow(box)
        self.display_mode.currentIndexChanged.connect(
            lambda: self._set_core('display_mode', self.display_mode.currentData()))
        self.fps_limit.currentIndexChanged.connect(
            lambda: self._set_core('fps_limit', self.fps_limit.currentData()))
        layout.addWidget(self.display_box)

        self.advanced_box = QGroupBox()
        self.advanced_box.setCheckable(True)
        self.advanced_box.setChecked(bool(self.settings.launcher['extra_env'].strip()))
        advanced = QVBoxLayout(self.advanced_box)
        self.extra_env = QPlainTextEdit(self.settings.launcher['extra_env'])
        self.extra_env.setFont(QFont('Consolas', 9))
        self.extra_env.setFixedHeight(70)
        self.extra_env.textChanged.connect(self._extra_env_changed)
        advanced.addWidget(self.extra_env)
        self.advanced_box.toggled.connect(self.extra_env.setVisible)
        self.extra_env.setVisible(self.advanced_box.isChecked())
        layout.addWidget(self.advanced_box)

        self.play = QPushButton()
        self.play.setObjectName('play')
        self.play.clicked.connect(self._play)
        layout.addWidget(self.play)

        tools = QHBoxLayout()
        self.log_toggle = QToolButton()
        self.log_toggle.setCheckable(True)
        self.log_toggle.toggled.connect(lambda on: self.log.setVisible(on))
        self.open_logs = QPushButton()
        self.open_logs.clicked.connect(lambda: open_folder(self.settings.folder / 'logs'))
        self.open_saves = QPushButton()
        self.open_saves.clicked.connect(lambda: open_folder(self.settings.folder / 'user'))
        tools.addWidget(self.log_toggle)
        tools.addStretch()
        tools.addWidget(self.open_logs)
        tools.addWidget(self.open_saves)
        layout.addLayout(tools)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(QFont('Consolas', 9))
        self.log.setVisible(False)
        layout.addWidget(self.log, 1)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet('color: #2e2540;')
        layout.addWidget(line)
        self.credits = QLabel()
        self.credits.setObjectName('credits')
        self.credits.setWordWrap(True)
        layout.addWidget(self.credits)

        self.setCentralWidget(root)
        self.resize(620, 760)

    def retranslate(self):
        t = self.t
        self.setWindowTitle(t('window_title'))
        self.subtitle.setText(t('subtitle'))
        self.game_box.setTitle(t('game_folder'))
        self.browse.setText(t('browse'))
        self.game_language_label.setText(t('game_language'))
        self.ui_language_label.setText(t('ui_language'))
        self.display_box.setTitle(t('display'))
        self.display_mode_label.setText(t('display_mode'))
        self.fps_limit_label.setText(t('fps_limit'))
        for key, box in self.checks.items():
            box.setText(t(key))
        self._fill(self.display_mode, [(t(m), m) for m in DISPLAY_MODES], self.settings.core['display_mode'].lower())
        self._fill(self.fps_limit, [(t('unlimited') if v == '0' else v, v) for v in FPS_LIMITS],
                   self.settings.core['fps_limit'])
        self._fill_game_languages()
        self.advanced_box.setTitle(t('advanced'))
        self.extra_env.setPlaceholderText(t('extra_env_hint'))
        self.log_toggle.setText(t('log'))
        self.open_logs.setText(t('open_logs'))
        self.open_saves.setText(t('open_saves'))
        self.credits.setText(t('credits'))
        self._show_game()
        self._update_play()

    @staticmethod
    def _fill(combo, items, current):
        combo.blockSignals(True)
        combo.clear()
        for text, data in items:
            combo.addItem(text, data)
        combo.setCurrentIndex(max(0, combo.findData(current)))
        combo.blockSignals(False)

    def _fill_game_languages(self):
        keys = (self.info.languages if self.info and self.info.languages else list(game.LANGUAGES))
        items = [(self.t('language_auto'), 'auto')] + [(game.LANGUAGE_NAMES[k], k) for k in keys]
        self._fill(self.game_language, items, self.settings.launcher['game_language'])

    # ---- game folder --------------------------------------------------------------------------
    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, self.t('game_folder'), self.folder.text())
        if folder:
            self._load_game(folder)

    def _load_game(self, folder, quiet=False):
        self.info, self.game_error = None, None
        if folder:
            try:
                self.info = game.inspect(folder)
            except ValueError as error:
                reason = str(error)
                self.game_error = self.t(reason) if reason in ('no_eboot', 'no_sfo') else self.t('bad_sfo', error=reason)
            except OSError as error:
                self.game_error = self.t('bad_sfo', error=error)
        if self.info and not quiet:
            self.settings.launcher['game_dir'] = str(self.info.folder)
            self.settings.save()
        self.folder.setText(str(folder or ''))
        self._fill_game_languages()
        self._show_game()
        self._update_play()

    def _show_game(self):
        t = self.t
        self.warning.clear()
        if self.info:
            self.status.setText(t('game_ok', title=self.info.title or '?', title_id=self.info.title_id or '?',
                                  version=self.info.version or '?'))
            if not self.info.known:
                self.warning.setText(t('game_untested', title_id=self.info.title_id or '?'))
            if self.info.icon:
                self.icon.setPixmap(QPixmap(str(self.info.icon)))
                self.setWindowIcon(QIcon(str(self.info.icon)))
        else:
            self.status.setText(getattr(self, 'game_error', None) or t('no_game'))

    # ---- settings -----------------------------------------------------------------------------
    def _set_core(self, key, value):
        if value is None:
            return
        self.settings.core[key] = value
        self.settings.save()

    def _extra_env_changed(self):
        self.settings.launcher['extra_env'] = self.extra_env.toPlainText()
        self.settings.save()

    def _extra_env(self):
        """The advanced variables (when the box is on); the ones run.py reads from tlg.ini go there."""
        extra = dict(game.FIRST_LIGHT_ENV)
        if self.advanced_box.isChecked():
            extra.update(parse_env(self.settings.launcher['extra_env']))
        for variable, key in INI_SWITCHES.items():
            value = extra.pop(variable, CORE_DEFAULTS[key])
            self.settings.core[key] = '0' if value == '0' else '1'
        return extra

    def _game_language_changed(self):
        value = self.game_language.currentData()
        if value:
            self.settings.launcher['game_language'] = value
            self.settings.save()

    def _ui_language_changed(self):
        value = self.ui_language.currentData()
        self.settings.launcher['ui_language'] = value
        self.settings.save()
        self.t = Texts(QLocale.system().name()[:2] if value == 'auto' else value)
        self.retranslate()

    # ---- running ------------------------------------------------------------------------------
    def _language_id(self):
        key = self.settings.launcher['game_language']
        available = self.info.languages if self.info else []
        if key == 'auto' or key not in game.LANGUAGES:
            key = system_language_key(QLocale.system().name(), available)
        return game.LANGUAGES[key]

    def _play(self):
        if self.runner.running:
            self.runner.stop()
            return
        if not self.info:
            return
        extra = self._extra_env()
        self.settings.save()
        self.log.clear()
        self._log_line(self.t('starting'))
        for key, value in extra.items():
            self._log_line(f'{key}={value}')
        self.runner.start(self.info.folder, self.settings.folder, self._language_id(), extra)
        self._update_play()

    def _log_line(self, text):
        self.log.appendPlainText(text)

    def _game_finished(self, code):
        self._log_line(self.t('exited', code=code))
        if code:
            self.log_toggle.setChecked(True)
        self._update_play()

    def _update_play(self):
        running = self.runner.running
        self.play.setText(self.t('stop') if running else self.t('play'))
        self.play.setEnabled(running or self.info is not None)
        self.play.setProperty('running', 'true' if running else 'false')
        self.play.style().unpolish(self.play)
        self.play.style().polish(self.play)

    def closeEvent(self, event):
        self.settings.save()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('inFAMOUS First Light')
    app.setStyleSheet(STYLE)
    window = LauncherWindow()
    window.show()
    return app.exec()
