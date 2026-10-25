# SPDX-License-Identifier: GPL-2.0-or-later
"""The launcher window: the console-style Qt Quick interface (qml/Main.qml) over controller.Launcher."""
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle

from . import fonts
from .controller import Launcher
from .updates import Updater

QML = Path(__file__).resolve().parent / 'qml'


def application(argv):
    QQuickStyle.setStyle('Basic')
    app = QGuiApplication.instance() or QGuiApplication(argv)
    app.setApplicationName('inFAMOUS First Light')
    # The port's own icon (dev/make_icon.py), on the window and the taskbar.
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / 'assets' / 'icon.png')))
    app.styleHints().setColorScheme(Qt.ColorScheme.Dark)   # dark title bar
    return app


def create(argv, launcher=None):
    """(application, engine, launcher), the window already loaded (for `launcher` when given)."""
    app = application(argv)
    launcher = launcher or Launcher(fonts.setup())
    updater = Updater(lambda: launcher.t, launcher.settings.folder, lambda: launcher.runner.running, launcher)
    updater.notice.connect(launcher.notice)
    launcher.textsChanged.connect(updater.changed)
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty('launcher', launcher)
    engine.rootContext().setContextProperty('updater', updater)
    engine.load(QUrl.fromLocalFile(str(QML / 'Main.qml')))
    if not engine.rootObjects():
        raise SystemExit('the launcher interface (qml/Main.qml) did not load')
    app.aboutToQuit.connect(launcher.save)
    return app, engine, launcher


def main():
    app, _engine, _launcher = create(sys.argv)
    return app.exec()


def play():
    """`<launcher> --play`: the game with the launcher's settings, without the launcher window,
    for shortcuts and Steam's or Playnite's "add a game" (issue #9); the process ends with the
    game. The window opens instead when there is no game folder yet or the game cannot start,
    and after a failed run, on its log."""
    app = application(sys.argv)
    launcher = Launcher(fonts.setup())
    failed = []
    launcher.showLog.connect(lambda: failed.append(True))

    def finished():
        if not launcher.running:
            app.quit()

    launcher.runningChanged.connect(finished)
    if launcher.info:
        launcher.play()
    if launcher.running:
        app.exec()
        if not failed:
            return 0
    launcher.runningChanged.disconnect(finished)
    _app, _engine, _launcher = create(sys.argv, launcher)
    if failed:
        launcher.showLog.emit()
    return app.exec()


def selftest():
    """Loads the interface (QT_QPA_PLATFORM=offscreen in CI), draws for a moment and quits:
    'selftest ok' when it loaded and found its fonts."""
    from PySide6.QtCore import QTimer
    app, engine, launcher = create(sys.argv[:1])
    QTimer.singleShot(1500, app.quit)
    app.exec()
    ok = bool(engine.rootObjects()) and bool(launcher.iconFont)
    print('selftest ok' if ok else f'selftest failed: interface {bool(engine.rootObjects())}, icons {launcher.iconFont!r}',
          flush=True)
    return 0 if ok else 1
