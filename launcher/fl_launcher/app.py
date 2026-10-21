# SPDX-License-Identifier: GPL-2.0-or-later
"""The launcher window: the console-style Qt Quick interface (qml/Main.qml) over controller.Launcher."""
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle

from .controller import Launcher

QML = Path(__file__).resolve().parent / 'qml'


def create(argv):
    """(application, engine, launcher), the window already loaded."""
    QQuickStyle.setStyle('Basic')
    app = QGuiApplication.instance() or QGuiApplication(argv)
    app.setApplicationName('inFAMOUS First Light')
    # The port's own icon (dev/make_icon.py), on the window and the taskbar.
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / 'assets' / 'icon.png')))
    app.styleHints().setColorScheme(Qt.ColorScheme.Dark)   # dark title bar
    launcher = Launcher()
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty('launcher', launcher)
    engine.load(QUrl.fromLocalFile(str(QML / 'Main.qml')))
    if not engine.rootObjects():
        raise SystemExit('the launcher interface (qml/Main.qml) did not load')
    app.aboutToQuit.connect(launcher.save)
    return app, engine, launcher


def main():
    app, _engine, _launcher = create(sys.argv)
    return app.exec()
