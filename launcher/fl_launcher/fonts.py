# SPDX-License-Identifier: GPL-2.0-or-later
"""The launcher's faces. The QML asks for one family, TEXT_ALIAS (qml/theme.js), and Qt
substitutes Segoe UI for it where Windows has it, else Open Sans (OFL, by Segoe UI's designer,
Cyrillic included; Linux and the Steam Deck). The icons are Segoe Fluent Icons / MDL2 on Windows,
else a subset of Microsoft's Fluent UI System Icons (MIT). Both bundled faces are in port/fonts.
FL_FONTS=bundled takes them on Windows too (the Linux look)."""
import os

from PySide6.QtGui import QFont, QFontDatabase

from .settings import PORT

FONTS = PORT / 'fonts'
TEXT_ALIAS = 'FL Sans'   # theme.js font: never a real family, always substituted
WINDOWS_TEXT = 'Segoe UI'


def setup():
    """After the application exists: substitutes the text family and returns the icon font's
    family ('' without one) and whether its glyphs are Fluent UI System Icons (theme.js
    iconFluent) rather than Segoe's."""
    bundled = os.environ.get('FL_FONTS') == 'bundled'
    families = set(QFontDatabase.families())
    text = WINDOWS_TEXT
    if bundled or WINDOWS_TEXT not in families:
        for name in ('OpenSans-Light.ttf', 'OpenSans-Regular.ttf', 'OpenSans-SemiBold.ttf'):
            font_id = QFontDatabase.addApplicationFont(str(FONTS / name))
            if font_id >= 0 and 'Open Sans' in QFontDatabase.applicationFontFamilies(font_id):
                text = 'Open Sans'
    QFont.insertSubstitution(TEXT_ALIAS, text)
    # theme.js mono (the log): Consolas is Windows'; elsewhere the system's fixed-width face.
    if 'Consolas' not in families:
        QFont.insertSubstitution('Consolas', QFontDatabase.systemFont(QFontDatabase.FixedFont).family())
    if not bundled:
        for family in ('Segoe Fluent Icons', 'Segoe MDL2 Assets'):
            if family in families:
                return family, False
    font_id = QFontDatabase.addApplicationFont(str(FONTS / 'FluentSystemIcons-Subset.ttf'))
    loaded = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
    return (loaded[0], True) if loaded else ('', False)
