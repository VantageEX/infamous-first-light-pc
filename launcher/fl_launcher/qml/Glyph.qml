// SPDX-License-Identifier: GPL-2.0-or-later
// One icon glyph (theme.js icon names): Segoe Fluent Icons on Windows, Fluent UI System Icons elsewhere.
import QtQuick
import "theme.js" as T

Text {
    property string name
    property real size: 28
    text: (launcher.iconFluent ? T.iconFluent : T.icon)[name] || ''
    font.family: launcher.iconFont
    // Fluent UI System Icons draw a little smaller in their em than Segoe's: scaled to match.
    font.pixelSize: launcher.iconFluent ? Math.round(size * 1.15) : size
    color: T.fg
    visible: launcher.iconFont !== ''
    horizontalAlignment: Text.AlignHCenter
    verticalAlignment: Text.AlignVCenter
}
