// SPDX-License-Identifier: GPL-2.0-or-later
// One Segoe Fluent Icons glyph (theme.js icon names).
import QtQuick
import "theme.js" as T

Text {
    property string name
    property real size: 28
    text: T.icon[name] || ''
    font.family: launcher.iconFont
    font.pixelSize: size
    color: T.fg
    visible: launcher.iconFont !== ''
    horizontalAlignment: Text.AlignHCenter
    verticalAlignment: Text.AlignVCenter
}
