// SPDX-License-Identifier: GPL-2.0-or-later
// Setting with a list of values: Enter/click opens the list, left/right step through it.
import QtQuick
import "theme.js" as T

SettingRow {
    id: row
    property var options: []      // [{value, text}]
    property string value
    signal picked(string value)

    readonly property int index: {
        for (let i = 0; i < options.length; i++)
            if (options[i].value === value) return i
        return 0
    }
    onActivated: Window.window.openPicker(row, options, value, v => row.picked(v))
    onStepped: direction => {
        const next = index + direction
        if (next >= 0 && next < options.length) picked(options[next].value)
    }

    Row {
        spacing: 14
        Text {
            text: row.options.length ? row.options[row.index].text : ''
            font.family: T.font
            font.pixelSize: 24
            color: Qt.rgba(1, 1, 1, 0.72)
            anchors.verticalCenter: parent.verticalCenter
        }
        Glyph {
            name: 'chevronDown'
            size: 16
            color: T.fg2
            anchors.verticalCenter: parent.verticalCenter
        }
    }
}
