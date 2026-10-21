// SPDX-License-Identifier: GPL-2.0-or-later
// On/off setting with the console's blue switch.
import QtQuick
import "theme.js" as T

SettingRow {
    id: row
    property bool checked
    signal toggled(bool on)
    onActivated: toggled(!checked)
    onStepped: direction => { if ((direction > 0) !== checked) toggled(direction > 0) }

    Rectangle {
        width: 64
        height: 34
        radius: 17
        color: row.checked ? T.blue : Qt.rgba(1, 1, 1, 0.28)
        Behavior on color { ColorAnimation { duration: 250 } }
        Rectangle {
            width: 26
            height: 26
            radius: 13
            y: 4
            x: row.checked ? 34 : 4
            color: 'white'
            Behavior on x { NumberAnimation { duration: 250; easing.type: Easing.BezierSpline; easing.bezierCurve: [0.2, 0.8, 0.2, 1, 1, 1] } }
        }
    }
}
