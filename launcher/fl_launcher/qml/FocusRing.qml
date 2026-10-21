// SPDX-License-Identifier: GPL-2.0-or-later
// The breathing white outline around the focused control.
import QtQuick
import QtQuick.Effects

Item {
    id: ring
    property real radius: 0       // the control's corner radius
    property real gap: 6
    property real thickness: 3
    property bool shown: parent ? parent.activeFocus : false
    property real pulse: 0

    anchors.fill: parent
    anchors.margins: -gap
    opacity: shown ? 1 : 0
    visible: opacity > 0
    Behavior on opacity { NumberAnimation { duration: 200 } }

    SequentialAnimation on pulse {
        running: ring.visible
        loops: Animation.Infinite
        NumberAnimation { from: 0; to: 1; duration: 1100; easing.type: Easing.InOutSine }
        NumberAnimation { from: 1; to: 0; duration: 1100; easing.type: Easing.InOutSine }
    }

    Rectangle {
        id: outline
        anchors.fill: parent
        radius: ring.radius + ring.gap
        color: 'transparent'
        border.width: ring.thickness
        border.color: Qt.rgba(1, 1, 1, 0.92 + 0.08 * ring.pulse)
        visible: false
    }
    MultiEffect {
        source: outline
        anchors.fill: outline
        shadowEnabled: true
        shadowColor: 'white'
        shadowOpacity: 0.45 + 0.3 * ring.pulse
        shadowBlur: 0.45 + 0.4 * ring.pulse
        blurMax: 24
        shadowHorizontalOffset: 0
        shadowVerticalOffset: 0
    }
}
