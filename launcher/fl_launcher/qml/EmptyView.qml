// SPDX-License-Identifier: GPL-2.0-or-later
// First start: nothing but the request for the game folder, in the middle of the room.
import QtQuick
import "theme.js" as T

Item {
    id: view
    property bool shown: true
    signal browse

    opacity: shown ? 1 : 0
    visible: opacity > 0
    enabled: shown
    Behavior on opacity { NumberAnimation { duration: 550; easing.type: Easing.BezierSpline; easing.bezierCurve: T.ease } }

    function focusDefault() { choose.forceActiveFocus() }

    // the user-select screen's timeline: title, then the line under it, then the button
    property real step: 0
    NumberAnimation on step { id: intro; from: 0; to: 3; duration: 1500; running: false }
    onShownChanged: if (shown) { intro.restart(); focusDefault() }
    Component.onCompleted: if (shown) { intro.restart(); focusDefault() }

    Column {
        anchors.centerIn: parent
        anchors.verticalCenterOffset: -30
        width: Math.min(parent.width - 192, 1300)
        spacing: 0

        Text {
            width: parent.width
            horizontalAlignment: Text.AlignHCenter
            text: 'inFAMOUS First Light'
            font.family: T.font
            font.pixelSize: 64
            font.weight: Font.Light
            color: T.fg
            opacity: Math.min(1, view.step / 0.5)
        }
        Item { width: 1; height: 26 }
        Text {
            width: parent.width
            horizontalAlignment: Text.AlignHCenter
            text: launcher.tr.subtitle
            font.family: T.font
            font.pixelSize: 32
            font.weight: Font.Light
            color: Qt.rgba(1, 1, 1, 0.72)
            wrapMode: Text.Wrap
            opacity: Math.max(0, Math.min(1, (view.step - 0.6) / 0.5))
        }
        Item { width: 1; height: 120 }
        Text {
            width: parent.width
            horizontalAlignment: Text.AlignHCenter
            text: launcher.tr.no_game
            font.family: T.font
            font.pixelSize: 26
            font.weight: Font.Light
            color: T.fg2
            wrapMode: Text.Wrap
            opacity: Math.max(0, Math.min(1, (view.step - 1.4) / 0.5))
        }
        Item { width: 1; height: 44 }
        Item {
            width: parent.width
            height: choose.height
            opacity: Math.max(0, Math.min(1, (view.step - 1.8) / 0.5))
            scale: 0.92 + 0.08 * opacity
            PillButton {
                id: choose
                anchors.horizontalCenter: parent.horizontalCenter
                text: launcher.tr.choose_folder
                glyph: 'folder'
                minWidth: 420
                onClicked: view.browse()
            }
        }
        Item { width: 1; height: 40 }
        Text {
            width: parent.width
            horizontalAlignment: Text.AlignHCenter
            text: launcher.gameError
            visible: text !== ''
            font.family: T.font
            font.pixelSize: 24
            color: T.bad
            wrapMode: Text.Wrap
        }
        Text {
            width: parent.width
            horizontalAlignment: Text.AlignHCenter
            text: launcher.gameError ? launcher.gameFolder : ''
            visible: text !== ''
            topPadding: 8
            font.family: T.mono
            font.pixelSize: 18
            color: T.fg3
            wrapMode: Text.WrapAnywhere
        }
    }
}
