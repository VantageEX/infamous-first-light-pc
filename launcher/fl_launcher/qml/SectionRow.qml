// SPDX-License-Identifier: GPL-2.0-or-later
// A section in the settings panel's left column (the console's Settings sub-page): focusing it
// shows its page, right/Enter moves into the page.
import QtQuick
import "theme.js" as T

Item {
    id: row
    property string label
    property string glyph
    property bool current: false
    signal selected
    signal enter

    width: parent ? parent.width : 240
    height: 64
    activeFocusOnTab: true
    onActiveFocusChanged: if (activeFocus) selected()

    Rectangle {
        anchors.fill: parent
        radius: 6
        gradient: Gradient {
            GradientStop { position: 0; color: Qt.rgba(1, 1, 1, 0.07) }
            GradientStop { position: 1; color: Qt.rgba(1, 1, 1, 0.14) }
        }
        opacity: row.activeFocus ? 1 : row.current ? 0.6 : area.containsMouse ? 0.4 : 0
        Behavior on opacity { NumberAnimation { duration: 180 } }
        FocusRing { radius: 6; gap: 2; thickness: 2; shown: row.activeFocus }
    }
    Row {
        anchors.left: parent.left
        anchors.leftMargin: 18
        anchors.verticalCenter: parent.verticalCenter
        spacing: 16
        Glyph {
            name: row.glyph
            size: 22
            color: row.current ? T.fg : T.fg2
            anchors.verticalCenter: parent.verticalCenter
        }
        Text {
            text: row.label
            font.family: T.font
            font.pixelSize: 25
            font.weight: row.current ? Font.Normal : Font.Light
            color: row.current ? T.fg : Qt.rgba(1, 1, 1, 0.7)
            anchors.verticalCenter: parent.verticalCenter
        }
    }
    MouseArea {
        id: area
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onClicked: { row.forceActiveFocus(); row.selected() }
    }
    Keys.onReturnPressed: enter()
    Keys.onEnterPressed: enter()
    Keys.onRightPressed: enter()
    Keys.onUpPressed: nextItemInFocusChain(false).forceActiveFocus()
    Keys.onDownPressed: nextItemInFocusChain(true).forceActiveFocus()
}
