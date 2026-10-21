// SPDX-License-Identifier: GPL-2.0-or-later
// A settings line: label (and a note) on the left, its value on the right, the console's focus
// highlight. Enter/click activates it, left/right step it, up/down move to the next line.
import QtQuick
import "theme.js" as T

Item {
    id: row
    property string label
    property string description
    property bool last: false
    default property alias trailing: slot.data
    signal activated
    signal stepped(int direction)

    width: parent ? parent.width : 600
    implicitHeight: Math.max(66, texts.implicitHeight + 24)
    activeFocusOnTab: true

    Rectangle {
        anchors.fill: parent
        anchors.leftMargin: -20
        anchors.rightMargin: -20
        radius: 6
        gradient: Gradient {
            GradientStop { position: 0; color: Qt.rgba(1, 1, 1, 0.07) }
            GradientStop { position: 1; color: Qt.rgba(1, 1, 1, 0.14) }
        }
        opacity: row.activeFocus ? 1 : area.containsMouse ? 0.55 : 0
        Behavior on opacity { NumberAnimation { duration: 180 } }
        FocusRing { radius: 6; gap: 2; thickness: 2; shown: row.activeFocus }
    }
    Rectangle {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: 1
        color: T.line
        visible: !row.last
        opacity: row.activeFocus ? 0 : 1
    }
    Column {
        id: texts
        anchors.left: parent.left
        anchors.right: slot.left
        anchors.rightMargin: 28
        anchors.verticalCenter: parent.verticalCenter
        spacing: 4
        Text {
            width: parent.width
            text: row.label
            font.family: T.font
            font.pixelSize: 25
            font.weight: Font.Light
            color: T.fg
            wrapMode: Text.Wrap
        }
        Text {
            width: parent.width
            visible: text !== ''
            text: row.description
            font.family: T.font
            font.pixelSize: 19
            color: T.fg2
            wrapMode: Text.Wrap
        }
    }
    Item {
        id: slot
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter
        width: childrenRect.width
        height: childrenRect.height
    }
    MouseArea {
        id: area
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onClicked: { row.forceActiveFocus(); row.activated() }
    }
    Keys.onReturnPressed: activated()
    Keys.onEnterPressed: activated()
    Keys.onSpacePressed: activated()
    Keys.onLeftPressed: stepped(-1)
    Keys.onRightPressed: stepped(1)
    Keys.onUpPressed: nextItemInFocusChain(false).forceActiveFocus()
    Keys.onDownPressed: nextItemInFocusChain(true).forceActiveFocus()
}
