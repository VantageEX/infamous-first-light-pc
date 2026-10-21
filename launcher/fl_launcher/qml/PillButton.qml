// SPDX-License-Identifier: GPL-2.0-or-later
// The console's rounded button: frosted when idle, white with black text when focused.
import QtQuick
import "theme.js" as T

Rectangle {
    id: button
    property string text
    property string glyph
    property real size: 74
    property real fontSize: 30
    property real minWidth: 336
    signal clicked

    readonly property bool lit: activeFocus || (area.pressed && enabled)
    implicitHeight: size
    implicitWidth: Math.max(minWidth, row.implicitWidth + size)
    radius: height / 2
    color: lit ? 'white' : area.containsMouse ? Qt.rgba(1, 1, 1, 0.24) : Qt.rgba(1, 1, 1, 0.14)
    opacity: enabled ? 1 : 0.45
    activeFocusOnTab: true
    scale: area.pressed ? 0.97 : 1
    Behavior on color { ColorAnimation { duration: 200 } }
    Behavior on scale { NumberAnimation { duration: 150; easing.type: Easing.OutCubic } }

    Row {
        id: row
        anchors.centerIn: parent
        spacing: 14
        Glyph {
            name: button.glyph
            size: button.fontSize * 0.85
            visible: button.glyph !== '' && launcher.iconFont !== ''
            color: button.lit ? '#000' : T.fg
            anchors.verticalCenter: parent.verticalCenter
            Behavior on color { ColorAnimation { duration: 200 } }
        }
        Text {
            text: button.text
            font.family: T.font
            font.pixelSize: button.fontSize
            font.weight: Font.Medium
            color: button.lit ? '#000' : T.fg
            anchors.verticalCenter: parent.verticalCenter
            Behavior on color { ColorAnimation { duration: 200 } }
        }
    }

    MouseArea {
        id: area
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onClicked: { button.forceActiveFocus(); button.clicked() }
    }
    Keys.onReturnPressed: clicked()
    Keys.onEnterPressed: clicked()
    Keys.onSpacePressed: clicked()
    Keys.onLeftPressed: nextItemInFocusChain(false).forceActiveFocus()
    Keys.onRightPressed: nextItemInFocusChain(true).forceActiveFocus()
    Keys.onUpPressed: nextItemInFocusChain(false).forceActiveFocus()
    Keys.onDownPressed: nextItemInFocusChain(true).forceActiveFocus()

    FocusRing { radius: button.radius }
}
