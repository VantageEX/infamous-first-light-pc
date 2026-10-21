// SPDX-License-Identifier: GPL-2.0-or-later
// Round icon button with its name above it while focused or hovered.
import QtQuick
import "theme.js" as T

Rectangle {
    id: button
    property string glyph
    property string tip
    signal clicked

    readonly property bool lit: activeFocus || area.pressed
    width: 74
    height: 74
    radius: width / 2
    color: lit ? 'white' : area.containsMouse ? Qt.rgba(1, 1, 1, 0.24) : Qt.rgba(1, 1, 1, 0.14)
    activeFocusOnTab: true
    Behavior on color { ColorAnimation { duration: 200 } }

    Glyph {
        anchors.centerIn: parent
        name: button.glyph
        size: 30
        color: button.lit ? '#000' : T.fg
    }
    Text {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.top
        anchors.bottomMargin: 18
        text: button.tip
        font.family: T.font
        font.pixelSize: 24
        font.weight: Font.Light
        color: T.fg
        style: Text.Raised
        styleColor: Qt.rgba(0, 0, 0, 0.35)
        opacity: button.activeFocus || area.containsMouse ? 1 : 0
        Behavior on opacity { NumberAnimation { duration: 200 } }
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
