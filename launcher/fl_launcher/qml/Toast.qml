// SPDX-License-Identifier: GPL-2.0-or-later
// Notification card sliding in at the top right.
import QtQuick
import QtQuick.Effects
import "theme.js" as T

Item {
    id: toast
    property string title
    property string text
    property bool shown: false
    width: 560
    height: card.height

    function show(t, x) {
        title = t
        text = x
        shown = true
        timer.restart()
    }

    opacity: shown ? 1 : 0
    visible: opacity > 0
    transform: Translate { x: toast.shown ? 0 : toast.width * 1.1; Behavior on x { NumberAnimation { duration: 500; easing.type: Easing.BezierSpline; easing.bezierCurve: [0.2, 0.8, 0.2, 1, 1, 1] } } }
    Behavior on opacity { NumberAnimation { duration: 350 } }
    Timer { id: timer; interval: 7000; onTriggered: toast.shown = false }

    RectangularShadow {
        anchors.fill: card
        offset.y: 18
        blur: 50
        radius: card.radius
        color: Qt.rgba(0, 0, 0, 0.55)
    }
    Rectangle {
        id: card
        width: parent.width
        height: Math.max(104, body.implicitHeight + 36)
        radius: 16
        color: Qt.rgba(32 / 255, 32 / 255, 35 / 255, 0.95)
        Rectangle {
            id: badge
            x: 22
            anchors.verticalCenter: parent.verticalCenter
            width: 68
            height: 68
            radius: 10
            color: Qt.rgba(1, 1, 1, 0.08)
            Glyph { anchors.centerIn: parent; name: 'warning'; size: 34; color: T.warn }
        }
        Column {
            id: body
            anchors.left: badge.right
            anchors.leftMargin: 20
            anchors.right: parent.right
            anchors.rightMargin: 22
            anchors.verticalCenter: parent.verticalCenter
            spacing: 4
            Text { width: parent.width; text: toast.title; font.family: T.font; font.pixelSize: 23; font.weight: Font.DemiBold; color: T.fg; elide: Text.ElideRight }
            Text { width: parent.width; text: toast.text; font.family: T.font; font.pixelSize: 20; color: T.fg2; wrapMode: Text.Wrap }
        }
        MouseArea { anchors.fill: parent; onClicked: toast.shown = false }
    }
}
