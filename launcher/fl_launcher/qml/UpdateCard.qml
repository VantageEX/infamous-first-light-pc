// SPDX-License-Identifier: GPL-2.0-or-later
// A newer release (updates.py Updater): its version, "Update" (or the release page where the
// launcher cannot install it), then the download's progress. Stays until dismissed.
import QtQuick
import QtQuick.Effects
import "theme.js" as T

Item {
    id: card
    readonly property bool shown: updater.state !== ''
    readonly property bool busy: updater.state === 'downloading' || updater.state === 'installing'
    width: 640
    height: box.height

    opacity: shown ? 1 : 0
    visible: opacity > 0
    transform: Translate { x: card.shown ? 0 : card.width * 1.1; Behavior on x { NumberAnimation { duration: 500; easing.type: Easing.BezierSpline; easing.bezierCurve: T.ease } } }
    Behavior on opacity { NumberAnimation { duration: 350 } }

    RectangularShadow {
        anchors.fill: box
        offset.y: 18
        blur: 50
        radius: box.radius
        color: Qt.rgba(0, 0, 0, 0.55)
    }
    Rectangle {
        id: box
        width: parent.width
        height: content.implicitHeight + 44
        radius: 16
        color: Qt.rgba(32 / 255, 32 / 255, 35 / 255, 0.95)

        Rectangle {
            id: badge
            x: 22
            y: 22
            width: 68
            height: 68
            radius: 10
            color: Qt.rgba(1, 1, 1, 0.08)
            Glyph {
                anchors.centerIn: parent
                name: updater.state === 'error' ? 'warning' : 'download'
                size: 34
                color: updater.state === 'error' ? T.warn : T.blue
            }
        }
        Column {
            id: content
            anchors.left: badge.right
            anchors.leftMargin: 20
            anchors.right: parent.right
            anchors.rightMargin: 22
            y: 22
            spacing: 10
            Text { width: parent.width; text: updater.title; font.family: T.font; font.pixelSize: 23; font.weight: Font.DemiBold; color: T.fg; wrapMode: Text.Wrap }
            Text { width: parent.width; text: updater.detail; visible: text !== ''; font.family: T.font; font.pixelSize: 20; color: T.fg2; wrapMode: Text.Wrap }
            Rectangle {
                width: parent.width
                height: 8
                radius: 4
                visible: card.busy
                color: Qt.rgba(1, 1, 1, 0.1)
                Rectangle {
                    width: parent.width * updater.progress / 100
                    height: parent.height
                    radius: parent.radius
                    color: T.blue
                    Behavior on width { NumberAnimation { duration: 200 } }
                }
            }
            Row {
                spacing: 12
                visible: !card.busy
                topPadding: 4
                PillButton {
                    size: 56
                    fontSize: 22
                    minWidth: 170
                    visible: updater.state === 'available'
                    text: updater.canInstall ? launcher.tr.update_install : launcher.tr.update_page
                    glyph: 'download'
                    enabled: !launcher.running
                    onClicked: updater.install()
                }
                PillButton {
                    size: 56
                    fontSize: 22
                    minWidth: 150
                    visible: updater.state === 'available' && updater.canInstall
                    text: launcher.tr.update_page
                    onClicked: updater.openPage()
                }
                PillButton {
                    size: 56
                    fontSize: 22
                    minWidth: 130
                    text: launcher.tr.update_later
                    onClicked: updater.dismiss()
                }
            }
        }
    }
}
