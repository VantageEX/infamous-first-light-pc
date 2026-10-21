// SPDX-License-Identifier: GPL-2.0-or-later
// The launcher window. Everything is laid out on a 1920x1080 stage scaled to the window, as the
// console's interface is; the backgrounds fill the whole window.
import QtQuick
import QtQuick.Dialogs
import "theme.js" as T

Window {
    id: root
    width: 1440
    height: 810
    minimumWidth: 960
    minimumHeight: 540
    visible: true
    color: 'black'
    title: launcher.tr.window_title
    // no system title bar: WindowChrome draws the buttons over the launcher's own background
    flags: Qt.Window | Qt.FramelessWindowHint

    readonly property real k: Math.min(width / 1920, height / 1080)

    function openPicker(anchor, options, value, callback) { picker.open(anchor, options, value, callback) }
    function browse() { folderDialog.open() }

    FolderDialog {
        id: folderDialog
        objectName: "folderDialog"
        title: launcher.tr.game_folder
        currentFolder: launcher.gameFolderUrl
        onAccepted: launcher.chooseFolder(selectedFolder.toString())
    }

    Connections {
        target: launcher
        function onNotice(title, text) { toast.show(title, text) }
        function onShowLog() { gameView.showLog() }
    }

    Shortcut {
        sequence: 'F11'
        onActivated: root.visibility = root.visibility === Window.FullScreen ? Window.Windowed : Window.FullScreen
    }

    // ---- backgrounds ---------------------------------------------------------------------------
    Ambient {
        anchors.fill: parent
        opacity: launcher.hasGame ? 0 : 1
        visible: opacity > 0
        Behavior on opacity { NumberAnimation { duration: 900 } }
    }

    Item {
        id: backdrop
        anchors.fill: parent
        opacity: launcher.hasGame ? 1 : 0
        visible: opacity > 0
        Behavior on opacity { NumberAnimation { duration: 900; easing.type: Easing.BezierSpline; easing.bezierCurve: T.ease } }

        // the game's own art; it settles in when it appears and then drifts slowly (paused in game)
        Item {
            id: settle
            anchors.fill: parent
            scale: launcher.hasGame ? 1 : 1.06
            Behavior on scale { NumberAnimation { duration: 1400; easing.type: Easing.BezierSpline; easing.bezierCurve: T.ease } }
            Image {
                id: art
                anchors.fill: parent
                anchors.margins: -40 * root.k
                source: launcher.backgroundUrl
                fillMode: Image.PreserveAspectCrop
                asynchronous: true
                smooth: true
                cache: false
                property real drift: 0
                scale: 1.02 + 0.1 * drift
                transform: Translate { x: -30 * root.k * art.drift; y: -14 * root.k * art.drift }
                SequentialAnimation on drift {
                    running: launcher.hasGame
                    paused: launcher.running || root.visibility === Window.Minimized
                    loops: Animation.Infinite
                    NumberAnimation { from: 0; to: 1; duration: 40000; easing.type: Easing.InOutSine }
                    NumberAnimation { from: 1; to: 0; duration: 40000; easing.type: Easing.InOutSine }
                }
            }
        }
        // shade: dark top and bottom, dark on the left under the text and on the right under the panel
        Rectangle {
            anchors.fill: parent
            gradient: Gradient {
                GradientStop { position: 0; color: Qt.rgba(0, 0, 0, 0.55) }
                GradientStop { position: 0.24; color: Qt.rgba(0, 0, 0, 0.12) }
                GradientStop { position: 0.45; color: Qt.rgba(0, 0, 0, 0) }
                GradientStop { position: 0.7; color: Qt.rgba(0, 0, 0, 0.18) }
                GradientStop { position: 1; color: Qt.rgba(0, 0, 0, 0.6) }
            }
        }
        Rectangle {
            anchors.fill: parent
            gradient: Gradient {
                orientation: Gradient.Horizontal
                GradientStop { position: 0; color: Qt.rgba(0, 0, 0, 0.62) }
                GradientStop { position: 0.38; color: Qt.rgba(0, 0, 0, 0.25) }
                GradientStop { position: 0.55; color: Qt.rgba(0, 0, 0, 0) }
                GradientStop { position: 0.75; color: Qt.rgba(0, 0, 0, 0.12) }
                GradientStop { position: 1; color: Qt.rgba(0, 0, 0, 0.4) }
            }
        }
    }

    // ---- the stage -----------------------------------------------------------------------------
    Item {
        id: stage
        width: root.width / root.k
        height: root.height / root.k
        scale: root.k
        transformOrigin: Item.TopLeft

        EmptyView {
            id: emptyView
            anchors.fill: parent
            shown: !launcher.hasGame
            onBrowse: root.browse()
        }
        GameView {
            id: gameView
            anchors.fill: parent
            shown: launcher.hasGame
            backdrop: backdrop
            stageScale: root.k
            onBrowse: root.browse()
        }
        Picker {
            id: picker
            z: 10
        }
        Toast {
            id: toast
            z: 20
            anchors.right: parent.right
            anchors.rightMargin: 72
            y: 130
        }
    }

    WindowChrome {
        z: 100
        window: root
    }
}
