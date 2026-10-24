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
    // 960 x 540 at most the screen's free area: a 200% handheld (2560 x 1600, OneXPlayer X1) has
    // 1280 x 800 logical pixels, less than the default size.
    minimumWidth: Math.min(960, Screen.width)
    minimumHeight: Math.min(540, Screen.height)
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
        function onNotice(title, text, kind) { toast.show(title, text, kind) }
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
        // Renderer v3 picked: its experimental notice across the whole launcher, in the free strip
        // under the settings panel and the credits (both end 56 px above the bottom).
        Rectangle {
            id: experimentalBar
            z: 5
            readonly property bool shown: launcher.hasGame && launcher.experimentalRenderer
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            height: 46
            color: Qt.rgba(64 / 255, 44 / 255, 8 / 255, 0.94)
            opacity: shown ? 1 : 0
            visible: opacity > 0
            Behavior on opacity { NumberAnimation { duration: 350 } }
            Rectangle {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                height: 2
                color: T.warn
            }
            Row {
                anchors.centerIn: parent
                spacing: 14
                Glyph { name: 'warning'; size: 24; color: T.warn; anchors.verticalCenter: parent.verticalCenter }
                Text {
                    anchors.verticalCenter: parent.verticalCenter
                    text: launcher.tr.v3_notice_title
                    font.family: T.font
                    font.pixelSize: 21
                    font.weight: Font.DemiBold
                    color: T.warn
                }
                Text {
                    anchors.verticalCenter: parent.verticalCenter
                    text: launcher.tr.v3_notice
                    font.family: T.font
                    font.pixelSize: 19
                    color: T.fg
                }
            }
        }
        Picker {
            id: picker
            z: 10
        }
        UpdateCard {
            id: updateCard
            z: 19
            anchors.right: parent.right
            anchors.rightMargin: 72
            y: 130
        }
        Toast {
            id: toast
            z: 20
            anchors.right: parent.right
            anchors.rightMargin: 72
            // under the update card while it is up
            y: updateCard.shown ? 130 + updateCard.height + 24 : 130
        }
    }

    // Within the window's screen, centred: a frameless window is never moved back by Windows, so
    // the default 1440 x 810 on a smaller screen (high DPI scaling) left the settings panel off
    // the bottom right. 16:9 kept, at most 92% of the width and 88% of the height (the taskbar).
    function fitScreen() {
        const sw = Screen.width, sh = Screen.height
        if (sw <= 0 || sh <= 0) return
        const scale = Math.min(1, sw * 0.92 / width, sh * 0.88 / height)
        width = Math.max(minimumWidth, Math.floor(width * scale))
        height = Math.max(minimumHeight, Math.floor(height * scale))
        x = Screen.virtualX + Math.max(0, Math.floor((sw - width) / 2))
        y = Screen.virtualY + Math.max(0, Math.floor((sh * 0.96 - height) / 2))
    }

    // the last update's outcome, then one look for a newer release
    Component.onCompleted: { fitScreen(); updater.check() }

    WindowChrome {
        z: 100
        window: root
    }
}
