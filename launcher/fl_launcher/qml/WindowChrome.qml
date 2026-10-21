// SPDX-License-Identifier: GPL-2.0-or-later
// The frameless window's own chrome: a transparent strip on top that drags the window (double
// click maximizes or restores), minimize / maximize / close buttons on the right, and resize
// borders on the edges. It lies over the launcher's own background, in window pixels.
import QtQuick
import "theme.js" as T

Item {
    id: chrome
    required property var window
    anchors.fill: parent

    readonly property bool maximized: window.visibility === Window.Maximized
    readonly property bool fullScreen: window.visibility === Window.FullScreen
    readonly property int stripHeight: 36
    readonly property int border: 6

    function toggleMaximized() {
        if (maximized) window.showNormal(); else window.showMaximized()
    }

    // Segoe Fluent Icons chrome glyphs (the system font Windows' own title bars use)
    readonly property var glyphs: ({ minimize: '', maximize: '', restore: '', close: '' })

    // ---- drag strip ---------------------------------------------------------------------------
    MouseArea {
        id: strip
        visible: !chrome.fullScreen
        anchors { left: parent.left; right: buttons.left; top: parent.top }
        height: chrome.stripHeight
        acceptedButtons: Qt.LeftButton
        onPressed: chrome.window.startSystemMove()
        onDoubleClicked: chrome.toggleMaximized()
    }

    // ---- buttons ------------------------------------------------------------------------------
    Row {
        id: buttons
        visible: !chrome.fullScreen
        anchors { top: parent.top; right: parent.right }
        Repeater {
            model: [
                { key: 'minimize', action: () => chrome.window.showMinimized() },
                { key: 'maximize', action: () => chrome.toggleMaximized() },
                { key: 'close', action: () => chrome.window.close() }
            ]
            delegate: Rectangle {
                id: button
                required property var modelData
                readonly property bool isClose: modelData.key === 'close'
                width: 48
                height: chrome.stripHeight
                color: area.containsMouse ? (isClose ? '#c42b1c' : Qt.rgba(1, 1, 1, area.pressed ? 0.2 : 0.12))
                                          : 'transparent'
                Behavior on color { ColorAnimation { duration: 120 } }
                Text {
                    anchors.centerIn: parent
                    text: button.modelData.key === 'maximize' && chrome.maximized ? chrome.glyphs.restore
                                                                                   : chrome.glyphs[button.modelData.key]
                    font.family: launcher.iconFont !== '' ? launcher.iconFont : 'Segoe Fluent Icons'
                    font.pixelSize: 11
                    color: T.fg
                    opacity: area.containsMouse ? 1 : 0.85
                }
                MouseArea {
                    id: area
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: button.modelData.action()
                }
            }
        }
    }

    // ---- resize borders (not when maximized or full screen) -------------------------------------
    component Edge: MouseArea {
        required property int edges
        visible: !chrome.maximized && !chrome.fullScreen
        hoverEnabled: true
        onPressed: chrome.window.startSystemResize(edges)
    }
    Edge { edges: Qt.LeftEdge; cursorShape: Qt.SizeHorCursor
           anchors { left: parent.left; top: parent.top; bottom: parent.bottom } width: chrome.border }
    Edge { edges: Qt.RightEdge; cursorShape: Qt.SizeHorCursor
           anchors { right: parent.right; top: buttons.bottom; bottom: parent.bottom } width: chrome.border }
    Edge { edges: Qt.BottomEdge; cursorShape: Qt.SizeVerCursor
           anchors { left: parent.left; right: parent.right; bottom: parent.bottom } height: chrome.border }
    Edge { edges: Qt.TopEdge; cursorShape: Qt.SizeVerCursor
           anchors { left: parent.left; right: buttons.left; top: parent.top } height: 3 }
    Edge { edges: Qt.LeftEdge | Qt.BottomEdge; cursorShape: Qt.SizeBDiagCursor
           anchors { left: parent.left; bottom: parent.bottom } width: chrome.border * 2; height: chrome.border * 2 }
    Edge { edges: Qt.RightEdge | Qt.BottomEdge; cursorShape: Qt.SizeFDiagCursor
           anchors { right: parent.right; bottom: parent.bottom } width: chrome.border * 2; height: chrome.border * 2 }
}
