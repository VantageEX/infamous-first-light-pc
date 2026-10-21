// SPDX-License-Identifier: GPL-2.0-or-later
// The console's pop-over list for a setting's values, anchored under the line it belongs to.
import QtQuick
import QtQuick.Effects
import "theme.js" as T

Item {
    id: picker
    property var options: []
    property string current
    property var callback: null
    property Item anchorItem: null
    readonly property bool isOpen: state === 'open'

    anchors.fill: parent
    visible: opacity > 0
    opacity: isOpen ? 1 : 0
    Behavior on opacity { NumberAnimation { duration: 200 } }

    function open(anchor, opts, value, cb) {
        anchorItem = anchor
        options = opts
        current = value
        callback = cb
        let index = 0
        for (let i = 0; i < opts.length; i++)
            if (opts[i].value === value) index = i
        const p = anchor.mapToItem(picker, anchor.width, anchor.height)
        const top = anchor.mapToItem(picker, 0, 0)
        box.x = Math.max(24, Math.min(p.x - box.width + 20, picker.width - box.width - 24))
        const below = p.y + 8
        box.y = below + box.height < picker.height - 24 ? below : Math.max(24, top.y - box.height - 8)
        state = 'open'
        list.currentIndex = index
        list.positionViewAtIndex(index, ListView.Center)
        list.forceActiveFocus()
    }
    function close(value) {
        state = ''
        if (anchorItem) anchorItem.forceActiveFocus()
        if (value !== undefined && callback) callback(value)
    }

    MouseArea {
        anchors.fill: parent
        enabled: picker.isOpen
        onClicked: picker.close()
        onWheel: wheel => wheel.accepted = true
    }
    RectangularShadow {
        anchors.fill: box
        offset.y: 20
        blur: 60
        radius: box.radius
        color: Qt.rgba(0, 0, 0, 0.6)
    }
    Rectangle {
        id: box
        width: 480
        height: Math.min(list.contentHeight + 24, 640)
        radius: 6
        color: T.popover
        transform: Translate { y: picker.isOpen ? 0 : 10; Behavior on y { NumberAnimation { duration: 250; easing.type: Easing.OutCubic } } }

        ListView {
            id: list
            anchors.fill: parent
            anchors.margins: 12
            clip: true
            model: picker.options
            boundsBehavior: Flickable.StopAtBounds
            highlightMoveDuration: 0
            delegate: Item {
                id: item
                required property var modelData
                required property int index
                readonly property bool focused: ListView.isCurrentItem && list.activeFocus
                width: list.width
                height: 60
                Rectangle {
                    anchors.fill: parent
                    radius: 4
                    color: Qt.rgba(1, 1, 1, item.focused ? 0.12 : hover.containsMouse ? 0.06 : 0)
                    FocusRing { radius: 4; gap: 3; thickness: 2; shown: item.focused }
                }
                Text {
                    anchors.left: parent.left
                    anchors.leftMargin: 20
                    anchors.right: tick.left
                    anchors.verticalCenter: parent.verticalCenter
                    text: item.modelData.text
                    elide: Text.ElideRight
                    font.family: T.font
                    font.pixelSize: 26
                    font.weight: Font.Light
                    color: T.fg
                }
                Glyph {
                    id: tick
                    anchors.right: parent.right
                    anchors.rightMargin: 20
                    anchors.verticalCenter: parent.verticalCenter
                    name: 'check'
                    size: 20
                    opacity: item.modelData.value === picker.current ? 1 : 0
                }
                MouseArea {
                    id: hover
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: picker.close(item.modelData.value)
                }
            }
            Keys.onReturnPressed: picker.close(picker.options[currentIndex].value)
            Keys.onEnterPressed: picker.close(picker.options[currentIndex].value)
            Keys.onSpacePressed: picker.close(picker.options[currentIndex].value)
            Keys.onEscapePressed: picker.close()
            Keys.onBackPressed: picker.close()
            Keys.onLeftPressed: picker.close()
            Keys.onTabPressed: picker.close()
        }
    }
}
