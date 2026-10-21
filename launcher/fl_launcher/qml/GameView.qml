// SPDX-License-Identifier: GPL-2.0-or-later
// The game's page over its own key art: the tile, title and Play at the top left (the art carries
// the logo in the middle and the hero on the right), the settings panel at the bottom right.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Effects
import "theme.js" as T

Item {
    id: view
    property bool shown: false
    property Item backdrop: null      // what the panel blurs (the window-sized background)
    property real stageScale: 1
    signal browse

    visible: opacity > 0
    enabled: shown
    opacity: shown ? 1 : 0
    Behavior on opacity { NumberAnimation { duration: 550; easing.type: Easing.BezierSpline; easing.bezierCurve: T.ease } }

    function focusDefault() { play.forceActiveFocus() }
    function showLog() { page = 3 }

    // the home screen's intro: the tile and title, then Play, then the panel slides in
    property real topIn: 0
    property real heroIn: 0
    property real panelIn: 0
    ParallelAnimation {
        id: intro
        SequentialAnimation {
            PauseAnimation { duration: 250 }
            NumberAnimation { target: view; property: 'topIn'; from: 0; to: 1; duration: 450; easing.type: Easing.OutCubic }
        }
        SequentialAnimation {
            PauseAnimation { duration: 550 }
            NumberAnimation { target: view; property: 'heroIn'; from: 0; to: 1; duration: 600; easing.type: Easing.BezierSpline; easing.bezierCurve: T.ease }
        }
        SequentialAnimation {
            PauseAnimation { duration: 750 }
            NumberAnimation { target: view; property: 'panelIn'; from: 0; to: 1; duration: 650; easing.type: Easing.BezierSpline; easing.bezierCurve: T.ease }
        }
    }
    onShownChanged: if (shown) { intro.restart(); focusDefault() }
    Component.onCompleted: if (shown) { intro.restart(); focusDefault() }

    // ---- the game's tile, name and version; the clock ------------------------------------------
    Item {
        id: topbar
        anchors.fill: parent
        opacity: view.topIn
        transform: Translate { y: (1 - view.topIn) * -16 }

        RectangularShadow {
            anchors.fill: tile
            offset.y: 12
            blur: 34
            radius: 14
            color: Qt.rgba(0, 0, 0, 0.5)
        }
        Item {
            id: tile
            x: 96
            y: 64
            width: 150
            height: 150
            Rectangle {
                id: tileMask
                anchors.fill: parent
                radius: 14
                visible: false
                layer.enabled: true
            }
            Rectangle {
                anchors.fill: parent
                radius: 14
                color: Qt.rgba(30 / 255, 30 / 255, 34 / 255, 0.85)
            }
            Image {
                id: icon
                anchors.fill: parent
                source: launcher.iconUrl
                sourceSize: Qt.size(300, 300)
                fillMode: Image.PreserveAspectCrop
                asynchronous: true
                smooth: true
                mipmap: true
                visible: false
            }
            MultiEffect {
                anchors.fill: parent
                source: icon
                maskEnabled: true
                maskSource: tileMask
                visible: launcher.iconUrl !== ''
            }
        }
        Column {
            anchors.left: tile.right
            anchors.leftMargin: 34
            anchors.verticalCenter: tile.verticalCenter
            spacing: 14
            Text {
                text: launcher.title
                font.family: T.font
                font.pixelSize: 40
                font.weight: Font.Light
                color: T.fg
                layer.enabled: true
                layer.effect: MultiEffect {
                    shadowEnabled: true
                    shadowColor: 'black'
                    shadowOpacity: 0.6
                    shadowBlur: 0.8
                    blurMax: 20
                    shadowVerticalOffset: 3
                }
            }
            Row {
                spacing: 16
                Rectangle {
                    visible: launcher.titleId !== ''
                    width: badge.implicitWidth + 16
                    height: 26
                    radius: 4
                    color: 'white'
                    anchors.verticalCenter: parent.verticalCenter
                    Text {
                        id: badge
                        anchors.centerIn: parent
                        text: launcher.titleId
                        font.family: T.font
                        font.pixelSize: 16
                        font.weight: Font.DemiBold
                        font.letterSpacing: 1.3
                        color: 'black'
                    }
                }
                Text {
                    text: launcher.versionText
                    font.family: T.font
                    font.pixelSize: 26
                    font.weight: Font.Light
                    color: T.fg
                    style: Text.Raised
                    styleColor: Qt.rgba(0, 0, 0, 0.35)
                    anchors.verticalCenter: parent.verticalCenter
                }
            }
        }
        Text {
            id: clock
            anchors.right: parent.right
            anchors.rightMargin: 96
            y: 52
            font.family: T.font
            font.pixelSize: 40
            font.weight: Font.Light
            color: T.fg
            style: Text.Raised
            styleColor: Qt.rgba(0, 0, 0, 0.3)
            function update() { text = Qt.formatTime(new Date(), 'HH:mm') }
            Component.onCompleted: update()
            Timer { interval: 5000; running: true; repeat: true; onTriggered: clock.update() }
        }
    }

    // ---- Play ------------------------------------------------------------------------------------
    Column {
        id: hero
        x: 96
        y: tile.y + tile.height + 56
        width: Math.min(900, panel.x - x - 60)
        spacing: 0
        opacity: view.heroIn
        transform: Translate { y: (1 - view.heroIn) * 24 }

        Text {
            width: parent.width
            visible: text !== ''
            bottomPadding: 28
            text: launcher.untested
            font.family: T.font
            font.pixelSize: 24
            color: T.warn
            wrapMode: Text.Wrap
            style: Text.Raised
            styleColor: Qt.rgba(0, 0, 0, 0.5)
        }
        Row {
            spacing: 20
            PillButton {
                id: play
                text: launcher.running ? launcher.tr.stop : launcher.tr.play
                glyph: launcher.running ? 'stop' : 'play'
                minWidth: 380
                onClicked: launcher.play()
            }
            CircleButton {
                anchors.verticalCenter: play.verticalCenter
                glyph: 'folder'
                tip: launcher.tr.change_folder
                onClicked: view.browse()
            }
        }
        Item {
            width: parent.width
            height: 60
            Rectangle {
                visible: launcher.running
                y: 34
                width: 12
                height: 12
                radius: 6
                color: T.ok
                SequentialAnimation on opacity {
                    running: launcher.running
                    loops: Animation.Infinite
                    NumberAnimation { to: 0.35; duration: 1200; easing.type: Easing.InOutSine }
                    NumberAnimation { to: 1; duration: 1200; easing.type: Easing.InOutSine }
                }
            }
            Text {
                x: launcher.running ? 24 : 0
                width: parent.width - x
                y: 24
                text: launcher.status
                font.family: T.font
                font.pixelSize: 24
                font.weight: Font.Light
                color: launcher.running ? T.fg : T.fg2
                elide: Text.ElideRight
                style: Text.Raised
                styleColor: Qt.rgba(0, 0, 0, 0.5)
            }
        }
    }

    Column {
        id: credits
        x: 96
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 56
        width: Math.min(900, panel.x - x - 60)
        spacing: 6
        opacity: view.heroIn
        Text {
            width: parent.width
            text: launcher.tr.subtitle
            font.family: T.font
            font.pixelSize: 20
            color: T.fg2
            wrapMode: Text.Wrap
        }
        Text {
            width: parent.width
            text: launcher.tr.credits
            font.family: T.font
            font.pixelSize: 18
            color: T.fg3
            wrapMode: Text.Wrap
        }
    }

    // ---- settings panel: sections on the left, the section's settings on the right ---------------
    property int page: 0

    Item {
        id: panel
        width: 900
        height: 560
        x: view.width - width - 72
        y: view.height - height - 56
        opacity: view.panelIn
        transform: Translate { x: (1 - view.panelIn) * 60 }

        // frosted glass: the art behind the panel, blurred
        ShaderEffectSource {
            id: behind
            sourceItem: view.backdrop
            visible: false
            live: true
            sourceRect: Qt.rect(panel.x * view.stageScale, panel.y * view.stageScale,
                                panel.width * view.stageScale, panel.height * view.stageScale)
        }
        Rectangle {
            id: panelMask
            anchors.fill: parent
            radius: 18
            visible: false
            layer.enabled: true
        }
        RectangularShadow {
            anchors.fill: parent
            offset.y: 20
            blur: 60
            radius: 18
            color: Qt.rgba(0, 0, 0, 0.45)
        }
        MultiEffect {
            anchors.fill: parent
            source: behind
            autoPaddingEnabled: false
            blurEnabled: true
            blur: 1
            blurMax: 64
            saturation: -0.25
            maskEnabled: true
            maskSource: panelMask
        }
        Rectangle {
            anchors.fill: parent
            radius: 18
            color: Qt.rgba(18 / 255, 18 / 255, 22 / 255, 0.7)
            border.width: 1
            border.color: Qt.rgba(1, 1, 1, 0.07)
        }

        Column {
            id: sections
            x: 24
            y: 28
            width: 236
            spacing: 6
            Repeater {
                id: sectionRows
                model: [
                    { label: launcher.tr.display, glyph: 'display' },
                    { label: launcher.tr.language, glyph: 'globe' },
                    { label: launcher.tr.advanced, glyph: 'code' },
                    { label: launcher.tr.log, glyph: 'log' }
                ]
                delegate: SectionRow {
                    required property var modelData
                    required property int index
                    label: modelData.label
                    glyph: modelData.glyph
                    current: view.page === index
                    onSelected: view.page = index
                    onEnter: pages.enter()
                }
            }
        }
        Rectangle {
            x: sections.x + sections.width + 22
            y: 28
            width: 1
            height: parent.height - 56
            color: T.line
        }

        Item {
            id: pages
            x: sections.x + sections.width + 52
            y: 22
            width: parent.width - x - 40
            height: parent.height - y - 28

            function enter() { [displayFirst, languageFirst, advancedRow, logArea][view.page].forceActiveFocus() }
            Keys.onEscapePressed: sectionRows.itemAt(view.page).forceActiveFocus()

            // display
            Flickable {
                anchors.fill: parent
                visible: view.page === 0
                contentHeight: displayRows.height
                clip: true
                boundsBehavior: Flickable.StopAtBounds
                Column {
                    id: displayRows
                    width: parent.width
                    ChoiceRow {
                        id: displayFirst
                        label: launcher.tr.display_mode
                        options: launcher.displayModes
                        value: launcher.core.display_mode.toLowerCase()
                        onPicked: v => launcher.setCore('display_mode', v)
                    }
                    ChoiceRow {
                        label: launcher.tr.fps_limit
                        options: launcher.fpsLimits
                        value: launcher.core.fps_limit
                        onPicked: v => launcher.setCore('fps_limit', v)
                    }
                    Repeater {
                        model: ['vsync', 'fsr1', 'rcas', 'show_fps', 'async_shaders']
                        delegate: ToggleRow {
                            required property string modelData
                            required property int index
                            label: launcher.tr[modelData]
                            checked: launcher.core[modelData] === '1'
                            last: index === 4
                            onToggled: on => launcher.setCore(modelData, on ? '1' : '0')
                        }
                    }
                }
            }

            // languages
            Column {
                anchors.left: parent.left
                anchors.right: parent.right
                visible: view.page === 1
                ChoiceRow {
                    id: languageFirst
                    label: launcher.tr.game_language
                    options: launcher.gameLanguages
                    value: launcher.gameLanguage
                    onPicked: v => launcher.setGameLanguage(v)
                }
                ChoiceRow {
                    label: launcher.tr.ui_language
                    options: launcher.uiLanguages
                    value: launcher.uiLanguage
                    last: true
                    onPicked: v => launcher.setUiLanguage(v)
                }
            }

            // advanced: KEY=VALUE core variables
            Column {
                anchors.fill: parent
                visible: view.page === 2
                spacing: 0
                ToggleRow {
                    id: advancedRow
                    label: launcher.tr.advanced_on
                    description: launcher.tr.advanced_note
                    checked: launcher.advanced
                    last: true
                    onToggled: on => launcher.setAdvanced(on)
                }
                Item { width: 1; height: 14 }
                ScrollView {
                    width: parent.width
                    height: parent.height - advancedRow.height - 14
                    opacity: launcher.advanced ? 1 : 0.5
                    Behavior on opacity { NumberAnimation { duration: 200 } }
                    TextArea {
                        id: extraEnv
                        font.family: T.mono
                        font.pixelSize: 20
                        color: T.fg
                        placeholderText: launcher.tr.extra_env_hint
                        placeholderTextColor: T.fg3
                        selectionColor: T.blue
                        selectedTextColor: 'white'
                        wrapMode: TextEdit.Wrap
                        padding: 18
                        background: Rectangle {
                            radius: 10
                            color: Qt.rgba(0, 0, 0, 0.32)
                            border.width: extraEnv.activeFocus ? 2 : 1
                            border.color: extraEnv.activeFocus ? Qt.rgba(1, 1, 1, 0.85) : Qt.rgba(1, 1, 1, 0.1)
                        }
                        Component.onCompleted: text = launcher.extraEnv
                        onTextChanged: launcher.setExtraEnv(text)
                    }
                }
            }

            // log
            Item {
                anchors.fill: parent
                anchors.topMargin: 6
                visible: view.page === 3
                ScrollView {
                    id: logScroll
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.bottom: logButtons.top
                    anchors.bottomMargin: 18
                    property bool follow: true
                    function toEnd() {
                        const flick = contentItem
                        flick.contentY = Math.max(0, flick.contentHeight - flick.height)
                    }
                    TextArea {
                        id: logArea
                        readOnly: true
                        font.family: T.mono
                        font.pixelSize: 16
                        color: Qt.rgba(1, 1, 1, 0.8)
                        selectByMouse: true
                        selectionColor: T.blue
                        selectedTextColor: 'white'
                        wrapMode: TextEdit.WrapAnywhere
                        padding: 16
                        background: Rectangle {
                            radius: 10
                            color: Qt.rgba(0, 0, 0, 0.38)
                            border.width: logArea.activeFocus ? 2 : 1
                            border.color: logArea.activeFocus ? Qt.rgba(1, 1, 1, 0.85) : Qt.rgba(1, 1, 1, 0.1)
                        }
                        Component.onCompleted: launcher.attachLog(textDocument)
                        onTextChanged: if (logScroll.follow) Qt.callLater(logScroll.toEnd)
                    }
                    Connections {
                        target: logScroll.contentItem
                        function onMovementEnded() {
                            const flick = logScroll.contentItem
                            logScroll.follow = flick.contentY >= flick.contentHeight - flick.height - 40
                        }
                    }
                }
                Row {
                    id: logButtons
                    anchors.bottom: parent.bottom
                    spacing: 16
                    PillButton {
                        text: launcher.tr.open_logs
                        glyph: 'openFolder'
                        size: 58
                        fontSize: 23
                        minWidth: 250
                        onClicked: launcher.openLogs()
                    }
                    PillButton {
                        text: launcher.tr.open_saves
                        glyph: 'save'
                        size: 58
                        fontSize: 23
                        minWidth: 250
                        onClicked: launcher.openSaves()
                    }
                }
            }
        }
    }
}
