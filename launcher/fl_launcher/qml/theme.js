// SPDX-License-Identifier: GPL-2.0-or-later
// Colours, fonts and timings of the console-style interface (all sizes are on the 1920x1080 stage).
.pragma library

var font = 'Segoe UI'
var mono = 'Consolas'
var fg = '#ffffff'
var fg2 = Qt.rgba(1, 1, 1, 0.64)
var fg3 = Qt.rgba(1, 1, 1, 0.38)
var blue = '#1f8fff'
var panel = Qt.rgba(28 / 255, 28 / 255, 30 / 255, 0.78)
var popover = Qt.rgba(36 / 255, 36 / 255, 40 / 255, 0.97)
var line = Qt.rgba(1, 1, 1, 0.1)
var ok = '#2ad16a'
var warn = '#ffc35a'
var bad = '#ff6b5a'

// cubic-bezier(.2, .8, .2, 1) and (.65, 0, .35, 1) of the reference
var ease = [0.2, 0.8, 0.2, 1, 1, 1]
var easeIO = [0.65, 0, 0.35, 1, 1, 1]

// Segoe Fluent Icons / MDL2 glyphs
var icon = {
    play: '', stop: '', folder: '', openFolder: '', save: '',
    log: '', display: '', globe: '', code: '', warning: '',
    info: '', report: '', done: '', chevronDown: '', check: '', document: ''
}
