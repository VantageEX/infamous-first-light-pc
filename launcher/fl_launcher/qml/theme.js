// SPDX-License-Identifier: GPL-2.0-or-later
// Colours, fonts and timings of the console-style interface (all sizes are on the 1920x1080 stage).
.pragma library

// fonts.py substitutes Segoe UI for this (Windows) or the bundled Open Sans (Linux)
var font = 'FL Sans'
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
    info: '', report: '', done: '', chevronDown: '', check: '', document: '', download: ''
}

// The same icons in Fluent UI System Icons (24 px regular; fonts.py loads the subset in port/fonts).
var iconFluent = {
    play: '\uF606', stop: '\uF72B', folder: '\uF419', openFolder: '\uF42F', save: '\uF680', log: '\uE558', display: '\uF35A', globe: '\uF45B', code: '\uF2F0', warning: '\uF86A', info: '\uF4A4', report: '\uF403', done: '\uF299', chevronDown: '\uF2A4', check: '\uF295', document: '\uF379', download: '\uF151'
}
