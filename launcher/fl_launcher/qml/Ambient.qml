// SPDX-License-Identifier: GPL-2.0-or-later
// The console's default theme: a graphite room lit from the top left, a misty floor and slow dust.
import QtQuick
import QtQuick.Effects
import QtQuick.Particles

Item {
    id: ambient
    clip: true

    Item {
        id: room
        width: 1920
        height: 1080
        transformOrigin: Item.TopLeft
        scale: Math.max(ambient.width / 1920, ambient.height / 1080)

        Canvas {
            anchors.fill: parent
            onPaint: {
                const ctx = getContext('2d')
                const w = width, h = height
                // linear-gradient(104deg, #36323a 0%, #2a292f 32%, #1b1e25 62%, #0e1119 100%)
                const a = 104 * Math.PI / 180, dx = Math.sin(a), dy = -Math.cos(a)
                const half = (Math.abs(w * dx) + Math.abs(h * dy)) / 2
                const base = ctx.createLinearGradient(w / 2 - dx * half, h / 2 - dy * half, w / 2 + dx * half, h / 2 + dy * half)
                base.addColorStop(0, '#36323a')
                base.addColorStop(0.32, '#2a292f')
                base.addColorStop(0.62, '#1b1e25')
                base.addColorStop(1, '#0e1119')
                ctx.fillStyle = base
                ctx.fillRect(0, 0, w, h)
                function ellipse(cx, cy, rx, ry, stops) {
                    ctx.save()
                    ctx.translate(cx, cy)
                    ctx.scale(rx / ry, 1)
                    const g = ctx.createRadialGradient(0, 0, 0, 0, 0, ry)
                    for (const s of stops) g.addColorStop(s[0], s[1])
                    ctx.fillStyle = g
                    ctx.fillRect(-w * ry / rx - ry, -h - ry, 2 * (w * ry / rx + ry), 2 * (h + ry))
                    ctx.restore()
                }
                ellipse(420, 930, 1100, 230, [[0, 'rgba(120,146,150,0.3)'], [0.72, 'rgba(0,0,0,0)']])
                ellipse(330, -120, 900, 1300, [[0, 'rgba(120,104,108,0.32)'], [0.7, 'rgba(0,0,0,0)']])
                ellipse(290, -40, 330, 760, [[0, 'rgba(226,206,198,0.46)'], [0.52, 'rgba(150,130,128,0.16)'], [0.78, 'rgba(0,0,0,0)']])
            }
        }

        // the wide slanted cone falling from the light
        Item {
            id: cone
            x: 120
            y: -260
            width: 620
            height: 1700
            rotation: -34
            transformOrigin: Item.Top
            Rectangle {
                id: coneFill
                anchors.fill: parent
                visible: false
                gradient: Gradient {
                    orientation: Gradient.Horizontal
                    GradientStop { position: 0; color: Qt.rgba(1, 0.94, 0.9, 0) }
                    GradientStop { position: 0.45; color: Qt.rgba(1, 0.93, 0.89, 0.075) }
                    GradientStop { position: 0.8; color: Qt.rgba(1, 0.94, 0.9, 0.02) }
                    GradientStop { position: 1; color: Qt.rgba(1, 0.94, 0.9, 0) }
                }
            }
            MultiEffect {
                anchors.fill: parent
                source: coneFill
                blurEnabled: true
                blur: 1
                blurMax: 32
            }
        }

        // gold dust drifting up through the light
        ParticleSystem { id: dust }
        Emitter {
            system: dust
            anchors.fill: parent
            emitRate: 9
            lifeSpan: 14000
            lifeSpanVariation: 5000
            size: 5
            sizeVariation: 4
            startTime: 14000
            velocity: AngleDirection { angle: 285; angleVariation: 50; magnitude: 10; magnitudeVariation: 7 }
        }
        Wander {
            system: dust
            anchors.fill: parent
            xVariance: 30
            pace: 20
        }
        ImageParticle {
            system: dust
            source: 'qrc:///particleresources/glowdot.png'
            color: '#ffe6b4'
            colorVariation: 0.08
            alpha: 0.45
            alphaVariation: 0.35
            entryEffect: ImageParticle.Fade
        }
    }
}
