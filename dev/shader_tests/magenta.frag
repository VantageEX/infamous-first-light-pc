#version 450
// Test replacement for a particle pixel shader: every fragment is bright HDR magenta, so the
// particles show where and how big they are drawn, whatever their colour and fade compute.
layout(location = 0) out vec4 color;
void main() {
    color = vec4(8.0, 0.4, 4.0, 1.0);
}
