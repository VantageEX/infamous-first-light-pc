// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 450

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 color;

layout (binding = 0) uniform sampler2D texSampler;

// hdr: 0 = SDR frame on an SDR screen, 1 = HDR screen (passed through), 2 = the game's HDR10 (PQ)
// frame on an SDR screen (1.1.1, GitHub #13: PQ decoded, mapped to the game's own SDR look),
// 3 = an SDR frame on an HDR screen (1.1.1: encoded to PQ, SDR white at sdr_white nits; before,
// its SDR values were taken as PQ codes - a near-black picture).
layout (push_constant) uniform settings {
    float gamma;
    uint hdr;
    float sdr_white;
} pp;

const float cutoff = 0.0031308, a = 1.055, b = 0.055, d = 12.92;
vec3 gamma(vec3 rgb) {
    return mix(
        a * pow(rgb, vec3(1.0 / (2.4 + 1.0 - pp.gamma))) - b,
        d * rgb / pp.gamma,
        lessThan(rgb, vec3(cutoff))
    );
}

// ST.2084 inverse EOTF: PQ code value -> nits.
vec3 pq_to_nits(vec3 e) {
    const float m1 = 2610.0 / 16384.0, m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0, c2 = 2413.0 / 128.0, c3 = 2392.0 / 128.0;
    vec3 vp = pow(clamp(e, 0.0, 1.0), vec3(1.0 / m2));
    vec3 num = max(vp - c1, vec3(0.0));
    vec3 den = max(c2 - c3 * vp, vec3(1e-6));
    return 10000.0 * pow(num / den, vec3(1.0 / m1));
}

vec3 pq_to_sdr_linear(vec3 e) {
    vec3 n2020 = pq_to_nits(e);
    // BT.2020 -> BT.709 primaries (rows), as in vk_presenter.cpp's screenshot conversion.
    vec3 n709 = vec3(dot(vec3(1.6605, -0.5876, -0.0728), n2020),
                     dot(vec3(-0.1246, 1.1329, -0.0083), n2020),
                     dot(vec3(-0.0182, -0.1006, 1.1187), n2020));
    // Fitted to the game's own SDR picture of the same frame (s1570 SDR vs s1572 HDR10, city,
    // 34 k pixels, SDR 0.02-0.5): SDR ~ 0.0323 * nits^0.63 (2 nits -> 0.05, 9.4 -> 0.13, 35 ->
    // 0.30; white ~233 nits), with a soft shoulder above 0.8 instead of a hard clip.
    vec3 y = 0.0323 * pow(max(n709, vec3(0.0)), vec3(0.63));
    vec3 over = max(y - 0.8, vec3(0.0));
    return min(y, vec3(0.8)) + 0.2 * (1.0 - exp(-over / 0.2));
}

// ST.2084 EOTF inverse: nits -> PQ code value.
vec3 nits_to_pq(vec3 nits) {
    const float m1 = 2610.0 / 16384.0, m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0, c2 = 2413.0 / 128.0, c3 = 2392.0 / 128.0;
    vec3 y = pow(clamp(nits / 10000.0, 0.0, 1.0), vec3(m1));
    return pow((c1 + c2 * y) / (1.0 + c3 * y), vec3(m2));
}

// An SDR frame as an SDR screen would show it (the gamma setting included), back to display
// light by the sRGB curve (as Windows shows SDR content in HDR mode), BT.709 -> BT.2020, PQ.
vec3 sdr_to_pq(vec3 linear) {
    vec3 e = clamp(gamma(linear), 0.0, 1.0);
    vec3 l709 = mix(pow((e + 0.055) / 1.055, vec3(2.4)), e / 12.92, lessThanEqual(e, vec3(0.04045)));
    vec3 l2020 = vec3(dot(vec3(0.6274, 0.3293, 0.0433), l709),
                      dot(vec3(0.0691, 0.9195, 0.0114), l709),
                      dot(vec3(0.0164, 0.0880, 0.8956), l709));
    return nits_to_pq(l2020 * pp.sdr_white);
}

void main() {
    vec4 color_linear = texture(texSampler, uv);
    if (pp.hdr == 1u) {
        color = color_linear;
    } else if (pp.hdr == 2u) {
        color = vec4(gamma(pq_to_sdr_linear(color_linear.rgb)), color_linear.a);
    } else if (pp.hdr == 3u) {
        color = vec4(sdr_to_pq(color_linear.rgb), color_linear.a);
    } else {
        color = vec4(gamma(color_linear.rgb), color_linear.a);
    }
}
