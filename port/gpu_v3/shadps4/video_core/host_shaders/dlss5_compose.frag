// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 450

// DLSS 5 (vk_dlss5.cpp): the passes around the Neural Rendering model, as a full-screen draw
// (fs_tri.vert) into the target - the graphics path, like the post-process pass (2026-10-18: the
// compute version, dlss5_compose.comp, wrote through bad descriptors when the native descriptor
// heaps were bound in the same command buffer - device lost at a never-bound address).
//   mode 0, encode game frame: the tonemap output (RGBA8 UNORM, sRGB-encoded values) holds red and
//           blue swapped (vk_dlss_upscale's composite swaps them back) - the model must see true
//           colour, or skin reads blue to it.
//   mode 1, encode video frame: the display buffer through the presenter's view; linear when that
//           view is sRGB (src_linear), so it is encoded again.
//   mode 2, compose for the game: the model's light onto the game's frame (a luminance ratio),
//           its colour mixed in by `colour`, written encoded and swapped back for DLSS 4.5.
//   mode 3, compose for video: the same, written linear for the post pass.
// At transfer 0 (and saturation 1) the frame comes back unchanged. The split compare shows the game's frame left of
// `split` (fraction of the width).


layout(location = 0) in vec2 uv; // fs_tri.vert's (unused: texels by gl_FragCoord)
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D src;      // encode: the game's / video frame
layout(set = 0, binding = 1) uniform sampler2D original; // compose: model input
layout(set = 0, binding = 2) uniform sampler2D model;    // compose: model answer

layout(push_constant) uniform Params {
    uint mode;
    uint src_linear;
    uint compare;
    float split;
    float transfer;
    float colour;
    float max_ratio;
    float saturation; // of DLSS 5's result: 1 = as composed
};

const vec3 Luma = vec3(0.2126, 0.7152, 0.0722);

vec3 SrgbToLinear(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), greaterThan(c, vec3(0.04045)));
}

vec3 LinearToSrgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, greaterThan(c, vec3(0.0031308)));
}

vec3 Cbrt(vec3 x) {
    return sign(x) * pow(abs(x), vec3(1.0 / 3.0));
}

// OkLab (Bjorn Ottosson's published matrices), from / to linear sRGB.
vec3 ToOkLab(vec3 c) {
    const vec3 lms = vec3(dot(c, vec3(0.4122214708, 0.5363325363, 0.0514459929)),
                          dot(c, vec3(0.2119034982, 0.6806995451, 0.1073969566)),
                          dot(c, vec3(0.0883024619, 0.2817188376, 0.6299787005)));
    const vec3 r = Cbrt(lms);
    return vec3(dot(r, vec3(0.2104542553, 0.7936177850, -0.0040720468)),
                dot(r, vec3(1.9779984951, -2.4285922050, 0.4505937099)),
                dot(r, vec3(0.0259040371, 0.7827717662, -0.8086757660)));
}

vec3 FromOkLab(vec3 lab) {
    const vec3 r = vec3(lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z,
                        lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z,
                        lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z);
    const vec3 lms = r * r * r;
    return vec3(dot(lms, vec3(4.0767416621, -3.3077115913, 0.2309699292)),
                dot(lms, vec3(-1.2684380046, 2.6097574011, -0.3413193965)),
                dot(lms, vec3(-0.0041960863, -0.7034186147, 1.7076147010)));
}

void main() {
    const ivec2 pos = ivec2(gl_FragCoord.xy);
    if (mode == 0u || mode == 1u) {
        const vec3 c = texelFetch(src, pos, 0).rgb;
        const vec3 encoded = mode == 0u ? c.bgr : (src_linear != 0u ? LinearToSrgb(c) : c);
        frag_color = vec4(encoded, 1.0);
        return;
    }
    const ivec2 size = textureSize(original, 0);

    // Both inputs hold sRGB-encoded true colour.
    const vec3 o = SrgbToLinear(texelFetch(original, pos, 0).rgb);
    const vec3 m = SrgbToLinear(texelFetch(model, pos, 0).rgb);
    vec3 result = o;
    const bool show_game = compare != 0u && float(pos.x) < split * float(size.x);
    if (!show_game) {
        // Light: the model's luminance as a ratio on the game's colour, bounded both ways.
        const float yo = dot(o, Luma);
        const float ym = dot(m, Luma);
        const float ratio = (ym + 1e-4) / (yo + 1e-4);
        const float r0 = max(1.0 + transfer * (ratio - 1.0), 0.0);
        const float r = max_ratio >= 7.99 ? r0 : clamp(r0, 1.0 / max_ratio, max_ratio); // 8 = off
        const vec3 lit = o * r;
        // Colour: the hue / chroma moves toward the model's by `colour` (scaled by transfer, so
        // transfer 0 is the game's frame exactly).
        const vec3 lab_lit = ToOkLab(lit);
        const vec3 lab_model = ToOkLab(m);
        const float w = colour * clamp(transfer, 0.0, 1.0);
        const vec2 ab = mix(lab_lit.yz, lab_model.yz, w) * saturation;
        result = max(FromOkLab(vec3(lab_lit.x, ab)), vec3(0.0));
        if (transfer <= 0.0) {
            result = o;
        }
    }
    if (compare != 0u && abs(float(pos.x) - split * float(size.x)) < 1.5) {
        result = vec3(1.0); // the split line
    }
    if (mode == 2u) {
        frag_color = vec4(LinearToSrgb(result).bgr, 1.0); // the game's channel order
    } else {
        frag_color = vec4(result, 1.0); // linear, for the post pass
    }
}
