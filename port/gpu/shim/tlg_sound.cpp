// SPDX-License-Identifier: GPL-2.0-or-later
#include "tlg_sound.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <SDL3/SDL.h>

extern "C" {
#include <libavcodec/avcodec.h>
}

namespace TlgSound {

namespace {

struct Clip {
    int rate = 48000, channels = 2;
    std::vector<float> samples; // interleaved
};

std::mutex mutex;
SDL_AudioStream* stream = nullptr;
bool tried_stream = false;

std::vector<unsigned char> ReadFile(const std::string& path) {
    std::vector<unsigned char> data;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return data;
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size > 0 && size < (32 << 20)) {
        data.resize(size_t(size));
        if (std::fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
    }
    std::fclose(f);
    return data;
}

// An MP3 file: FFmpeg's parser cuts the frames, its decoder gives float samples (planar or not).
bool DecodeMp3(const std::vector<unsigned char>& data, Clip& out) {
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_MP3);
    AVCodecParserContext* parser = codec ? av_parser_init(codec->id) : nullptr;
    AVCodecContext* context = parser ? avcodec_alloc_context3(codec) : nullptr;
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    bool ok = context && packet && frame && avcodec_open2(context, codec, nullptr) == 0;
    const auto drain = [&] {
        while (avcodec_receive_frame(context, frame) == 0) {
            const int channels = frame->ch_layout.nb_channels;
            if (channels < 1 || channels > 2) continue;
            out.rate = frame->sample_rate;
            out.channels = channels;
            for (int i = 0; i < frame->nb_samples; ++i) {
                for (int c = 0; c < channels; ++c) {
                    float v = 0.0f;
                    switch (frame->format) {
                    case AV_SAMPLE_FMT_FLTP: v = reinterpret_cast<const float*>(frame->extended_data[c])[i]; break;
                    case AV_SAMPLE_FMT_FLT: v = reinterpret_cast<const float*>(frame->extended_data[0])[i * channels + c]; break;
                    case AV_SAMPLE_FMT_S16P: v = reinterpret_cast<const int16_t*>(frame->extended_data[c])[i] / 32768.0f; break;
                    case AV_SAMPLE_FMT_S16: v = reinterpret_cast<const int16_t*>(frame->extended_data[0])[i * channels + c] / 32768.0f; break;
                    default: break;
                    }
                    out.samples.push_back(v);
                }
            }
        }
    };
    for (size_t at = 0; ok && at < data.size();) {
        const int used = av_parser_parse2(parser, context, &packet->data, &packet->size, data.data() + at,
                                          int(data.size() - at), AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
        if (used < 0) break;
        at += size_t(used);
        if (packet->size && avcodec_send_packet(context, packet) == 0) drain();
    }
    if (ok) {
        avcodec_send_packet(context, nullptr);
        drain();
    }
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&context);
    if (parser) av_parser_close(parser);
    return !out.samples.empty();
}

// A 16-bit PCM WAV file.
bool DecodeWav(const std::vector<unsigned char>& data, Clip& out) {
    if (data.size() < 44 || std::memcmp(data.data(), "RIFF", 4) || std::memcmp(data.data() + 8, "WAVE", 4)) return false;
    int channels = 0, rate = 0, bits = 0;
    for (size_t at = 12; at + 8 <= data.size();) {
        uint32_t size;
        std::memcpy(&size, data.data() + at + 4, 4);
        const unsigned char* body = data.data() + at + 8;
        if (size > data.size() - at - 8) break;
        if (!std::memcmp(data.data() + at, "fmt ", 4) && size >= 16) {
            uint16_t format, ch, bps;
            uint32_t sr;
            std::memcpy(&format, body, 2);
            std::memcpy(&ch, body + 2, 2);
            std::memcpy(&sr, body + 4, 4);
            std::memcpy(&bps, body + 14, 2);
            if (format != 1) return false;
            channels = ch;
            rate = int(sr);
            bits = bps;
        } else if (!std::memcmp(data.data() + at, "data", 4) && bits == 16 && (channels == 1 || channels == 2)) {
            out.rate = rate;
            out.channels = channels;
            for (uint32_t i = 0; i + 1 < size; i += 2) {
                int16_t v;
                std::memcpy(&v, body + i, 2);
                out.samples.push_back(v / 32768.0f);
            }
            return !out.samples.empty();
        }
        at += 8 + size + (size & 1);
    }
    return false;
}

// The built-in chime: soft bell notes (a sine and its octave, fading out); two rising notes for a
// trophy, a four-note arpeggio with a longer tail for the platinum one.
Clip Chime(bool platinum) {
    Clip clip;
    clip.rate = 48000;
    clip.channels = 2;
    struct Note {
        float start, hz, length, gain;
    };
    const std::vector<Note> notes = platinum
        ? std::vector<Note>{{0.00f, 1046.5f, 1.4f, 0.5f}, {0.09f, 1318.5f, 1.4f, 0.45f},
                            {0.18f, 1568.0f, 1.5f, 0.45f}, {0.30f, 2093.0f, 1.9f, 0.5f}}
        : std::vector<Note>{{0.00f, 1318.5f, 0.9f, 0.5f}, {0.11f, 1975.5f, 1.1f, 0.45f}};
    float total = 0.0f;
    for (const Note& n : notes) total = std::max(total, n.start + n.length);
    const size_t frames = size_t(total * float(clip.rate));
    clip.samples.assign(frames * 2, 0.0f);
    constexpr float tau = 6.2831853f;
    for (const Note& n : notes) {
        const size_t first = size_t(n.start * float(clip.rate));
        for (size_t i = first; i < frames; ++i) {
            const float t = float(i - first) / float(clip.rate);
            if (t > n.length) break;
            const float attack = std::min(t / 0.004f, 1.0f);
            const float envelope = attack * std::exp(-t * 5.0f / n.length);
            const float v = n.gain * envelope * (std::sin(tau * n.hz * t) + 0.3f * std::sin(tau * n.hz * 2.0f * t));
            clip.samples[i * 2] += v * 0.18f;
            clip.samples[i * 2 + 1] += v * 0.18f;
        }
    }
    return clip;
}

// This sound's file: the player's (<data>/sounds or TLG_SOUND_DIR), else the port's own
// (port/sounds beside the core's out/, tools/make_trophy_sounds.py), else the chime made here.
Clip Load(bool platinum) {
    std::vector<std::string> dirs;
    if (const char* env = std::getenv("TLG_SOUND_DIR"); env && *env) {
        dirs.push_back(env);
    } else if (const char* data = std::getenv("TLG_DATA_DIR"); data && *data) {
        dirs.push_back(std::string(data) + "/sounds");
    }
    if (const char* base = SDL_GetBasePath()) dirs.push_back(std::string(base) + "../sounds");
    const char* name = platinum ? "trophy-platinum" : "trophy";
    for (const std::string& dir : dirs) {
        for (const char* ext : {".mp3", ".wav"}) {
            const std::string path = dir + "/" + name + ext;
            const std::vector<unsigned char> data = ReadFile(path);
            if (data.empty()) continue;
            Clip clip;
            if (ext[1] == 'm' ? DecodeMp3(data, clip) : DecodeWav(data, clip)) {
                std::printf("Sound: %s\n", path.c_str());
                return clip;
            }
            std::printf("Sound: %s could not be decoded\n", path.c_str());
        }
    }
    return Chime(platinum);
}

} // namespace

void PlayTrophy(bool platinum) {
    static Clip clips[2];
    static bool loaded[2] = {};
    std::scoped_lock lock{mutex};
    Clip& clip = clips[platinum];
    if (!loaded[platinum]) {
        clip = Load(platinum);
        loaded[platinum] = true;
    }
    if (clip.samples.empty()) return;
    if (!tried_stream) {
        tried_stream = true;
        if (SDL_InitSubSystem(SDL_INIT_AUDIO)) {
            const SDL_AudioSpec spec{SDL_AUDIO_F32, 2, 48000};
            stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
        }
        if (!stream) std::printf("Sound: no audio stream (%s)\n", SDL_GetError());
    }
    if (!stream) return;
    const SDL_AudioSpec spec{SDL_AUDIO_F32, clip.channels, clip.rate};
    SDL_ClearAudioStream(stream);
    SDL_SetAudioStreamFormat(stream, &spec, nullptr);
    SDL_PutAudioStreamData(stream, clip.samples.data(), int(clip.samples.size() * sizeof(float)));
    SDL_ResumeAudioStreamDevice(stream);
}

} // namespace TlgSound
