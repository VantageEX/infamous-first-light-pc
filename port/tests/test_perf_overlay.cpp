// Performance overlay text and settings (tlg_perf_stats.h, tlg_settings.cpp).
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include "tlg_perf_stats.h"
#include "tlg_settings.h"

namespace P = TlgPerfStats;

static std::string Read(const char* path) {
    std::ifstream in(path);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

int main() {
    assert(P::CpuPercent(50, 100, 1) == 50.0f);
    assert(P::CpuPercent(400, 100, 8) == 50.0f);
    assert(P::CpuPercent(5000, 100, 4) == 100.0f);
    assert(P::CpuPercent(10, 0, 4) < 0.0f && P::CpuPercent(10, 100, 0) < 0.0f);

    P::Sample s;
    s.fps = 238.4f;
    s.frame_ms = 4.19f;
    s.upscaler = "FSR 3.1";
    s.gpu_name = "RTX 4070";
    s.gpu_percent = 87.2f;
    s.cpu_percent = 34.0f;
    s.ram_used_gb = 6.12;
    s.ram_total_gb = 31.9;
    s.vram_used_gb = 5.4;
    s.vram_total_gb = 12.0;
    assert(P::Format(s, P::DefaultItems, true) == "238 FPS   4.2 ms   RTX 4070 87%   VRAM 5.4 / 12.0 GB");
    assert(P::Format(s, P::AllItems, false) ==
           "238 FPS\n4.2 ms\nFSR 3.1\nRTX 4070 87%\nCPU 34%\nRAM 6.1 / 31.9 GB\nVRAM 5.4 / 12.0 GB");
    assert(P::Format(s, P::Cpu | P::Vram, true) == "CPU 34%   VRAM 5.4 / 12.0 GB");
    assert(P::Format(s, 0, true).empty());

    P::Sample unknown;
    unknown.upscaler = "";
    assert(P::Format(unknown, P::Upscaler | P::Gpu | P::Cpu | P::Ram | P::Vram, false) ==
           "Native\nGPU n/a\nCPU n/a\nRAM n/a\nVRAM n/a");
    unknown.ram_used_gb = 2.0;
    assert(P::Format(unknown, P::Ram, true) == "RAM 2.0 GB");

    const char* path = "perf-overlay-test.ini";
    {
        std::ofstream out(path);
        out << "show_fps=1\noverlay_fps=0\noverlay_frametime=1\noverlay_upscaler=0\noverlay_gpu=1\n"
               "overlay_cpu=1\noverlay_ram=0\noverlay_vram=1\noverlay_graph=0\noverlay_scale=400\noverlay_corner=2\n"
               "overlay_opacity=-5\noverlay_layout=1\nmenu_tab=3\noverlay_style=4\noverlay_offset_x=2000\n"
               "overlay_offset_y=12\n";
    }
#ifdef _WIN32
    _putenv_s("TLG_CONFIG", path);
#else
    setenv("TLG_CONFIG", path, 1);
#endif
    auto& v = TlgSettings::Get();
    assert(v.overlay_items == P::DefaultItems && v.overlay_scale == 100 && v.overlay_corner == 1);
    TlgSettings::Load();
    assert(v.overlay_items == (P::FrameTime | P::Gpu | P::Cpu | P::Vram));
    assert(v.overlay_scale == 300 && v.overlay_corner == 2 && v.overlay_opacity == 0);
    assert(v.overlay_layout == 1 && v.menu_tab == 3);
    assert(v.overlay_style == 4 && v.overlay_offset_x == 960 && v.overlay_offset_y == 12);
    v.overlay_items = P::Fps | P::Ram;
    v.overlay_scale = 150;
    v.overlay_opacity = 70;
    v.menu_tab = 1;
    TlgSettings::Save();
    const std::string saved = Read(path);
    for (const char* line : {"overlay_fps=1\n", "overlay_frametime=0\n", "overlay_ram=1\n", "overlay_vram=0\n",
                             "overlay_scale=150\n", "overlay_corner=2\n", "overlay_opacity=70\n",
                             "overlay_layout=1\n", "menu_tab=1\n", "overlay_style=4\n", "overlay_offset_x=960\n",
                             "overlay_offset_y=12\n"}) {
        assert(saved.find(line) != std::string::npos);
    }
    {
        std::ofstream out(path);
        out << "overlay_corner=9\nmenu_tab=-1\noverlay_scale=10\noverlay_style=7\n";
    }
    TlgSettings::Load();
    assert(v.overlay_corner == 1 && v.menu_tab == 0 && v.overlay_scale == 50);
    assert(v.overlay_style == TlgSettings::OverlayHistogram);
    std::remove(path);
    std::puts("PASS: performance overlay");
}
