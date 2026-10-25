// tlg.ini graphics keys (tlg_settings.cpp, tlg_graphics.h).
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include "tlg_graphics.h"
#include "tlg_settings.h"

namespace G = TlgGraphics;

static std::string Read(const char* path) {
    std::ifstream in(path);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

// An empty value removes the variable, as _putenv_s does.
static void SetEnv(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (*value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

int main() {
#ifdef _WIN32
    _set_error_mode(_OUT_TO_STDERR); // a failed assert prints instead of opening a dialog
#endif
    assert(G::ParseDisplayMode("borderless") == G::Borderless && G::ParseDisplayMode("bogus") == G::Windowed);
    assert(G::ParseFpsLimit(240) == 240 && G::ParseFpsLimit(75) == 0);
    assert(G::ParseResolution("800p") == 3 && G::ParseResolution("1440p") == 4 && G::ParseResolution("8K") == 0);
    assert(G::ParseEngineFps(240) == 240 && G::ParseEngineFps(90) == 120);
    assert(G::ParseFramesQueued(2) == 2 && G::ParseFramesQueued(0) == 0 && G::ParseFramesQueued(5) == 1);
    assert(G::ParseRcasStrength(100) == 100 && G::ParseRcasStrength(101) == 60 && G::ParseRcasStrength(-1) == 60);
    assert(G::RcasAttenuation(100) == 0.0f && G::RcasAttenuation(0) == 2.0f && G::RcasAttenuation(75) == 0.5f);
    assert(G::FsrPassRuns(true, false, 1280, 720, 2560, 1440) && !G::FsrPassRuns(false, false, 1280, 720, 2560, 1440));
    assert(G::FsrPassRuns(false, true, 1920, 1080, 1920, 1080) && !G::FsrPassRuns(true, false, 1920, 1080, 1920, 1080));
    assert(!G::FsrPassRuns(true, true, 2560, 1440, 1920, 1080));  // larger image: regular scaling
    assert(G::EffectiveFps(0, 120) == 120 && G::EffectiveFps(240, 120) == 120 && G::EffectiveFps(60, 240) == 60);
    // An unpatched engine steps 1/60 s per frame: presenting faster speeds the game up.
    assert(G::PresentLimit(120, 60) == 60 && G::PresentLimit(240, 120) == 120 && G::PresentLimit(240, 240) == 240);
    assert(G::PresentLimit(0, 60) == 0 && G::PresentLimit(30, 60) == 30);  // 0: paced by vblank

    const char* path = "graphics-settings-test.ini";
    {
        std::ofstream out(path);
        out << "display_mode=borderless\nvsync=0\nfps_limit=240\nasync_shaders=0\n"
               "render_resolution=1440p\nengine_fps=240\ndeferred_readback=0\nstale_readback=1\n"
               "fsr1=1\nrcas=0\nrcas_strength=40\nframes_queued=2\n";
    }
    SetEnv("TLG_CONFIG", path);
    SetEnv("TLG_ASYNC_SHADERS", "");
    auto& v = TlgSettings::Get();
    TlgSettings::Load();
    assert(v.display_mode == G::Borderless && !v.vsync && v.fps_limit == 240 && !v.async_shaders);
    assert(v.render_resolution == 4 && v.engine_fps == 240 && !v.deferred_readback && v.stale_readback);
    assert(v.fsr1 && !v.rcas && v.rcas_strength == 40 && v.frames_queued == 2);
    assert(!TlgSettings::GraphicsNeedRestart());
    assert(v.applied_engine_fps == 240);  // no TLG_ENGINE_FPS: the configured rate
    SetEnv("TLG_ENGINE_FPS", "60");    // run.py: the frame rate patch did not apply
    TlgSettings::Load();
    assert(v.engine_fps == 240 && v.applied_engine_fps == 60 && !TlgSettings::GraphicsNeedRestart());
    SetEnv("TLG_ENGINE_FPS", "");
    v.engine_fps = 120;
    assert(TlgSettings::GraphicsNeedRestart());
    v.fps_limit = 60;  // live: no restart because of it
    v.engine_fps = 240;
    assert(!TlgSettings::GraphicsNeedRestart());

    TlgSettings::Save();
    const std::string saved = Read(path);
    for (const char* line : {"display_mode=borderless\n", "vsync=0\n", "fps_limit=60\n", "async_shaders=0\n",
                             "render_resolution=1440p\n", "engine_fps=240\n", "deferred_readback=0\n",
                             "stale_readback=1\n", "fsr1=1\n", "rcas=0\n", "rcas_strength=40\n",
                             "frames_queued=2\n"}) {
        assert(saved.find(line) != std::string::npos);
    }

    {
        std::ofstream out(path);
        out << "display_mode=huge\nfps_limit=7\nrender_resolution=8K\nengine_fps=1\n"
               "rcas_strength=250\nframes_queued=9\n";
    }
    SetEnv("TLG_ASYNC_SHADERS", "1");  // the environment still wins at start
    v.async_shaders = false;
    TlgSettings::Load();
    assert(v.display_mode == G::Windowed && v.fps_limit == 0 && v.render_resolution == 0 && v.engine_fps == 120);
    assert(v.rcas_strength == 60 && v.frames_queued == 1);
    assert(v.async_shaders);
    {
        std::ofstream out(path);
        out << "cheat_double_jumps=1\ncheat_very_hard=1\n";
    }
    TlgSettings::Load();
    TlgSettings::Save();
    const std::string cheats = Read(path);
    assert(cheats.find("cheat_double_jumps=1\n") != std::string::npos);
    assert(cheats.find("cheat_very_hard=1\n") != std::string::npos);
    {
        std::ofstream out(path);
        out << "bp06=1\nbp12=0\nwqm=0\nbc_tiny=1\nbp99=1\n";
    }
    TlgSettings::Load();
    assert(v.backport_pref[5] == 1 && v.backport_pref[11] == 0 && v.backport_pref[0] == -1);
    TlgSettings::Save();
    const std::string fixes = Read(path);
    for (const char* line : {"bp06=1\n", "bp12=0\n", "wqm=0\n", "bc_tiny=1\n"}) {
        assert(fixes.find(line) != std::string::npos);
    }
    assert(fixes.find("bp01=") == std::string::npos && fixes.find("bp99") == std::string::npos);
    std::remove(path);
    std::puts("PASS: graphics settings");
}
