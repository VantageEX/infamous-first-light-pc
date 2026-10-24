// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/release_mode.h"

#ifdef _WIN32

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <windows.h>
#include <shobjidl.h>

#pragma comment(lib, "ole32.lib")

// The reader below must run before every emulator static initializer (they are in the default
// "user" segment); "lib" runs first. C4073 is the compiler's note that this is on purpose.
#pragma warning(disable : 4073)
#pragma init_seg(lib)

namespace Common::ReleaseMode {

namespace {

constexpr const wchar_t* FlagsFileName = L"shadps4-flags.txt";
constexpr const wchar_t* GameFileName = L"shadps4-game.txt";
/// The player's own flags (2026-10-19): never shipped, so an update keeps them; read before
/// shadps4-flags.txt and wins over it.
constexpr const wchar_t* UserFlagsFileName = L"shadps4-flags-user.txt";

std::wstring ExeDir() {
    std::wstring buf(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
        if (n == 0) {
            return {};
        }
        if (n < buf.size()) {
            buf.resize(n);
            break;
        }
        buf.resize(buf.size() * 2);
    }
    const size_t slash = buf.find_last_of(L"\\/");
    return slash == std::wstring::npos ? std::wstring{} : buf.substr(0, slash);
}

char* Trim(char* s) {
    while (*s == ' ' || *s == '\t') {
        ++s;
    }
    size_t len = std::strlen(s);
    while (len && (s[len - 1] == '\n' || s[len - 1] == '\r' || s[len - 1] == ' ' ||
                   s[len - 1] == '\t')) {
        s[--len] = '\0';
    }
    return s;
}

struct FlagsFile {
    bool active = false;
    int applied = 0;
    int overridden = 0;
    int user_applied = 0;
    int defaults_copied = 0;
    std::wstring dir;

    FlagsFile() {
        dir = ExeDir();
        if (dir.empty()) {
            return;
        }
        const std::wstring path = dir + L"\\" + FlagsFileName;
        FILE* f = nullptr;
        if (_wfopen_s(&f, path.c_str(), L"rb") != 0 || !f) {
            return;
        }
        active = true;
        // Portable user\ folder next to the EXE, whatever "Start in" a shortcut gives
        // (path_util.cpp picks current_path()/user during static initialization).
        SetCurrentDirectoryW(dir.c_str());
        CopyMissingDefaults();
        // The player's own file first: a variable it sets is then "already set" for the release
        // file (the environment of a launcher still wins over both).
        FILE* uf = nullptr;
        if (_wfopen_s(&uf, (dir + L"\\" + UserFlagsFileName).c_str(), L"rb") == 0 && uf) {
            int skipped = 0;
            Apply(uf, user_applied, skipped);
            std::fclose(uf);
        }
        Apply(f, applied, overridden);
        std::fclose(f);
    }

    /// defaults\ (2026-10-19): the starting files a release ships (user\config.json, the per-game
    /// configs, the key layout), each copied to the same path under the EXE's folder only when that
    /// file does not exist yet - an update unzipped over an old folder keeps the player's settings.
    void CopyMissingDefaults() {
        namespace fs = std::filesystem;
        const fs::path root{dir};
        const fs::path defaults = root / L"defaults";
        std::error_code ec;
        if (!fs::is_directory(defaults, ec)) {
            return;
        }
        for (auto it = fs::recursive_directory_iterator(defaults, ec);
             !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
            std::error_code fec;
            if (!it->is_regular_file(fec)) {
                continue;
            }
            const fs::path rel = fs::relative(it->path(), defaults, fec);
            if (fec || rel.empty()) {
                continue;
            }
            const fs::path target = root / rel;
            if (fs::exists(target, fec)) {
                continue;
            }
            fs::create_directories(target.parent_path(), fec);
            if (fs::copy_file(it->path(), target, fs::copy_options::skip_existing, fec)) {
                ++defaults_copied;
            }
        }
    }

    /// NAME=VALUE lines; a name the environment already holds is skipped (counted in `skipped`).
    static void Apply(FILE* f, int& set, int& skipped) {
        char line[4096];
        bool first = true;
        while (std::fgets(line, sizeof(line), f)) {
            char* s = line;
            // 1.1.1: a file saved as "UTF-8 with BOM" starts with EF BB BF - without this skip the
            // first line's name carried those bytes and its setting was silently ignored.
            if (first && std::memcmp(s, "\xEF\xBB\xBF", 3) == 0) {
                s += 3;
            }
            first = false;
            s = Trim(s);
            if (*s == '\0' || *s == '#' || *s == ';') {
                continue;
            }
            char* eq = std::strchr(s, '=');
            if (!eq || eq == s) {
                continue;
            }
            *eq = '\0';
            const char* name = Trim(s);
            const char* value = Trim(eq + 1);
            if (*name == '\0') {
                continue;
            }
            size_t have = 0;
            getenv_s(&have, nullptr, 0, name);
            if (have != 0) {
                ++skipped; // a launcher's environment (or the player's file) wins
                continue;
            }
            _putenv_s(name, value);
            SetEnvironmentVariableA(name, value);
            ++set;
        }
    }
};

FlagsFile g_flags_file;

std::optional<std::filesystem::path> ReadSavedGame(const std::filesystem::path& file) {
    std::ifstream in{file, std::ios::binary};
    std::string utf8;
    if (!in || !std::getline(in, utf8)) {
        return std::nullopt;
    }
    while (!utf8.empty() && (utf8.back() == '\r' || utf8.back() == ' ')) {
        utf8.pop_back();
    }
    const std::filesystem::path p{std::u8string{utf8.begin(), utf8.end()}};
    std::error_code ec;
    // 1.1.1: a saved folder without sce_sys\param.sfo asks again (see GamePath).
    if (utf8.empty() || !std::filesystem::is_regular_file(p, ec) ||
        !std::filesystem::is_regular_file(p.parent_path() / "sce_sys" / "param.sfo", ec)) {
        return std::nullopt;
    }
    return p;
}

/// Native folder picker (Windows file dialog in folder mode). nullopt = cancelled / failed.
std::optional<std::filesystem::path> PickFolder() {
    const HRESULT init = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    std::optional<std::filesystem::path> result;
    IFileOpenDialog* dialog = nullptr;
    if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                   IID_PPV_ARGS(&dialog)))) {
        DWORD options = 0;
        dialog->GetOptions(&options);
        dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
        dialog->SetTitle(L"shadPS4: choose the game folder (the one that holds eboot.bin)");
        if (SUCCEEDED(dialog->Show(nullptr))) {
            IShellItem* item = nullptr;
            if (SUCCEEDED(dialog->GetResult(&item))) {
                PWSTR name = nullptr;
                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &name))) {
                    result = std::filesystem::path{name};
                    CoTaskMemFree(name);
                }
                item->Release();
            }
        }
        dialog->Release();
    }
    if (SUCCEEDED(init)) {
        CoUninitialize();
    }
    return result;
}

} // namespace

bool Active() {
    return g_flags_file.active;
}

int FlagsApplied() {
    return g_flags_file.applied;
}

int FlagsOverridden() {
    return g_flags_file.overridden;
}

int UserFlagsApplied() {
    return g_flags_file.user_applied;
}

int DefaultsCopied() {
    return g_flags_file.defaults_copied;
}

std::string FlagsFilePath() {
    const std::filesystem::path p = std::filesystem::path{g_flags_file.dir} / FlagsFileName;
    const auto u8 = p.u8string();
    return std::string{u8.begin(), u8.end()};
}

void HideOwnConsole() {
    DWORD pids[2];
    if (GetConsoleWindow() != nullptr && GetConsoleProcessList(pids, 2) == 1) {
        FreeConsole();
        // 2026-10-18: with the console gone, the last-words lines the emulator writes to stderr on a
        // crash (FATAL-FAULT / FATAL-UNHANDLED in signals.cpp, asserts, terminate) would be lost.
        // Keep them in user/log/fatal.txt (one file per session) for players' bug reports.
        std::error_code ec;
        const auto dir = std::filesystem::path{ExeDir()} / L"user" / L"log";
        std::filesystem::create_directories(dir, ec);
        FILE* f = nullptr;
        _wfreopen_s(&f, (dir / L"fatal.txt").c_str(), L"w", stderr);
    }
}

void ShowOutOfMemoryNotice(unsigned long long bytes) {
    static std::atomic<bool> shown{false};
    if (shown.exchange(true)) {
        return;
    }
    wchar_t text[1024];
    swprintf_s(text,
               L"Windows could not give the game %llu MB of memory for the graphics card, so the "
               L"game has to close.\n\nWhat usually helps:\n"
               L"- close other programs (browser, Discord, overlays)\n"
               L"- turn the page file on: Advanced system settings > Performance Settings > "
               L"Advanced > Virtual memory > \"Automatically manage paging file size\"\n"
               L"- keep at least 20 GB free on C:\n\n"
               L"user\\log\\shad_log.txt has the details - please attach it to a bug report.",
               bytes >> 20);
    MessageBoxW(nullptr, text, L"shadPS4: out of memory", MB_OK | MB_ICONERROR | MB_TOPMOST);
}

std::optional<std::filesystem::path> GamePath() {
    const std::filesystem::path saved = std::filesystem::path{g_flags_file.dir} / GameFileName;
    if (auto p = ReadSavedGame(saved)) {
        return p;
    }
    for (;;) {
        const auto folder = PickFolder();
        if (!folder) {
            return std::nullopt;
        }
        std::error_code ec;
        const auto eboot = *folder / "eboot.bin";
        // 1.1.1 (GitHub #38): a folder with eboot.bin but no sce_sys\param.sfo started and then
        // closed with "sceAppContentInitialize: Failed to get TITLE_ID" - the game's identity is in
        // that file. Said here instead.
        if (std::filesystem::is_regular_file(eboot, ec) &&
            !std::filesystem::is_regular_file(*folder / "sce_sys" / "param.sfo", ec)) {
            if (MessageBoxW(nullptr,
                            L"That folder has eboot.bin but no sce_sys\\param.sfo, so the game "
                            L"cannot start (the game's name and ID are in that file).\n\nChoose the "
                            L"game's complete folder: the one that holds eboot.bin AND the sce_sys "
                            L"folder. If your copy has no sce_sys folder, it is incomplete.",
                            L"shadPS4", MB_OKCANCEL | MB_ICONINFORMATION) != IDOK) {
                return std::nullopt;
            }
            continue;
        }
        if (std::filesystem::is_regular_file(eboot, ec)) {
            std::ofstream out{saved, std::ios::binary | std::ios::trunc};
            const auto u8 = eboot.u8string();
            out.write(reinterpret_cast<const char*>(u8.data()),
                      static_cast<std::streamsize>(u8.size()));
            out << "\n";
            return eboot;
        }
        if (MessageBoxW(nullptr,
                        L"That folder has no eboot.bin.\n\nChoose the game's own folder: the one "
                        L"that holds eboot.bin and the sce_sys folder.",
                        L"shadPS4", MB_OKCANCEL | MB_ICONINFORMATION) != IDOK) {
            return std::nullopt;
        }
    }
}

} // namespace Common::ReleaseMode

#else

namespace Common::ReleaseMode {
bool Active() {
    return false;
}
int FlagsApplied() {
    return 0;
}
int FlagsOverridden() {
    return 0;
}
int UserFlagsApplied() {
    return 0;
}
int DefaultsCopied() {
    return 0;
}
std::string FlagsFilePath() {
    return {};
}
void HideOwnConsole() {}
std::optional<std::filesystem::path> GamePath() {
    return std::nullopt;
}
void ShowOutOfMemoryNotice(unsigned long long) {}
} // namespace Common::ReleaseMode

#endif
