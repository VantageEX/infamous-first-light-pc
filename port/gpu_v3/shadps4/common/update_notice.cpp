// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <fmt/format.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include "common/logging/log.h"
#include "common/thread.h"
#include "common/update_notice.h"

namespace Common {

namespace {
/// "second-son-v1.0.2" -> {1, 0, 2}; missing parts are 0.
std::array<int, 3> ParseTag(std::string_view tag) {
    std::array<int, 3> v{};
    const size_t p = tag.rfind('v');
    if (p == std::string_view::npos) {
        return v;
    }
    size_t i = p + 1;
    for (int& part : v) {
        int n = 0;
        bool any = false;
        while (i < tag.size() && tag[i] >= '0' && tag[i] <= '9') {
            n = n * 10 + (tag[i++] - '0');
            any = true;
        }
        if (!any) {
            break;
        }
        part = n;
        if (i >= tag.size() || tag[i] != '.') {
            break;
        }
        ++i;
    }
    return v;
}

/// "second-son-v1.0.2" -> "1.0.2".
std::string_view ShortName(std::string_view tag) {
    const size_t p = tag.rfind('v');
    return p == std::string_view::npos ? tag : tag.substr(p + 1);
}

std::mutex g_line_mutex;
UpdateLine g_line; // guarded by g_line_mutex; written by the check thread, read by boot frames

void SetLine(std::string text, std::string detail = {}, bool newer = false) {
    std::scoped_lock lock{g_line_mutex};
    g_line = {std::move(text), std::move(detail), newer};
}
} // namespace

UpdateLine GetUpdateLine() {
    std::scoped_lock lock{g_line_mutex};
    return g_line;
}

void StartUpdateNotice() {
    const char* env = std::getenv("SHADPS4_UPDATE_CHECK");
    // SHADPS4_UPDATE_TEST_TAG=second-son-v1.0.0 (test): this build pretends to be that release.
    const char* test_tag = std::getenv("SHADPS4_UPDATE_TEST_TAG");
    const std::string mine = test_tag ? test_tag : ReleaseTag;
    if (env && env[0] == '0') {
        SetLine(fmt::format("Version {}", ShortName(mine)));
        return;
    }
    SetLine(fmt::format("Version {} - checking for updates...", ShortName(mine)));
    std::thread([mine] {
        SetCurrentThreadName("UpdateCheck");
        httplib::Client cli("https://api.github.com");
        cli.set_connection_timeout(5);
        cli.set_read_timeout(5);
        const httplib::Headers headers = {{"User-Agent", "shadPS4-second-son"},
                                          {"Accept", "application/vnd.github+json"}};
        const auto res = cli.Get("/repos/axylpup/shadPS4/releases/latest", headers);
        const auto no_answer = [&] {
            SetLine(fmt::format("Version {} - could not check for updates", ShortName(mine)));
        };
        if (!res || res->status != 200) {
            LOG_INFO(Common, "UpdateCheck: no answer ({})",
                     res ? std::to_string(res->status) : httplib::to_string(res.error()));
            no_answer();
            return;
        }
        const auto json = nlohmann::json::parse(res->body, nullptr, false);
        if (json.is_discarded() || !json.contains("tag_name") || !json["tag_name"].is_string()) {
            LOG_INFO(Common, "UpdateCheck: unexpected answer");
            no_answer();
            return;
        }
        const std::string latest = json["tag_name"].get<std::string>();
        if (ParseTag(latest) == ParseTag(mine)) {
            LOG_INFO(Common, "UpdateCheck: {} is the latest release ({})", mine, latest);
            SetLine(fmt::format("Latest version: {} - Current", ShortName(mine)));
            return;
        }
        if (ParseTag(latest) < ParseTag(mine)) {
            // A test build ahead of the newest release.
            LOG_INFO(Common, "UpdateCheck: {} is newer than the latest release ({})", mine, latest);
            SetLine(fmt::format("Version {} (newest release: {})", ShortName(mine),
                                ShortName(latest)));
            return;
        }
        LOG_INFO(Common, "UpdateCheck: {} is out (this build: {})", latest, mine);
        SetLine(fmt::format("Latest version: {} - New update out!", ShortName(latest)),
                fmt::format("You have {}. Download: github.com/axylpup/shadPS4/releases",
                            ShortName(mine)),
                true);
        // The 45 s in-game notice is gone (user 10-21: it would interrupt the game); the boot
        // screen's line is the notice.
    }).detach();
}

} // namespace Common
