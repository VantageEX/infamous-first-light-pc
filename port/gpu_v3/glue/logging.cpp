// SPDX-License-Identifier: GPL-2.0-or-later
// Renderer v3: the fork's log lines into this port's game log (glue/common/logging/log.h).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string_view>

#include "common/logging/log.h"

namespace Common::Log {

spdlog::level Threshold() {
    static const spdlog::level threshold = [] {
        const char* value = std::getenv("TLG_V3_LOG");
        if (!value || !*value) return spdlog::level::warn;
        static constexpr std::pair<const char*, spdlog::level> names[] = {
            {"trace", spdlog::level::trace}, {"debug", spdlog::level::debug},
            {"info", spdlog::level::info},   {"warning", spdlog::level::warn},
            {"error", spdlog::level::err},   {"off", spdlog::level::off}};
        for (const auto& [name, level] : names) {
            if (std::strcmp(value, name) == 0) return level;
        }
        return spdlog::level::warn;
    }();
    return threshold;
}

void Write(std::string_view log_class, spdlog::level level, const char* file, int line,
           std::string_view function, std::string_view text) {
    static constexpr const char* names[] = {"Trace", "Debug", "Info", "Warning", "Error", "Critical", "Off"};
    const char* base = std::strrchr(file, '/');
    const char* back = std::strrchr(file, '\\');
    if (back > base) base = back;
    base = base ? base + 1 : file;
    // One printf per line: the runtime's other threads write to the same stdout.
    static std::mutex mutex;
    std::scoped_lock lock{mutex};
    std::printf("V3 [%.*s] <%s> %s:%d %.*s: %.*s\n", int(log_class.size()), log_class.data(),
                names[static_cast<int>(level)], base, line,
                int(function == "operator()" ? 6 : function.size()),
                function == "operator()" ? "lambda" : function.data(), int(text.size()), text.data());
    if (level >= spdlog::level::err) std::fflush(stdout);
}

void Setup(std::string_view) {}
void Flush() {
    std::fflush(stdout);
}

} // namespace Common::Log
