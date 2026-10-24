// SPDX-License-Identifier: GPL-2.0-or-later
// Renderer v3: the fork's logging (spdlog, its own log files) replaced by this port's game log, the
// runtime's stdout that the launcher keeps. Same macros (LOG_INFO(Class, "fmt", ...)); warnings
// and errors by default, TLG_V3_LOG=debug|info|warning|error|off to change it (the fork writes
// about ten thousand Info lines in twelve minutes of play).
#pragma once

#include <string_view>
#include <fmt/format.h>

#include "common/logging/classes.h"
#include "v3_stutter.h"

namespace spdlog {
enum class level { trace, debug, info, warn, err, critical, off };
}

namespace Common::Log {
// The lowest level written (TLG_V3_LOG, read once).
spdlog::level Threshold();
void Write(std::string_view log_class, spdlog::level level, const char* file, int line,
           std::string_view function, std::string_view text);
inline bool ShouldLog(spdlog::level level) {
    return level >= Threshold();
}
void Setup(std::string_view);
void Flush();
} // namespace Common::Log

// Each site counts its calls, logged or not, for the stutter report (v3_stutter.h).
#define LOG_GENERIC(log_class, log_level, log_format, ...)                                         \
    do {                                                                                           \
        static V3::Stutter::Site v3_log_site{__FILE__, __LINE__, log_format};                      \
        v3_log_site.Hit();                                                                         \
        if (Common::Log::ShouldLog(log_level)) {                                                   \
            Common::Log::Write(log_class, log_level, __FILE__, __LINE__, __func__,                 \
                               fmt::format(log_format, ##__VA_ARGS__));                            \
        }                                                                                          \
    } while (false)

#define LOG_TRACE(log_class, ...) (void(0))
#define LOG_DEBUG(log_class, ...)                                                                  \
    LOG_GENERIC(Common::Log::Class::log_class, spdlog::level::debug, __VA_ARGS__)
#define LOG_INFO(log_class, ...)                                                                   \
    LOG_GENERIC(Common::Log::Class::log_class, spdlog::level::info, __VA_ARGS__)
#define LOG_WARNING(log_class, ...)                                                                \
    LOG_GENERIC(Common::Log::Class::log_class, spdlog::level::warn, __VA_ARGS__)
#define LOG_ERROR(log_class, ...)                                                                  \
    LOG_GENERIC(Common::Log::Class::log_class, spdlog::level::err, __VA_ARGS__)
#define LOG_CRITICAL(log_class, ...)                                                               \
    LOG_GENERIC(Common::Log::Class::log_class, spdlog::level::critical, __VA_ARGS__)
