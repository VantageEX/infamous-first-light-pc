// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstdint>

namespace iss::visual_capture {

// All behavior is opt-in. F10 starts one five-frame GameOnly sequence.
bool Enabled() noexcept;
bool Start(std::uint64_t tick_ns) noexcept;
void OnFrame(std::uint64_t tick_ns, unsigned frame, bool reused) noexcept;

} // namespace iss::visual_capture
