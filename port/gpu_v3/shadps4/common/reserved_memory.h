// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>

namespace Common {

/// Address space reserved once and committed in place as it grows, so objects stored in it never
/// move (SlotVector with a stable capacity: notes/parallel-back-design-2026-10-02.md sec. 7a).
/// All three return / take page-aligned ranges; failures are fatal (asserted by the caller).
void* ReserveAddressSpace(std::size_t bytes);
bool CommitAddressSpace(void* base, std::size_t offset, std::size_t bytes);
void ReleaseAddressSpace(void* base, std::size_t bytes);
std::size_t AddressSpacePageSize();

} // namespace Common
