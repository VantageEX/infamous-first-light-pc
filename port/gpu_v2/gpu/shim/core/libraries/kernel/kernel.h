// Adapted from bloodborne_pc@4d4d20812c0af23a11c8b8d40f5e06a9054925ef, GPL-2.0-or-later.
// bbport: the parts of shadPS4's kernel.h used by vendored libraries.
#pragma once
#include "common/types.h"
#include "core/libraries/kernel/orbis_error.h"
namespace Libraries::Kernel {
int* PS4_SYSV_ABI __Error();
} // namespace Libraries::Kernel
