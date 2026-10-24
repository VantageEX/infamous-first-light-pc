// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct IssGpuCallbacks {
    uint32_t struct_size,sdk_version,psf_attributes,width,height;
    const char *serial,*title,*user_dir;
    uint64_t (*tsc_frequency)(void);
    uint64_t (*process_time_us)(void);
    int32_t *(*errno_location)(void);
    uint8_t *(*backing)(uint64_t address,uint64_t size);
    uint64_t (*clamp)(uint64_t address,uint64_t size);
    void (*visit_mappings)(void (*visit)(uint64_t address,uint64_t size,uint8_t *backing));
    int (*protect)(uint64_t address,uint64_t size,uint32_t permissions);
    uint64_t image_base;
    // Lock a short packet publish against native VA mapping changes. The lock must
    // never be held while reading guest memory, waiting or calling the GPU driver.
    uint64_t (*mapping_epoch)(void);
    uint64_t (*mapping_lock_shared)(void);
    void (*mapping_unlock_shared)(void);
} IssGpuCallbacks;
int iss_gpu_initialize(const IssGpuCallbacks *callbacks);
uintptr_t iss_gpu_resolve(const char *identity);
int iss_gpu_fault(void *exception_pointers,void *address);
void iss_gpu_map(uint64_t address,uint64_t size,uint8_t *backing);
int iss_gpu_readback(uint64_t address,uint64_t size);
/* The runtime is about to write [address, address+size) of guest memory from a host thread (file
   reads): the GPU tracking of the whole range is invalidated at once, as its page faults would
   one page at a time. Returns 1 when the range is GPU mapped. */
int iss_gpu_invalidate(uint64_t address,uint64_t size);
#ifdef __cplusplus
}
#endif
