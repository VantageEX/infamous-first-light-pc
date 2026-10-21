// SPDX-License-Identifier: GPL-2.0-or-later
// Preserve the driver's cache during long sessions and diagnostic termination.
#include <windows.h>
#include <cstdio>
#include <mutex>
#include <xxhash.h>
#include "common/perf_stats.h"
#include "core/emulator_settings.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/driver_cache_data.h"

namespace Vulkan {
bool PipelineCache::CheckpointDriverCache(){
    // Vulkan internally synchronizes this cache (no EXTERNALLY_SYNCHRONIZED flag).
    // This mutex serializes our snapshot writers; no compiler or draw is skipped.
    static std::mutex mutex;
    static u64 previous_hash=0;
    std::scoped_lock guard{mutex};
    if(!EmulatorSettings.IsPipelineCacheEnabled()||!pipeline_cache)return false;
    const auto started=Common::Perf::NowNs();
    const auto snapshot=instance.GetDevice().getPipelineCacheData(*pipeline_cache);
    if(snapshot.result!=vk::Result::eSuccess||snapshot.value.size()<32||snapshot.value.size()>256ULL*1024*1024-24){
        std::fprintf(stderr,"PIPELINE_CHECKPOINT unavailable result=%d\n",int(snapshot.result));return false;}
    const u64 hash=XXH3_64bits(snapshot.value.data(),snapshot.value.size());
    if(hash==previous_hash)return true;
    const auto bytes=PackDriverCache(snapshot.value);
    auto temporary=driver_cache_path;temporary+=L".pending-"+std::to_wstring(GetCurrentProcessId());
    HANDLE file=CreateFileW(temporary.c_str(),GENERIC_WRITE,0,nullptr,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(file==INVALID_HANDLE_VALUE){std::fprintf(stderr,"PIPELINE_CHECKPOINT open error=%lu\n",GetLastError());return false;}
    DWORD written=0;bool ok=WriteFile(file,bytes.data(),DWORD(bytes.size()),&written,nullptr)&&written==bytes.size();
    if(ok)ok=FlushFileBuffers(file)!=0;const DWORD write_error=ok?0:GetLastError();CloseHandle(file);
    if(ok)ok=MoveFileExW(temporary.c_str(),driver_cache_path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)!=0;
    if(ok)previous_hash=hash;
    std::fprintf(stderr,"PIPELINE_CHECKPOINT ok=%d bytes=%zu elapsed_ms=%.3f error=%lu\n",ok,snapshot.value.size(),double(Common::Perf::NowNs()-started)/1e6,ok?0:write_error?write_error:GetLastError());
    return ok;
}
}
