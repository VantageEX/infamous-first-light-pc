// SPDX-License-Identifier: GPL-2.0-or-later
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <string>
#include <fmt/format.h>
#include "common/logging/log.h"
namespace Common::Log {
std::array<Level,NUM_LOG_CLASSES> g_class_levels=[] {std::array<Level,NUM_LOG_CLASSES> a;a.fill(Level::Warning);a[size_t(Class::Render)]=Level::Info;a[size_t(Class::Render_Vulkan)]=Level::Info;return a;}();
void VLog(Class c,Level level,const char *file,int line,const char *function,fmt::string_view format,fmt::format_args args){
    static std::mutex mutex;std::scoped_lock guard{mutex};
    const auto text=fmt::vformat(format,args);
    // Preserve the first occurrence and cumulative counts at powers of two.
    // Bound the table so novel messages cannot grow diagnostic memory forever.
    static std::unordered_map<std::string,unsigned long long> counts;
    const auto key=fmt::format("{}:{}:{}:{}",unsigned(c),unsigned(level),function,text);
    unsigned long long count=1;
    if(counts.size()<4096 || counts.contains(key)) count=++counts[key];
    if(count>1 && (count&(count-1)))return;
    if(count>1)std::fprintf(stderr,"GPU_REPEAT count=%llu ",count);
    std::fprintf(stderr,"GPU [%u:%u] %s:%d %s: %s\n",unsigned(c),unsigned(level),file,line,function,text.c_str());
}
void Flush(){std::fflush(stderr);}
void Shutdown(){Flush();}
void Terminate(){Flush();}
}
