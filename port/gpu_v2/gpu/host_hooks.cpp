// SPDX-License-Identifier: GPL-2.0-or-later
// Optional emulator UI is absent; game presentation remains in imgui_display.cpp.
#include <cstdio>
#include <cstdlib>
#include "common/types.h"
#include "core/devtools/layer.h"
#include "imgui/notifications_layer.h"
#include <csignal>
#include <cstring>
#include <exception>
#include <windows.h>
#include <stdexcept>
// Ahead-of-time shader threads (vk_pipeline_cache.cpp, ISS_AOT_COMPUTE) translate shaders with no
// game state bound: a failed assertion there drops that shader instead of ending the game.
namespace iss {thread_local bool assert_throws=false;
// The shader stage a recipe regeneration works on (RegenerateRecipeSpirv): named when it ends the
// process, so a recipe that crashes the build tool can be found and left out.
thread_local unsigned long long recipe_hash=0;
static void NoteRecipe(const char* what){if(recipe_hash)std::fprintf(stderr,"GPU %s while regenerating recipe stage %llx\n",what,recipe_hash);std::fflush(stderr);}
// abort() and CRT invalid parameters end the process with a fast fail (0xC0000409) and no log:
// print the stage and the return addresses (module+offset, for the linker map) first.
static void FatalTrace(const char* what){
    NoteRecipe(what);
    void* frames[48];const USHORT n=RtlCaptureStackBackTrace(1,48,frames,nullptr);
    for(USHORT i=0;i<n;++i){
        HMODULE module=nullptr;char name[MAX_PATH]="?";
        if(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              static_cast<LPCSTR>(frames[i]),&module)){
            GetModuleFileNameA(module,name,MAX_PATH);
            const char* base=std::strrchr(name,'\\');
            std::fprintf(stderr,"GPU_FATAL %s frame %u %s+0x%llx\n",what,unsigned(i),base?base+1:name,
                         (unsigned long long)(static_cast<char*>(frames[i])-reinterpret_cast<char*>(module)));
        }
    }
    std::fflush(stderr);
}
static const bool terminate_hook=[]{
    std::set_terminate([]{FatalTrace("terminate");std::abort();});
    std::signal(SIGABRT,[](int){FatalTrace("abort");});
    _set_invalid_parameter_handler([](const wchar_t*,const wchar_t*,const wchar_t*,unsigned,uintptr_t){
        FatalTrace("invalid_parameter");std::abort();});
    return true;}();}
void assert_fail_impl(){if(iss::assert_throws)throw std::runtime_error("assertion");iss::NoteRecipe("assertion");std::fflush(stderr);std::abort();}
[[noreturn]] void unreachable_impl(){if(iss::assert_throws)throw std::runtime_error("unreachable");iss::NoteRecipe("unreachable");std::fputs("GPU reached unreachable code\n",stderr);std::abort();}
void assert_fail_debug_msg(const char *message){if(iss::assert_throws)throw std::runtime_error(message);std::fprintf(stderr,"GPU assertion: %s\n",message);std::abort();}
namespace MemoryPatcher {u64 g_eboot_address=0;}
namespace Libraries::Kernel {struct Pthread;thread_local Pthread *g_curthread=nullptr;}
namespace ImGui {
void Layer::AddLayer(Layer*){}
void Layer::RemoveLayer(Layer*){}
namespace Friends {void Register(){}void Unregister(){}}
namespace ShadNetNotify {void Register(){}void Unregister(){}}
namespace InvitationPrompt {void Register(){}void Unregister(){}}
}
namespace Core::Devtools {
void Layer::Draw(){}
bool Layer::ShouldKeepDrawing(){return false;}
void Layer::DrawNullGpuNotice(){std::fputs("Unexpected null GPU request\n",stderr);}
}
namespace shadNotifications {
void QueueNotification(std::string message,float,position,std::vector<u8>,bool){std::fprintf(stderr,"GPU: %s\n",message.c_str());}
}
