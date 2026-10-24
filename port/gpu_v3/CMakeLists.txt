# SPDX-License-Identifier: GPL-2.0-or-later
# Renderer v3, behind this port's GPU library interface (gpu/tlggpu.h). Origin and credit:
#   shadps4/  axylpup's shadPS4 second-son-v1.1.1 (528aea02): the video core, the shader
#             recompiler, ImGui's renderer, common/, the core headers, GnmDriver, VideoOut and the
#             kernel event queue (README.md)
#   glue/     this port's: the runtime's memory, kernel and window under that video core, as
#             renderer v2's gpu/ does for Castlesea's
cmake_minimum_required(VERSION 3.22)
project(issgpu3 CXX C)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PORT ${CMAKE_CURRENT_SOURCE_DIR}/../gpu)
set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/shadps4)
set(GLUE ${CMAKE_CURRENT_SOURCE_DIR}/glue)

find_package(Vulkan REQUIRED)
find_package(fmt REQUIRED)
find_package(Boost REQUIRED)
find_package(tsl-robin-map REQUIRED)
find_package(VulkanMemoryAllocator CONFIG QUIET)
if (NOT VulkanMemoryAllocator_FOUND)
    find_path(VMA_INCLUDE_DIR vk_mem_alloc.h PATH_SUFFIXES vma REQUIRED)
    add_library(VulkanMemoryAllocator INTERFACE)
    add_library(GPUOpen::VulkanMemoryAllocator ALIAS VulkanMemoryAllocator)
    target_include_directories(VulkanMemoryAllocator INTERFACE ${VMA_INCLUDE_DIR})
endif()
find_package(glslang CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(XXHASH REQUIRED IMPORTED_TARGET libxxhash)
pkg_check_modules(SDL3 REQUIRED IMPORTED_TARGET sdl3)
find_package(miniz QUIET)
if (NOT miniz_FOUND)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    add_subdirectory(${PORT}/third_party/miniz miniz EXCLUDE_FROM_ALL)
    add_library(miniz::miniz ALIAS miniz)
endif()
set(SIRIT_USE_SYSTEM_SPIRV_HEADERS OFF CACHE BOOL "" FORCE)
add_subdirectory(${PORT}/third_party/sirit sirit EXCLUDE_FROM_ALL)

# Host shaders: the fork's own list, embedded as GLSL text (compiled with glslang at run time).
add_subdirectory(${SRC}/video_core/host_shaders host_shaders)

file(GLOB_RECURSE VIDEO_CORE ${SRC}/video_core/*.cpp)
file(GLOB_RECURSE RECOMPILER ${SRC}/shader_recompiler/*.cpp)
set(ALL ${VIDEO_CORE} ${RECOMPILER})
foreach(n decoder error fault_monitor io_file native_clock number_utils path_util rdtsc
          recursive_lock reserved_memory signal_context spin_lock stb string_util thread)
    list(APPEND ALL ${SRC}/common/${n}.cpp)
endforeach()
list(APPEND ALL ${SRC}/core/libraries/gnmdriver/gnmdriver.cpp ${SRC}/core/libraries/kernel/equeue.cpp
    ${GLUE}/logging.cpp ${GLUE}/dlss_off.cpp ${GLUE}/assert.cpp)
file(GLOB VIDEOOUT ${SRC}/core/libraries/videoout/*.cpp)
list(APPEND ALL ${VIDEOOUT})
# The fork's files this port replaces (glue/overrides/, each says what it changes).
foreach(original video_core/renderer_vulkan/vk_presenter.cpp)
    get_filename_component(name ${original} NAME)
    list(REMOVE_ITEM ALL ${SRC}/${original})
    list(APPEND ALL ${GLUE}/overrides/${name})
endforeach()
# This port's own interface (its look): the overlay, menu, settings, save dialogs, game hooks.
set(OURS_SOURCES ${PORT}/shim/tlg_overlay.cpp ${PORT}/shim/tlg_settings.cpp
    ${PORT}/shim/tlg_perf_stats.cpp ${PORT}/shim/tlg_png.cpp ${PORT}/shim/tlg_orbs.cpp
    ${PORT}/shim/tlg_actors.cpp ${PORT}/shim/tlg_view.cpp)
set(IMGUI_SOURCES ${PORT}/third_party/imgui/backends/imgui_impl_vulkan.cpp
    ${PORT}/third_party/imgui/imgui.cpp ${PORT}/third_party/imgui/imgui_draw.cpp
    ${PORT}/third_party/imgui/imgui_tables.cpp ${PORT}/third_party/imgui/imgui_widgets.cpp)

# Windows: a static library linked into tlg-probe-v3.exe (as renderer v2's).
add_library(issgpu3 STATIC ${ALL} ${OURS_SOURCES} ${IMGUI_SOURCES})
add_dependencies(issgpu3 host_shaders)
# glue/ first: its headers (common/logging/log.h) stand in for the fork's.
target_include_directories(issgpu3 PRIVATE ${GLUE} ${SRC} ${HOST_SHADERS_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/../gpu_v2/compat
    ${PORT}/third_party/gcn/include ${PORT}/third_party/half/include
    ${PORT}/third_party/magic_enum/include ${PORT}/third_party/magic_enum/include/magic_enum
    ${PORT}/third_party/imgui ${PORT}/third_party/imgui/backends ${PORT}/third_party/xbyak
    ${PORT}/third_party/vulkan-headers/include ${PORT}/shim ${PORT} ${PORT}/..)
target_compile_definitions(issgpu3 PRIVATE BOOST_ASIO_STANDALONE NDEBUG IMGUI_USER_CONFIG="imgui/imgui_config.h"
    TLG_RENDERER_V3 TLG_RENDERER_HOSTED TLG_FONT_PATH="${PORT}/third_party/fonts/DejaVuSans.ttf")
# Our overlay's ImGui Vulkan backend loads its functions itself (as in this port's own build).
set_source_files_properties(${OURS_SOURCES} ${PORT}/third_party/imgui/backends/imgui_impl_vulkan.cpp
    PROPERTIES COMPILE_DEFINITIONS "IMGUI_IMPL_VULKAN_NO_PROTOTYPES")
# As the fork builds: AVX2 (x86-64-v3).
target_compile_options(issgpu3 PRIVATE -O2 -g1 -march=x86-64-v3 -fdeclspec -Wno-deprecated-declarations)
target_link_libraries(issgpu3 PUBLIC Vulkan::Vulkan fmt::fmt Boost::headers tsl::robin_map PkgConfig::XXHASH
    PkgConfig::SDL3 GPUOpen::VulkanMemoryAllocator glslang::glslang glslang::SPIRV sirit miniz::miniz)
if (WIN32)
    target_compile_definitions(issgpu3 PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0A00 WINVER=0x0A00
        NTDDI_VERSION=0x0A000006)
    target_compile_options(issgpu3 PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/../gpu_v2/compat/mingw.h)
endif()
target_sources(issgpu3 PRIVATE ${GLUE}/gpu_host.cpp ${GLUE}/window.cpp ${GLUE}/bridge.cpp
    ${GLUE}/precompile_jobs.cpp ${GLUE}/thread_cpu.cpp)
pkg_check_modules(PNG REQUIRED IMPORTED_TARGET libpng)
find_package(Zydis QUIET)
if (NOT Zydis_FOUND)
    pkg_check_modules(ZYDIS REQUIRED IMPORTED_TARGET zydis)
    add_library(Zydis::Zydis ALIAS PkgConfig::ZYDIS)
endif()
target_link_libraries(issgpu3 PUBLIC PkgConfig::PNG Zydis::Zydis)

# Windows: tlg-probe-v3.exe, this port's loader and runtime (as port/gpu/CMakeLists.txt builds
# tlg-probe.exe) linked with this renderer.
if (WIN32)
    enable_language(RC)
    set(RT ${PORT}/..)
    file(GLOB V3_RUNTIME CONFIGURE_DEPENDS ${RT}/src/runtime*.c)
    file(GLOB V3_ATRAC9 CONFIGURE_DEPENDS ${RT}/third_party/LibAtrac9/C/src/*.c)
    add_library(atrac9_v3 STATIC ${V3_ATRAC9})
    target_compile_options(atrac9_v3 PRIVATE -w)
    set_target_properties(atrac9_v3 PROPERTIES C_STANDARD 99 INTERPROCEDURAL_OPTIMIZATION OFF)
    add_executable(tlg-probe-v3 ${RT}/src/probe.c ${V3_RUNTIME} ${RT}/src/compat_win.c ${RT}/src/cpu_check.c
        ${RT}/src/vulkan_smoke.c ${RT}/src/isa_rewrite.c ${RT}/src/isa_emulate.c
        ${RT}/packaging/windows/tlg-probe.rc)
    set_source_files_properties(${RT}/src/isa_emulate.c PROPERTIES COMPILE_OPTIONS "-mgeneral-regs-only")
    set_target_properties(tlg-probe-v3 PROPERTIES C_STANDARD 11 C_EXTENSIONS OFF)
    if (DEFINED ENV{TLG_BUILD_COMMIT})
        target_compile_definitions(tlg-probe-v3 PRIVATE TLG_BUILD_COMMIT="$ENV{TLG_BUILD_COMMIT}")
    endif()
    target_include_directories(tlg-probe-v3 PRIVATE ${RT} ${RT}/src)
    pkg_check_modules(FREETYPE REQUIRED IMPORTED_TARGET freetype2)
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswscale libswresample)
    target_link_libraries(tlg-probe-v3 PRIVATE issgpu3 atrac9_v3 PkgConfig::FREETYPE PkgConfig::FFMPEG
        onecore bcrypt ws2_32 ntdll synchronization winmm pdh psapi ole32 dbghelp -pthread)
    target_link_options(tlg-probe-v3 PRIVATE -Wl,--error-limit=0 -Wl,--disable-high-entropy-va "SHELL:-Xlinker --stack -Xlinker 16777216,16777216")
endif()
