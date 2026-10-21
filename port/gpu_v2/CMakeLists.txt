# SPDX-License-Identifier: GPL-2.0-or-later
# Renderer v2 on Linux, behind this port's GPU library interface (gpu/tlggpu.h). Origin and credit:
#   shadps4/  the weshayoub92 shadPS4-perf fork (153a5aa7) with the Castlesea port's local patch
#             (pc-port/third_party/shadPS4-perf-local.patch), the files this build uses
#   gpu/      Castlesea/infamous-second-son-pc-port 8e167dc pc-port/gpu, adapted to Linux and to
#             this port's runtime (gpu_host.cpp, precompile_jobs.inl)
#   compat/   stb and nlohmann/json headers, tracy stubs; compat/linux/ the Win32 subset it uses,
#             over POSIX (Linux only: on Windows the real headers)
# The player-facing interface is this port's own (gpu/shim/tlg_overlay.cpp and its settings).
# Sources as pc-port/scripts/build_gpu.py lists them. See README.md.
cmake_minimum_required(VERSION 3.22)
project(issgpu CXX C)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PORT ${CMAKE_CURRENT_SOURCE_DIR}/../gpu)
set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/shadps4)
set(CS ${CMAKE_CURRENT_SOURCE_DIR}/gpu)

find_package(Vulkan REQUIRED)
find_package(fmt REQUIRED)
find_package(Boost REQUIRED)
find_package(tsl-robin-map REQUIRED)
find_package(VulkanMemoryAllocator CONFIG QUIET)
if (NOT VulkanMemoryAllocator_FOUND)
    # MSYS2 ships the header (include/vma/vk_mem_alloc.h) without a CMake package.
    find_path(VMA_INCLUDE_DIR vk_mem_alloc.h PATH_SUFFIXES vma REQUIRED)
    add_library(VulkanMemoryAllocator INTERFACE)
    add_library(GPUOpen::VulkanMemoryAllocator ALIAS VulkanMemoryAllocator)
    target_include_directories(VulkanMemoryAllocator INTERFACE ${VMA_INCLUDE_DIR})
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(XXHASH REQUIRED IMPORTED_TARGET libxxhash)
pkg_check_modules(SDL3 REQUIRED IMPORTED_TARGET sdl3)
find_package(Zydis QUIET)
if (NOT Zydis_FOUND)
    pkg_check_modules(ZYDIS REQUIRED IMPORTED_TARGET zydis)
    add_library(Zydis::Zydis ALIAS PkgConfig::ZYDIS)
endif()
if (NOT WIN32)
    pkg_check_modules(CRYPTO REQUIRED IMPORTED_TARGET libcrypto) # BCrypt's SHA-256 (compat/linux/bcrypt.h)
endif()
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswscale libswresample)
find_package(miniz QUIET)
if (NOT miniz_FOUND)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    add_subdirectory(${PORT}/third_party/miniz miniz EXCLUDE_FROM_ALL)
    add_library(miniz::miniz ALIAS miniz)
endif()
find_program(GLSLANG_COMPILER NAMES glslang glslangValidator REQUIRED)
set(SIRIT_USE_SYSTEM_SPIRV_HEADERS OFF CACHE BOOL "" FORCE)
add_subdirectory(${PORT}/third_party/sirit sirit EXCLUDE_FROM_ALL)

# Host shaders (the fork's list).
set(SHADER_SRC ${SRC}/video_core/host_shaders)
set(SHADER_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/include)
set(SHADER_DIR ${SHADER_INCLUDE}/video_core/host_shaders)
set(SHADER_HEADERS "")
function(add_host_shader FILENAME SHADER_NAME)
    set(OUT ${SHADER_DIR}/${SHADER_NAME}.h)
    string(TOUPPER ${SHADER_NAME} SYMBOL)
    set(DEFINES ${ARGN})
    list(TRANSFORM DEFINES PREPEND "-D")
    add_custom_command(OUTPUT ${OUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_DIR}
        COMMAND ${GLSLANG_COMPILER} --target-env vulkan1.3 -o ${OUT} --vn ${SYMBOL} ${DEFINES} ${SHADER_SRC}/${FILENAME}
        COMMAND ${CMAKE_COMMAND} "-DHEADER_FILE=${OUT}" -P ${SHADER_SRC}/AddShaderHeaderInclude.cmake
        DEPENDS ${SHADER_SRC}/${FILENAME} VERBATIM)
    set(SHADER_HEADERS ${SHADER_HEADERS} ${OUT} PARENT_SCOPE)
endfunction()
add_host_shader(color_to_ms_depth.frag color_to_ms_depth_frag)
add_host_shader(ms_image_blit.frag ms_image_blit_frag)
add_host_shader(ms_image_blit.frag ms_image_blit_msaa_frag SRC_MSAA=1)
add_host_shader(fault_buffer_process.comp fault_buffer_process_comp)
add_host_shader(fs_tri.vert fs_tri_vert)
add_host_shader(fsr.comp fsr_easu_comp SAMPLE_EASU=1)
add_host_shader(fsr.comp fsr_rcas_comp SAMPLE_RCAS=1)
add_host_shader(post_process.frag post_process_frag)
foreach(BPP IN ITEMS 8 16 32 64 96 128)
    add_host_shader(tiling.comp tiling_micro_${BPP}_comp BITS_PER_PIXEL=${BPP} IS_MACRO_TILED=0)
    add_host_shader(tiling.comp tiling_macro_${BPP}_comp BITS_PER_PIXEL=${BPP} IS_MACRO_TILED=1)
endforeach()
add_custom_target(host_shaders DEPENDS ${SHADER_HEADERS})

# Sources as build_gpu.py: the fork's video core and recompiler, 24 files replaced by overrides.
file(GLOB_RECURSE VIDEO_CORE ${SRC}/video_core/*.cpp)
file(GLOB_RECURSE RECOMPILER ${SRC}/shader_recompiler/*.cpp)
set(ALL ${VIDEO_CORE} ${RECOMPILER})
list(FILTER ALL EXCLUDE REGEX "renderdoc\\.cpp$")
set(OVERRIDES
    video_core/buffer_cache/buffer_cache.cpp=buffer_cache.cpp
    video_core/renderer_vulkan/vk_rasterizer.cpp=vk_rasterizer.cpp
    video_core/renderer_vulkan/vk_pipeline_cache.cpp=vk_pipeline_cache.cpp
    video_core/renderer_vulkan/vk_graphics_pipeline.cpp=vk_graphics_pipeline.cpp
    video_core/renderer_vulkan/vk_pipeline_serialization.cpp=vk_pipeline_serialization.cpp
    video_core/renderer_vulkan/vk_runtime.cpp=vk_runtime.cpp
    video_core/amdgpu/liverpool.cpp=liverpool.cpp
    video_core/page_manager.cpp=page_manager.cpp
    video_core/renderer_vulkan/vk_instance.cpp=vk_instance.cpp
    video_core/renderer_vulkan/vk_resource_pool.cpp=vk_resource_pool.cpp
    video_core/buffer_cache/buffer.cpp=buffer.cpp
    video_core/renderer_vulkan/vk_presenter.cpp=vk_presenter.cpp
    video_core/renderer_vulkan/vk_scheduler.cpp=vk_scheduler.cpp
    video_core/renderer_vulkan/vk_staging_buffer_pool.cpp=vk_staging_buffer_pool.cpp
    video_core/texture_cache/texture_cache.cpp=texture_cache.cpp
    shader_recompiler/ir/passes/flatten_extended_userdata_pass.cpp=flatten_extended_userdata_pass.cpp
    shader_recompiler/recompiler.cpp=recompiler.cpp
    shader_recompiler/ir/passes/resource_patching_pass.cpp=resource_patching_pass.cpp
    shader_recompiler/frontend/fetch_shader.cpp=fetch_shader.cpp
    video_core/cache_storage.cpp=cache_storage.cpp
    video_core/renderer_vulkan/vk_swapchain.cpp=vk_swapchain.cpp
    video_core/texture_cache/image.cpp=image.cpp)
foreach(pair ${OVERRIDES})
    string(REPLACE "=" ";" parts ${pair})
    list(GET parts 0 original)
    list(GET parts 1 replacement)
    list(REMOVE_ITEM ALL ${SRC}/${original})
    list(APPEND ALL ${CS}/overrides/${replacement})
endforeach()
foreach(n decoder error io_file native_clock number_utils rdtsc spin_lock recursive_lock string_util thread signal_context)
    list(APPEND ALL ${SRC}/common/${n}.cpp)
endforeach()
list(APPEND ALL ${SRC}/common/logging/classes.cpp ${CS}/overrides/gnmdriver.cpp
    ${CS}/overrides/videoout_driver.cpp ${SRC}/core/libraries/videoout/video_out.cpp
    ${SRC}/core/libraries/kernel/equeue.cpp
    ${SRC}/imgui/renderer/imgui_impl_vulkan.cpp ${SRC}/imgui/renderer/texture_manager.cpp ${SRC}/common/stb.cpp)
# Ajm, Videodec: this port's runtime implements them (runtime_*.c).
list(APPEND ALL ${SRC}/core/libraries/videodec/video_utils.cpp)
file(GLOB CS_SOURCES ${CS}/*.cpp)
# This port's own interface (its look): the overlay, menu, settings, save dialogs, game hooks.
set(OURS_SOURCES ${PORT}/shim/tlg_overlay.cpp ${PORT}/shim/tlg_settings.cpp
    ${PORT}/shim/tlg_perf_stats.cpp ${PORT}/shim/tlg_png.cpp ${PORT}/shim/tlg_orbs.cpp
    ${PORT}/shim/tlg_actors.cpp ${PORT}/shim/tlg_view.cpp)
# The bridge to the Windows runtime, its window and save dialogs: replaced by this port's own.
list(REMOVE_ITEM CS_SOURCES ${CS}/save_dialog_ui.cpp)
list(APPEND ALL ${CS_SOURCES})

# Windows: a static library linked into tlg-probe-v2.exe (as port/gpu does for tlg-probe.exe), since
# a DLL cannot leave the runtime_* symbols for the executable to resolve.
if (WIN32)
    set(ISSGPU_KIND STATIC)
else()
    set(ISSGPU_KIND SHARED)
endif()
add_library(issgpu ${ISSGPU_KIND} ${ALL} ${OURS_SOURCES}
    ${PORT}/third_party/imgui/backends/imgui_impl_vulkan.cpp
    ${PORT}/third_party/imgui/imgui.cpp ${PORT}/third_party/imgui/imgui_draw.cpp
    ${PORT}/third_party/imgui/imgui_tables.cpp ${PORT}/third_party/imgui/imgui_widgets.cpp)
add_dependencies(issgpu host_shaders)
if (NOT WIN32)
    target_include_directories(issgpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat/linux)
endif()
target_include_directories(issgpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compat ${CS}/shim ${CS}
    ${PORT}/third_party/imgui/backends ${PORT}/.. ${SRC}/core/libraries/gnmdriver ${SRC} ${SRC}/..
    ${SHADER_INCLUDE} ${PORT}/third_party/gcn/include ${PORT}/third_party/half/include
    ${PORT}/third_party/magic_enum/include ${PORT}/third_party/magic_enum/include/magic_enum
    ${PORT}/third_party/imgui ${PORT}/third_party/xbyak ${PORT} ${PORT}/third_party/miniz ${PORT}/third_party/vulkan-headers/include)
# This port's interface headers (tlg_overlay.h...), after the renderer's own shims: the names they
# share (core/memory.h, sdl_window.h) are the renderer's.
target_include_directories(issgpu PRIVATE ${PORT}/shim)
target_compile_definitions(issgpu PRIVATE BOOST_ASIO_STANDALONE NDEBUG IMGUI_USER_CONFIG="imgui/imgui_config.h")
target_compile_options(issgpu PRIVATE -O2 -g1 -fdeclspec -Wno-deprecated-declarations)
target_link_libraries(issgpu PUBLIC Vulkan::Vulkan fmt::fmt Boost::headers tsl::robin_map PkgConfig::XXHASH
    PkgConfig::SDL3 Zydis::Zydis GPUOpen::VulkanMemoryAllocator miniz::miniz PkgConfig::FFMPEG sirit)
if (WIN32)
    target_compile_definitions(issgpu PRIVATE NOMINMAX)
    target_compile_options(issgpu PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/compat/mingw.h)
    target_link_libraries(issgpu PUBLIC ws2_32 bcrypt ntdll synchronization winmm pdh gdiplus psapi ole32
        dbghelp onecore)
else()
    target_compile_options(issgpu PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/compat/linux/iss_win_compat.h)
    target_link_options(issgpu PRIVATE -Wl,--allow-shlib-undefined)
    target_link_libraries(issgpu PRIVATE PkgConfig::CRYPTO)
endif()

# Our overlay's ImGui Vulkan backend loads its functions itself (as in this port's own build).
set_source_files_properties(${OURS_SOURCES} ${PORT}/third_party/imgui/backends/imgui_impl_vulkan.cpp
    PROPERTIES COMPILE_DEFINITIONS "IMGUI_IMPL_VULKAN_NO_PROTOTYPES")
target_compile_definitions(issgpu PRIVATE TLG_RENDERER_V2
    TLG_FONT_PATH="${PORT}/third_party/fonts/DejaVuSans.ttf")

# Windows: tlg-probe-v2.exe, this port's loader and runtime (as port/gpu/CMakeLists.txt builds
# tlg-probe.exe) linked with this renderer.
if (WIN32)
    enable_language(RC)
    set(RT ${PORT}/..)
    file(GLOB V2_RUNTIME CONFIGURE_DEPENDS ${RT}/src/runtime*.c)
    file(GLOB V2_ATRAC9 CONFIGURE_DEPENDS ${RT}/third_party/LibAtrac9/C/src/*.c)
    add_library(atrac9_v2 STATIC ${V2_ATRAC9})
    target_compile_options(atrac9_v2 PRIVATE -w)
    set_target_properties(atrac9_v2 PROPERTIES C_STANDARD 99 INTERPROCEDURAL_OPTIMIZATION OFF)
    add_executable(tlg-probe-v2 ${RT}/src/probe.c ${V2_RUNTIME} ${RT}/src/compat_win.c ${RT}/src/cpu_check.c
        ${RT}/src/vulkan_smoke.c ${RT}/src/isa_rewrite.c ${RT}/src/isa_emulate.c
        ${RT}/packaging/windows/tlg-probe.rc)
    set_source_files_properties(${RT}/src/isa_emulate.c PROPERTIES COMPILE_OPTIONS "-mgeneral-regs-only")
    set_target_properties(tlg-probe-v2 PROPERTIES C_STANDARD 11 C_EXTENSIONS OFF)
    if (DEFINED ENV{TLG_BUILD_COMMIT})
        target_compile_definitions(tlg-probe-v2 PRIVATE TLG_BUILD_COMMIT="$ENV{TLG_BUILD_COMMIT}")
    endif()
    target_include_directories(tlg-probe-v2 PRIVATE ${RT} ${RT}/src)
    pkg_check_modules(FREETYPE REQUIRED IMPORTED_TARGET freetype2)
    target_link_libraries(tlg-probe-v2 PRIVATE issgpu atrac9_v2 PkgConfig::FREETYPE onecore bcrypt ws2_32 -pthread)
    target_link_options(tlg-probe-v2 PRIVATE -Wl,--disable-high-entropy-va "SHELL:-Xlinker --stack -Xlinker 16777216,16777216")
endif()
