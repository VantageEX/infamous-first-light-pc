#pragma once

#include <cstdint>

namespace AmdGpu { union Regs; }
namespace Core { class MemoryManager; }
namespace VideoCore { class BufferCache; }
namespace Vulkan { class GraphicsPipeline; }

namespace iss::native_geometry {

// Opt-in, bounded read-only capture for the two PS 1318DDD4 draw families.
// Call from Rasterizer::DrawRecord after BindResources succeeds and before
// BindVertexBuffers/BindIndexBuffer change residency. When the flag is absent,
// this returns immediately. Never use this capture as a cache or draw source.
void ProbeSecondaryDraw(const Vulkan::GraphicsPipeline& pipeline,
                        const AmdGpu::Regs& regs, Core::MemoryManager& memory,
                        VideoCore::BufferCache& buffer_cache, int queue,
                        std::uint32_t index_offset, std::uint64_t tick);

} // namespace iss::native_geometry
