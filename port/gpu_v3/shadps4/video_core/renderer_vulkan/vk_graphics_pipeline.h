// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include <memory>

#include <boost/container/static_vector.hpp>
#include <xxhash.h>

#include "shader_recompiler/frontend/fetch_shader.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/amdgpu/regs_depth.h"
#include "video_core/amdgpu/regs_primitive.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"

namespace VideoCore {
class BufferCache;
class TextureCache;
} // namespace VideoCore

namespace AmdGpu::CpSplit {
struct VertexInputsDerived;
}

namespace Vulkan {

static constexpr u32 MaxShaderStages = static_cast<u32>(Shader::LogicalStage::NumLogicalStages);
static constexpr u32 MaxVertexBufferCount = 32;

class Instance;
class Scheduler;
class DescriptorHeap;

template <typename T>
using VertexInputs = boost::container::static_vector<T, MaxVertexBufferCount>;

struct GraphicsPipelineKey {
    std::array<size_t, MaxShaderStages> stage_hashes;
    std::array<vk::Format, MaxVertexBufferCount> vertex_buffer_formats;
    u32 patch_control_points;
    u32 num_color_attachments;
    std::array<Shader::PsColorBuffer, AmdGpu::NUM_COLOR_BUFFERS> color_buffers;
    std::array<AmdGpu::BlendControl, AmdGpu::NUM_COLOR_BUFFERS> blend_controls;
    std::array<vk::ColorComponentFlags, AmdGpu::NUM_COLOR_BUFFERS> write_masks;
    AmdGpu::ColorBufferMask cb_shader_mask;
    AmdGpu::ColorControl::LogicOp logic_op;
    u8 num_samples;
    u8 depth_samples;
    std::array<u8, AmdGpu::NUM_COLOR_BUFFERS> color_samples;
    u32 mrt_mask;
    struct {
        AmdGpu::DepthBuffer::ZFormat z_format : 2;
        AmdGpu::DepthBuffer::StencilFormat stencil_format : 1;
        u32 depth_clamp_enable : 1;
    };
    struct {
        AmdGpu::PrimitiveType prim_type : 5;
        AmdGpu::PolygonMode polygon_mode : 2;
        AmdGpu::ClipSpace clip_space : 1;
        AmdGpu::ProvokingVtxLast provoking_vtx_last : 1;
        u32 depth_clip_enable : 1;
    };

    GraphicsPipelineKey() {
        std::memset(this, 0, sizeof(*this));
    }

    bool operator==(const GraphicsPipelineKey& key) const noexcept {
        return std::memcmp(this, &key, sizeof(key)) == 0;
    }

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

class GraphicsPipeline : public Pipeline {
public:
    struct SerializationSupport {
        VertexInputs<vk::VertexInputAttributeDescription> vertex_attributes{};
        VertexInputs<vk::VertexInputBindingDescription> vertex_bindings{};
        VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors{};
        vk::PipelineMultisampleStateCreateInfo multisampling{};
        std::vector<u32> tcs{};
        std::vector<u32> tes{};
        std::vector<u32> fragment{};
        // Descriptor count per image, in stage then declaration order. Recomputing these at
        // preload would need the user data, which is not available during deserialization, and
        // getting them wrong shifts every later binding index. See BuildDescSetLayout.
        std::vector<u32> image_binding_counts{};

        void Serialize(Serialization::Archive& ar) const;
        bool Deserialize(Serialization::Archive& ar);
    };

    /// Descriptor set layout bindings of every stage, in unified binding order.
    using DescSetBindings = boost::container::small_vector<vk::DescriptorSetLayoutBinding, 32>;

    /// `derived` (SHADPS4_ASYNC_PIPELINES): the result of DeriveStageInputs, computed on the
    /// lookup thread. With it the constructor reads no Info field except the immutable
    /// stage/pgm_hash (debug name), so it can run on the pipeline compiler thread.
    GraphicsPipeline(const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
                     const Shader::Profile& profile, const GraphicsPipelineKey& key,
                     vk::PipelineCache pipeline_cache,
                     std::span<const Shader::Info*, MaxShaderStages> stages,
                     std::span<const Shader::RuntimeInfo, MaxShaderStages> runtime_infos,
                     std::shared_ptr<const Shader::Gcn::FetchShaderData> fetch_shader,
                     std::span<const vk::ShaderModule> modules, SerializationSupport& sdata,
                     bool preloading, const DescSetBindings* derived = nullptr,
                     u32 native_records_binding = NoNativeRecords, bool native_bindless = false,
                     bool refusable = false);
    ~GraphicsPipeline();

    static constexpr u32 NoNativeRecords = ~0u;

    /// SHADPS4_ASYNC_PIPELINES: the parts of construction that read the stage Infos' per-draw
    /// state (user data behind image sharps and fetch V#s), run on the lookup thread. Fills
    /// sdata's vertex inputs and image binding counts and returns the descriptor bindings.
    static DescSetBindings DeriveStageInputs(
        const Instance& instance, std::span<const Shader::Info*, MaxShaderStages> stages,
        std::span<const Shader::RuntimeInfo, MaxShaderStages> runtime_infos,
        const Shader::Gcn::FetchShaderData* fetch_shader, SerializationSupport& sdata);

    /// False only for a preloaded or `refusable` pipeline the driver refused (the constructor
    /// logged it).
    bool IsCreated() const noexcept {
        return static_cast<bool>(pipeline);
    }

    const Shader::Gcn::FetchShaderData* GetFetchShader() const noexcept {
        return fetch_shader.get();
    }

    const GraphicsPipelineKey& GetGraphicsKey() const {
        return key;
    }

    /// Gets the attributes and bindings for vertex inputs.
    template <typename Attribute, typename Binding>
    void GetVertexInputs(VertexInputs<Attribute>& attributes, VertexInputs<Binding>& bindings,
                         VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
                         VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0,
                         u32 step_rate_1, const Shader::BoundStage& vs) const;

    /// CP split FRONT_DERIVE: GetVertexInputs plus BufferCache::BindVertexBuffers' range merge
    /// as one record block (cp_split.h), the same function on either thread. Reads the fetch
    /// shader, `vs` (one V# per attribute through its user data) and the two step rates. Returns
    /// the block's bytes.
    u32 DeriveVertexInputs(AmdGpu::CpSplit::VertexInputsDerived& out, u32 step_rate_0,
                           u32 step_rate_1, const Shader::BoundStage& vs) const;

    /// SHADPS4_DRAW_REPEAT (Rasterizer::DrawRepeatMatch, back thread only): the static part of
    /// the repeat test's eligibility (0xff unknown, 0 never, 1 eligible, 2 eligible while user
    /// clip planes are off), its first ineligibility reason, and the adaptive gate.
    mutable u8 repeat_static = 0xff;
    mutable u8 repeat_why = 0;
    /// SHADPS4_DRAW_REPEAT_FLAT_CENSUS: ineligible only because of flat-buffer (SRT) bindings.
    mutable u8 repeat_flat_only = 0;
    /// First special BufferType that made the pipeline ineligible (census).
    mutable u8 repeat_special_type = 0;
    mutable u8 repeat_miss_run = 0;
    mutable u8 repeat_skip = 0;
    /// SHADPS4_PB_CENSUS (Rasterizer::PbCensusEnd, back thread only): static class bits
    /// (0xff unknown; 1 DMA, 2 written buffer, 4 formatted buffer, 8 special buffer other than
    /// the clip-plane slot, 16 clip-plane slot, 32 written image), the previous draw of this
    /// pipeline (census index, pure) and its binding inputs.
    mutable u8 pb_static = 0xff;
    mutable u8 pb_last_pure = 0;
    mutable u64 pb_last_draw = 0;
    mutable std::vector<u32> pb_key;
    /// SHADPS4_NATIVE_PASS (notes/native-renderer-2026-10-09.md sec. 3): the native batch
    /// variant of this pipeline (front thread writes once, back reads), whether the front has
    /// decided eligibility, and for a variant itself the records binding.
    mutable std::atomic<const GraphicsPipeline*> native_variant{nullptr};
    mutable bool native_checked = false;
    u32 native_records_binding = NoNativeRecords;
    /// SHADPS4_NATIVE_BINDLESS: a colour variant whose PS samples the global arrays (set 1).
    bool native_bindless = false;
    bool native_bindless_vs = false; ///< q3: the VS samples bindless too (after the FS's slots)
    /// SHADPS4_NATIVE_COLOUR (notes/colour-pass-2026-10-09.md sec. 3): textured native variants
    /// by vertex description. The front claims a slot by storing its key (front-only writer);
    /// the compiler thread publishes the variant; the back reads key then variant (acquire).
    struct NativeColourSlot {
        std::atomic<u64> key{0};
        std::atomic<const GraphicsPipeline*> variant{nullptr};
        /// SHADPS4_NATIVE_SEGB census: 2 = the claimed build was refused (no variant will come).
        std::atomic<u8> state{0};
        /// l4g (SHADPS4_NATIVE_SET_VARIANTS): a variant without push descriptors (descriptor
        /// set at set 0); only the A2 stretch translator uses it.
        std::atomic<const GraphicsPipeline*> set_variant{nullptr};
    };
    static constexpr u32 NativeColourSlots = 4;
    mutable std::array<NativeColourSlot, NativeColourSlots> native_colour{};
    mutable u8 native_colour_static = 0; ///< front: 0 undecided, 1 eligible, 2 not
    mutable u8 native_colour_untextured = 0; ///< l4b census: 1 = refused as "untextured"
    /// Native census (sec. 14): the last reason the front logged for no colour variant.
    mutable std::atomic<const char*> native_colour_why{nullptr};
    u32 native_vertex_binding = NoNativeRecords; ///< a colour variant: its vertex SSBO binding
    u32 native_table_binding = NoNativeRecords;  ///< SHADPS4_NATIVE_GPU_DIRECT: the batch table
    u32 native_vertex2_binding = NoNativeRecords;   ///< q53 (SHADPS4_A2_VBUF2): the second vertex SSBO
    u32 native_pt_debug_binding = NoNativeRecords;  ///< SHADPS4_NATIVE_PT=2: the compare SSBO
    const GraphicsPipeline* NativeColourSetVariant(u64 key) const noexcept {
        for (const auto& slot : native_colour) {
            if (slot.key.load(std::memory_order_acquire) == key) {
                return slot.set_variant.load(std::memory_order_acquire);
            }
        }
        return nullptr;
    }
    const GraphicsPipeline* NativeColourVariant(u64 key) const noexcept {
        for (const auto& slot : native_colour) {
            if (slot.key.load(std::memory_order_acquire) == key) {
                return slot.variant.load(std::memory_order_acquire);
            }
        }
        return nullptr;
    }

private:
    /// One binding per buffer and sampler, one per image with its descriptor count (recorded
    /// in sdata.image_binding_counts on first creation, replayed from it while preloading).
    static void CollectDescSetBindings(std::span<const Shader::Info* const> stages,
                                       SerializationSupport& sdata, bool preloading,
                                       DescSetBindings& bindings);
    void BuildDescSetLayout(std::span<const vk::DescriptorSetLayoutBinding> bindings);

private:
    GraphicsPipelineKey key;
    std::shared_ptr<const Shader::Gcn::FetchShaderData> fetch_shader;
};

struct ClipDistanceShaderKey {
    std::array<std::tuple<u8, u8>, 8> clip_locations;

    bool operator==(const ClipDistanceShaderKey& key) const noexcept {
        return std::memcmp(this, &key, sizeof(key)) == 0;
    }
};

} // namespace Vulkan

template <>
struct std::hash<Vulkan::GraphicsPipelineKey> {
    std::size_t operator()(const Vulkan::GraphicsPipelineKey& key) const noexcept {
        return XXH3_64bits(&key, sizeof(key));
    }
};

template <>
struct std::hash<Vulkan::ClipDistanceShaderKey> {
    std::size_t operator()(const Vulkan::ClipDistanceShaderKey& key) const noexcept {
        return XXH3_64bits(&key, sizeof(key));
    }
};
