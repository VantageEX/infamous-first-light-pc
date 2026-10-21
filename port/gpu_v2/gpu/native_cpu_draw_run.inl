// SPDX-License-Identifier: GPL-2.0-or-later
// CPU resource bundle: retain each original indexed draw and its original fetch
// addresses. Resolve shared resources once and resolve geometry through BufferCache.
std::size_t Rasterizer::TryNativeCpuRun(const GraphicsPipeline* pipeline,
                                       std::span<const u32> following_commands,
                                       std::size_t packet_budget) {
    namespace ng=iss::native_geometry;
    static const bool enabled=[] {const char* p=std::getenv("ISS_NATIVE_CPU_DRAWS");return p&&p[0]=='1';}();
    static const bool verified=[] {const char* p=std::getenv("ISS_CPU_BUNDLE_SPV_VERIFIED");return p&&p[0]=='1';}();
    static const bool index_once=[] {const char* p=std::getenv("ISS_CPU_BUNDLE_INDEX_ONCE");return p&&p[0]=='1';}();
    if(!enabled || !verified || cp_split_enabled || draw_pipe_state || OnDrawWorker() ||
       DrawQueue()!=0 || iss::command_trace::enabled || IsDumping() || host_markers_enabled ||
       guest_markers_enabled || GpuCrashDiag::Enabled() || EmulatorSettings.IsPatchShaders() ||
       scheduler.HasPriorPendingOperations())return 0;
    auto& regs=DrawRegs();
    if(regs.num_instances.NumInstances()!=1 || u32(regs.primitive_type)!=4 ||
       regs.index_buffer_type.raw!=0 || regs.enable_primitive_restart ||
       regs.clipper_control.user_clip_plane_enable || regs.depth_render_control.depth_clear_enable ||
       regs.depth_render_control.stencil_clear_enable)return 0;
    if(std::hash<GraphicsPipelineKey>{}(pipeline->GetGraphicsKey())!=0x6794975e42e32374ULL ||
       pipeline->GetGraphicsKey().stage_hashes[u32(Shader::SwStage::Vertex)]!=0xbf5de0e198ULL)return 0;
    const auto& vs=pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fs=pipeline->GetStage(Shader::SwStage::Fragment);
    const auto& fetch=pipeline->GetFetchShader();
    if(vs.pgm_hash!=0x0beb7156 || fs.pgm_hash!=0x6353e48b || fetch.size!=48 ||
       fetch.attributes.size()!=3 || vs.buffers.size()!=3 || fs.buffers.size()!=5 ||
       !vs.images.empty() || fs.images.size()!=5)return 0;
    const auto [vertex_offset,instance_offset]=GetDrawOffsets(regs,vs,fetch);
    if(vertex_offset || instance_offset)return 0;
    for(const auto* stage:pipeline->GetStages())if(stage) {
        if((stage!=&vs && stage!=&fs) || stage->uses_dma || !stage->fmasks.empty())return 0;
        for(const auto& desc:stage->buffers)if(desc.is_written ||
            (desc.IsSpecial() && desc.buffer_type!=Shader::BufferType::ClipPlanes))return 0;
        for(const auto& desc:stage->images)if(desc.is_written || desc.is_atomic)return 0;
    }
    const auto pointer_word=reinterpret_cast<const u32*>(&regs.vs_program.user_data[2])-regs.reg_array.data();
    const u64 first_fetch=u64(regs.vs_program.user_data[2])|(u64(regs.vs_program.user_data[3])<<32);
    const ng::RunInputs input{regs.reg_array,AmdGpu::Regs::ShRegWordOffset,
        AmdGpu::Regs::UconfigRegWordOffset,std::size_t(pointer_word),regs.index_buffer_type.raw,
        regs.num_instances.num_instances,{first_fetch,regs.index_base_address.Address<VAddr>(),
        regs.num_indices,regs.max_index_size,regs.draw_initiator,0}};
    auto run=ng::CollectIndexedRun(following_commands,input,64,packet_budget);
    if(run.draws.size()<2)return 0;
    static thread_local u64 attempts{},groups{},saved{},fallbacks{};
    ++attempts;
    const auto reject=[&](const char* reason)->std::size_t {
        ++fallbacks;
        if(fallbacks<=8 || (fallbacks&8191)==0)std::fprintf(stderr,
            "NATIVE_CPU_DRAWS fallback=%s attempts=%llu fallback_count=%llu\n",reason,
            static_cast<unsigned long long>(attempts),static_cast<unsigned long long>(fallbacks));
        return 0;
    };
    const auto map_epoch=memory->MappingEpoch(),buffer_generation=buffer_cache.MappingGeneration();
    const auto upload_epoch=VideoCore::CurrentUploadEpoch(),host_generation=VideoCore::CurrentHostWriteGen();
    const auto tick=scheduler.CurrentTick();
    const auto unchanged=[&] {
        return map_epoch==memory->MappingEpoch() && buffer_generation==buffer_cache.MappingGeneration() &&
            upload_epoch==VideoCore::CurrentUploadEpoch() && host_generation==VideoCore::CurrentHostWriteGen() &&
            tick==scheduler.CurrentTick() && !scheduler.HasPriorPendingOperations();
    };
    // Cache only uniqueness of physical alias metadata, never content/dirty state.
    // A page-sized proof can cover several neighbouring descriptor lists.
    // Linux: ~600 KB, allocated on first use (glibc places static TLS in every thread's stack).
    static thread_local std::unique_ptr<ng::BackingAliasCache<8192>> alias_proofs_storage;
    if(!alias_proofs_storage)alias_proofs_storage=std::make_unique<ng::BackingAliasCache<8192>>();
    auto& alias_proofs=*alias_proofs_storage;
    std::vector<Core::MemoryManager::BackingSpan> read_backings;
    read_backings.reserve(run.draws.size()*5+16);
    std::vector<iss::g4::NegativeImageSnapshot> image_proofs;
    image_proofs.reserve(run.draws.size()*5+16);
    const auto cpu_read=[&](u64 address,u64 size)->bool {
        if(!address || !size || size>UINT32_MAX || address>UINT64_MAX-size ||
           !memory->IsValidGpuMapping(address,size) || buffer_cache.HasGpuData(address,size))
            return false;
        const auto start=Common::AlignDown(address,u64{4096}),end=Common::AlignUp(address+size,u64{4096});
        const auto owner=reinterpret_cast<std::uintptr_t>(memory);
        const auto* proof=alias_proofs.Find(owner,start,end,map_epoch,buffer_generation);
        if(!proof) {
            const auto aliases=memory->EnumerateBackingAliases(start,end-start);
            const bool unique=aliases && aliases->size()==1 && aliases->front().guest_address==start &&
                aliases->front().size==end-start && unchanged();
            alias_proofs.Store({owner,start,end,map_epoch,buffer_generation,
                unique?aliases->front().physical_start:0,unique});
            proof=alias_proofs.Find(owner,start,end,map_epoch,buffer_generation);
        }
        if(!proof || !proof->unique)return false;
        iss::g4::NegativeImageSnapshot negative{iss::g4::NegativeImageResult::NoRegisteredPage,
            proof->image_epoch,start,std::size_t(end-start)};
        if(!proof->no_images || !texture_cache.NegativeImageSnapshotCurrent(negative)) {
            negative=texture_cache.ProbeNoRegisteredImage(start,end-start);
            auto updated=*proof;updated.image_epoch=negative.epoch;
            updated.no_images=negative.result==iss::g4::NegativeImageResult::NoRegisteredPage;
            alias_proofs.Store(updated);proof=alias_proofs.Find(owner,start,end,map_epoch,buffer_generation);
            if(!proof || !proof->no_images)return false;
        }
        image_proofs.push_back(negative);
        read_backings.push_back({proof->physical+address-start,size});return unchanged();
    };
    for(const auto* stage:pipeline->GetStages())if(stage)
        for(const auto& desc:stage->buffers)if(!desc.IsSpecial()) {
            const auto sharp=desc.GetSharp(*stage);
            if(sharp.GetSize()>16384 || !cpu_read(sharp.base_address,sharp.GetSize()))return reject("material");
        }
    struct Geometry {std::array<AmdGpu::Buffer,3> streams;std::size_t first_read{};};
    std::vector<Geometry> geometry(run.draws.size());
    std::vector<ng::MemoryRead> reads;reads.reserve(run.draws.size()*4);
    constexpr std::array<u32,3> formats{418732,299948,193068};
    std::vector<u16> indices;
    const auto validate_draw=[&](std::size_t n)->const char* {
        const auto& source=run.draws[n];auto& draw=geometry[n];
        std::array<u32,12> raw{};
        if(!cpu_read(source.fetch_descriptors,sizeof(raw)))return "fetch_table";
        memory->CopySparseMemory(source.fetch_descriptors,reinterpret_cast<u8*>(raw.data()),sizeof(raw));
        draw.first_read=reads.size();
        for(u32 i=0;i<3;++i) {
            const auto& attribute=fetch.attributes[i];
            if(attribute.sgpr_base!=2 || attribute.dword_offset>8 || attribute.instance_data!=0)return "fetch_layout";
            auto& sharp=draw.streams[i];std::memcpy(&sharp,raw.data()+attribute.dword_offset,sizeof(sharp));
            sharp.base_address+=attribute.inst_offset;
            if(attribute.data_format){sharp.data_format=attribute.data_format;sharp.num_format=attribute.num_format;}
            const auto words=std::bit_cast<std::array<u32,4>>(sharp);
            const auto size=sharp.GetSize();
            if(sharp.GetStride()!=8 || words[3]!=formats[i])return "vertex_layout";
            if(!size || size>64*1024*1024)return "vertex_size";
        }
        if(!source.index_count || source.index_count>131072)return "index_count";
        if(!cpu_read(source.index_address,u64(source.index_count)*2))return "index_cpu_proof";
        indices.resize(source.index_count);
        memory->CopySparseMemory(source.index_address,reinterpret_cast<u8*>(indices.data()),indices.size()*2);
        // All streams use the same unchanged zero-base uint16 indices. Prefix
        // validation depends only on their maximum; each stream still checks its
        // own stride, fetch width and declared bounds through the original helper.
        const u16 maximum=index_once?*std::max_element(indices.begin(),indices.end()):0;
        const auto bounds_indices=index_once?std::span<const u16>(&maximum,1):std::span<const u16>(indices);
        // Wider merged bindings must never make an originally invalid vertex fetch valid.
        for(u32 i=0;i<3;++i) {
            const auto& sharp=draw.streams[i];
            const auto bits=AmdGpu::NumBitsPerBlock(sharp.GetDataFmt());
            const auto used=ng::UsedVertexPrefix(bounds_indices,sharp.GetStride(),(bits+7)/8,sharp.GetSize());
            if(!used)return "index_bounds_or_fetch_width";
            if(!cpu_read(sharp.base_address,*used))return "vertex_cpu_proof";
            reads.push_back({sharp.base_address,*used});
        }
        reads.push_back({source.index_address,u64(source.index_count)*2});
        return nullptr;
    };
    for(std::size_t n=0;n<run.draws.size();++n) {
        const auto reads_before=reads.size(),backings_before=read_backings.size(),images_before=image_proofs.size();
        if(const auto reason=validate_draw(n)) {
            if(n<2)return reject(reason);
            reads.resize(reads_before);read_backings.resize(backings_before);
            image_proofs.resize(images_before);
            run=ng::PrefixIndexedRun(run,following_commands,n);
            if(run.draws.size()<2)return reject("prefix_boundary");
            geometry.resize(n);break;
        }
    }
    const auto plan=ng::PlanReadSnapshot(reads);
    if(!plan || !unchanged())return reject("geometry_plan");
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> first_buffers;
    pipeline->GetVertexInputs(attributes,bindings,divisors,first_buffers,
        regs.vgt_instance_step_rate_0,regs.vgt_instance_step_rate_1);
    if(attributes.size()!=3 || bindings.size()!=3 || first_buffers.size()!=3)return reject("host_layout");
    for(u32 i=0;i<3;++i)if(std::bit_cast<std::array<u32,4>>(first_buffers[i])!=
        std::bit_cast<std::array<u32,4>>(geometry.front().streams[i]))return reject("fetch_parity");
    PrepareRenderState(pipeline);
    for(u32 i=0;i<AmdGpu::NUM_COLOR_BUFFERS;++i)if(cb_descs[i].first &&
        texture_cache.IsMetaCleared(regs.color_buffers[i].CmaskAddress(),cb_descs[i].second.view_info.range.base.layer)) {
        ResetBindings(false);return reject("clear");
    }
    if(db_desc.first && texture_cache.IsMetaCleared(regs.depth_htile_data_base.GetAddress(),
        db_desc.second.view_info.range.base.layer)){ResetBindings(false);return reject("depth_clear");}
    const auto targets=bound_images.size();
    if(!BindResources(pipeline)){ResetBindings(false);return reject("resources");}
    for(std::size_t i=targets;i<bound_images.size();++i) {
        const auto& image=texture_cache.GetImage(bound_images[i]);
        if(image.binding.is_target || image.binding.force_general || image.binding.needs_rebind) {
            ResetBindings(false);return reject("sampled_feedback");
        }
        const auto backing=memory->DescribeBacking(image.info.guest_address,image.info.guest_size);
        if(!backing || backing->size!=image.info.guest_size){ResetBindings(false);return reject("image_backing");}
        read_backings.push_back(*backing);
        for(std::size_t j=0;j<targets;++j)if(image.backing==texture_cache.GetImage(bound_images[j]).backing) {
            ResetBindings(false);return reject("image_alias");
        }
    }
    for(std::size_t i=0;i<targets;++i) {
        const auto& image=texture_cache.GetImage(bound_images[i]);
        const auto target=memory->DescribeBacking(image.info.guest_address,image.info.guest_size);
        if(!target || target->size!=image.info.guest_size){ResetBindings(false);return reject("target_backing");}
        for(const auto& source:read_backings)if(source.physical_start<target->physical_start+target->size &&
            target->physical_start<source.physical_start+source.size){ResetBindings(false);return reject("target_alias");}
    }
    struct Resolved {const VideoCore::Buffer* buffer{};u64 offset{};};
    std::vector<Resolved> resolved(plan->ranges.size());
    const bool reuse=Common::Perf::GetTuning().reuse_buffer_checks.load(std::memory_order_relaxed);
    for(std::size_t i=0;i<plan->ranges.size();++i) {
        const auto& range=plan->ranges[i];bool known=false;
        if(!cpu_read(range.address,range.size)){ResetBindings(false);return reject("merged_range");}
        std::tie(resolved[i].buffer,resolved[i].offset)=buffer_cache.ObtainBuffer(
            range.address,u32(range.size),false,false,reuse?&known:nullptr);
        if(resolved[i].buffer!=&buffer_cache.GetStreamBuffer()) {
            if(known)Common::Perf::Add(Common::Perf::BufferChecksReused);
            else needs_barrier|=runtime.IsBufferAccessed(resolved[i].buffer,resolved[i].offset,range.size);
        }
    }
    const auto images_current=[&] {
        return std::ranges::all_of(image_proofs,[&](const auto& proof){
            return texture_cache.NegativeImageSnapshotCurrent(proof);});
    };
    if(!unchanged() || !images_current()){ResetBindings(false);return reject("generation_or_flush");}
    struct HostDraw {std::array<vk::Buffer,3> buffers;std::array<vk::DeviceSize,3> offsets,sizes,strides;
        vk::Buffer index;vk::DeviceSize index_offset;u32 count;};
    std::vector<HostDraw> host(run.draws.size());
    const auto resolve=[&](std::size_t request)->std::pair<vk::Buffer,u64> {
        const auto& read=reads[request];
        auto found=std::upper_bound(plan->ranges.begin(),plan->ranges.end(),read.address,
            [](u64 address,const auto& range){return address<range.address;});
        if(found!=plan->ranges.begin()) {--found;const auto i=std::size_t(found-plan->ranges.begin());
            if(read.address-found->address<=found->size &&
               read.size<=found->size-(read.address-found->address))
                return {resolved[i].buffer->Handle(),resolved[i].offset+read.address-found->address};}
        UNREACHABLE_MSG("Native CPU geometry range was not resolved");
    };
    for(std::size_t n=0;n<host.size();++n) {
        auto& draw=host[n];const auto& source=geometry[n];
        for(u32 i=0;i<3;++i){std::tie(draw.buffers[i],draw.offsets[i])=resolve(source.first_read+i);
            draw.sizes[i]=reads[source.first_read+i].size;draw.strides[i]=source.streams[i].GetStride();}
        std::tie(draw.index,draw.index_offset)=resolve(source.first_read+3);draw.count=run.draws[n].index_count;
    }
    const auto render_state=BeginRendering(pipeline);
    if(needs_barrier)runtime.FlushBarriers();
    pipeline->BindResources(set_writes,push_data);UpdateDynamicState(pipeline,true);
    scheduler.BeginRendering(render_state);
    if(attachment_feedback_loop || !unchanged() || !images_current()){
        ResetBindings(false);return reject("late_generation");}
    const auto cmd=scheduler.Recorder();cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,pipeline->Handle());
    if(instance.IsVertexInputDynamicState())cmd.setVertexInputEXT(bindings,attributes);
    // All checks and uploads precede the first draw. No failure can fall back after a prefix.
    for(const auto& draw:host) {
        if(instance.IsVertexInputDynamicState())cmd.bindVertexBuffers(0,3,draw.buffers.data(),draw.offsets.data());
        else cmd.bindVertexBuffers2(0,3,draw.buffers.data(),draw.offsets.data(),draw.sizes.data(),draw.strides.data());
        cmd.bindIndexBuffer(draw.index,draw.index_offset,vk::IndexType::eUint16);
        cmd.drawIndexed(draw.count,1,0,0,0);DebugState.IncDrawCall();
    }
    ResetBindings(false);
    const auto& last=run.draws.back();
    regs.vs_program.user_data[2]=u32(last.fetch_descriptors);regs.vs_program.user_data[3]=u32(last.fetch_descriptors>>32);
    regs.index_base_address.base_addr_lo=u32(last.index_address);regs.index_base_address.base_addr_hi=u32(last.index_address>>32);
    regs.num_indices=last.index_count;regs.max_index_size=last.max_index_size;regs.draw_initiator=last.initiator;
    ++groups;saved+=run.draws.size()-1;
    if(groups==1 || (groups&1023)==0)std::fprintf(stderr,
        "NATIVE_CPU_DRAWS groups=%llu repeated_draw_preparation_removed=%llu attempts=%llu fallback=%llu\n",
        static_cast<unsigned long long>(groups),static_cast<unsigned long long>(saved),
        static_cast<unsigned long long>(attempts),static_cast<unsigned long long>(fallbacks));
    return run.consumed_words;
}
