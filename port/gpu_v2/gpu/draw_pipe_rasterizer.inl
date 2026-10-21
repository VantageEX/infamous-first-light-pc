// SPDX-License-Identifier: GPL-2.0-or-later
// Included in namespace Vulkan. Stage A owns selection; B owns recording between drains.
struct DrawPipeState {
    static constexpr u32 BlockWords=32;
    static constexpr u32 NumBlocks=AmdGpu::Regs::NumRegs/BlockWords;
    std::array<u64,(NumBlocks+63)/64> dirty{};
    std::array<u16,NumBlocks> changed{};
    u32 changed_count{};
    AmdGpu::Regs regs{};
    std::array<AmdGpu::CbDbExtent,AmdGpu::NUM_COLOR_BUFFERS> cb{};
    AmdGpu::CbDbExtent db{};
    const AmdGpu::ComputeProgram* cs{};
    u64 context_changes{};
    int queue_id{};bool async{};
    u32 verify_interval{};
    std::atomic<u64> verify_checks{},draws{},dispatches{},gds_fills{};
    bool pipe_gds{true};
    bool lazy{}; // ISS_DRAW_PIPE_LAZY: skip optional drains (task end, yield, early retire)
    bool write_guard{true};
    bool fast{}; // ISS_DRAW_PIPE_FAST: the worker takes the fast draw paths
    bool defer{}; // ISS_DRAW_PIPE_DEFER: fences and indirect draws are queued, not drained
    bool waits{}; // ISS_DRAW_PIPE_WAITS: waits held in queue order are queued too (slower so far)
    std::atomic<u64> deferred_fences{},deferred_indirect{};
    // ISS_DRAW_PIPE_DEFER: the memory the queued fences will write (A only), by fence sequence;
    // the worker publishes the last sequence it ran.
    struct PendingWrite {u64 seq;VAddr address;u32 value;bool known;};
    static constexpr u32 NumPendingWrites=256;
    std::array<PendingWrite,NumPendingWrites> pending_writes{};
    u64 pending_head{},fence_seq{};
    std::atomic<u64> fences_done{},wait_skips{},wait_drains{};
    void NotePendingWrite(VAddr address,u32 value,bool known){
        pending_writes[pending_head++%NumPendingWrites]={fence_seq,address,value,known};
    }
    /// The wait holds once the worker reaches it: the last queued write to its dword
    /// satisfies it, or (no queued write) memory does now. False: drain and poll.
    bool WaitHoldsInOrder(const AmdGpu::PM4CmdWaitRegMem& wait){
        using W=AmdGpu::PM4CmdWaitRegMem;
        if(wait.mem_space.Value()!=W::MemSpace::Memory)return false;
        const u64 done=fences_done.load(std::memory_order_acquire);
        // Records overwritten before the worker ran them: unknown, drain.
        if(fence_seq-done>NumPendingWrites/2)return false;
        const VAddr address=wait.Address<VAddr>();
        const u64 count=std::min<u64>(pending_head,NumPendingWrites);
        for(u64 i=0;i<count;++i){
            const auto& w=pending_writes[(pending_head-1-i)%NumPendingWrites];
            if(w.seq<=done)break;
            if(w.address==address)return w.known&&wait.TestValue(w.value);
        }
        return wait.TestValue(*wait.Address<const u32*>());
    }
    std::array<u64,256> op_drains{},op_ns{}; // A only: drains forced per PM4 opcode
    std::atomic<u64> guard_commands{};
    std::unique_ptr<iss::DrawPipe> queue;
    FILE* log{};u64 last_report{};
    void Mark(u32 first,u32 count) {
        if(!count)return;ASSERT(first<AmdGpu::Regs::NumRegs && count<=AmdGpu::Regs::NumRegs-first);
        const u32 last=(first+count-1)/BlockWords;
        for(u32 block=first/BlockWords;block<=last;++block){const u64 mask=u64{1}<<(block%64);
            if(!(dirty[block/64]&mask)){dirty[block/64]|=mask;changed[changed_count++]=static_cast<u16>(block);}}
    }
    void Clear(){dirty.fill(0);changed_count=0;}
    ~DrawPipeState(){queue.reset();if(log)std::fclose(log);}
};
namespace {
thread_local DrawPipeState* active_draw_pipe{};
constexpr u32 PipeAlign(u32 x){return (x+7)&~7u;}
constexpr u32 DrainOptional=0x100; // Reason flag: the drain only keeps the old schedule
bool DrawPipeBusyLazy(const DrawPipeState* s){return s&&s->lazy&&!s->queue->Idle();}
struct PipePacket {
    const Pipeline* pipeline;
    u32 blocks,stages,index_offset;
    u32 num_changed; // ISS_DRAW_PIPE_FAST: register dwords changed since the last posted draw
    std::array<u32,32> changed;
    u32 fence_op,fence_words,fence_pipe; // ISS_DRAW_PIPE_DEFER fence packet (words follow)
    u64 fence_seq;
    bool indirect; std::array<u64,5> ind; // address, offset, stride, max count, count address
    bool compute,indexed,verify,async;
    int queue_id;
    u64 context_changes;
    std::array<AmdGpu::CbDbExtent,AmdGpu::NUM_COLOR_BUFFERS> cb;
    AmdGpu::CbDbExtent db;
};
struct PipeStage {const Shader::Info* info;VAddr base;u32 ud_size,flat_size,reads,reserved;};
struct PipeGdsFill {VAddr address;u32 size,value;};
bool PipeFlag(const char* name){const char* p=std::getenv(name);return p&&p[0]=='1';}
}
void Rasterizer::InitializeDrawPipe(){
    if(!PipeFlag("ISS_DRAW_PIPE"))return;
    if(host_markers_enabled||guest_markers_enabled||iss::command_trace::enabled||
       PipeFlag("ISS_COMPUTE_PHASES")||PipeFlag("ISS_FEEDBACK_EARLY_SUBMIT")){
        std::fprintf(stderr,"ISS_DRAW_PIPE fallback: incompatible marker/legacy trace/phase mode\n");return;
    }
    draw_pipe_state=std::make_unique<DrawPipeState>();auto& s=*draw_pipe_state;
    s.Mark(0,AmdGpu::Regs::NumRegs);
    if(const char* p=std::getenv("ISS_DRAW_PIPE_VERIFY"))s.verify_interval=std::strtoul(p,nullptr,10);
    if(const char* p=std::getenv("ISS_DRAW_PIPE_GDS"))s.pipe_gds=p[0]!='0';
    s.lazy=PipeFlag("ISS_DRAW_PIPE_LAZY");
    s.fast=PipeFlag("ISS_DRAW_PIPE_FAST");
    s.defer=PipeFlag("ISS_DRAW_PIPE_DEFER");
    s.waits=PipeFlag("ISS_DRAW_PIPE_WAITS");
    if(const char* p=std::getenv("ISS_DRAW_PIPE_WRITE_GUARD"))s.write_guard=p[0]!='0';
    if(const char* p=std::getenv("ISS_DRAW_PIPE_LOG")){s.log=std::fopen(p,"w");if(s.log){std::setvbuf(s.log,nullptr,_IONBF,0);std::fputs("time_ns,packets,completed,draws,dispatches,worker_busy_ns,producer_wait_ns,drains,verify_checks,gds_fills,drain_packet_ns,drain_commands_ns,drain_submission_ns,drain_yield_ns,drain_fault_ns,drain_resource_ns,drain_fence_ns,drain_indirect_ns\n",s.log);}}
    s.queue=std::make_unique<iss::DrawPipe>(&Rasterizer::RunDrawPipePacket,this);
    std::fprintf(stderr,"ISS_DRAW_PIPE enabled: ordered direct draws/dispatches; fences and indirects drain; verify=%u lazy=%d write_guard=%d\n",s.verify_interval,int(s.lazy),int(s.write_guard));
}
bool Rasterizer::OnDrawWorker() const {return active_draw_pipe&&active_draw_pipe==draw_pipe_state.get();}
bool Rasterizer::UseDrawPipe() const {
    return draw_pipe_state && !OnDrawWorker() && !IsDumping() &&
        !Common::Perf::GetTuning().async_pipelines.load(std::memory_order_relaxed)&&
        !Common::Perf::GetTuning().asc_queue.load(std::memory_order_relaxed);
}
void Rasterizer::DrainDrawPipe(u32 reason){
    if(!draw_pipe_state||iss::DrawPipe::OnWorker()||std::this_thread::get_id()!=liverpool->GetGpuCommandProcessorThread())return;
    if(reason&DrainOptional){if(draw_pipe_state->lazy)return;reason&=~DrainOptional;}
    draw_pipe_state->queue->Drain(reason);
}
void Rasterizer::DrainDrawPipeForIdle(){
    // No guest signal is deferred by this version. All old fence paths run on A after drain.
    DrainDrawPipe(iss::DrawPipe::Submission);
}
bool Rasterizer::PrepareDrawPipeFault(bool is_write){
    if(OnDrawWorker())return true;
    if(!liverpool->IsCurrentCpThread()){
        // A protected guest page can still be clean before queued B bindings register it.
        // Serialize a write before InvalidateRegion takes any region lock/unprotects it.
        // Only A may read the non-atomic producer head or drain the SPSC queue.
        // ISS_DRAW_PIPE_WRITE_GUARD=0 is an unproven hypothesis (diagnostic only): a guest
        // write racing a submitted, unrecorded draw would also race the GPU on the PS4.
        if(is_write&&draw_pipe_state&&draw_pipe_state->write_guard&&!draw_pipe_state->queue->Idle()){
            draw_pipe_state->guard_commands.fetch_add(1,std::memory_order_relaxed);
            liverpool->SendCommand<true>([this]{DrainDrawPipe(iss::DrawPipe::Fault);});
        }
        return false;
    }
    DrainDrawPipe(iss::DrawPipe::Fault);return true;
}
AmdGpu::Regs& Rasterizer::DrawRegs() const {return OnDrawWorker()?active_draw_pipe->regs:liverpool->regs;}
const AmdGpu::ComputeProgram& Rasterizer::DrawCsRegs() const {return OnDrawWorker()?*active_draw_pipe->cs:liverpool->GetCsRegs();}
const std::array<AmdGpu::CbDbExtent,AmdGpu::NUM_COLOR_BUFFERS>& Rasterizer::DrawCbExtents() const {return OnDrawWorker()?active_draw_pipe->cb:liverpool->last_cb_extent;}
const AmdGpu::CbDbExtent& Rasterizer::DrawDbExtent() const {return OnDrawWorker()?active_draw_pipe->db:liverpool->last_db_extent;}
int Rasterizer::DrawQueue() const {return OnDrawWorker()?active_draw_pipe->queue_id:liverpool->CurrentQueue();}
bool Rasterizer::DrawAsync() const {
    if (OnDrawWorker()) return active_draw_pipe->async;
    if (cp_split_enabled && split_thread_state.rasterizer == this)
        return split_thread_state.async;
    return async_compute_translation;
}
u64 Rasterizer::DrawContextChanges() const {return OnDrawWorker()?active_draw_pipe->context_changes:liverpool->ContextRegChanges();}
bool Rasterizer::FilterDrawPasses() const {
    const auto& r=liverpool->regs;using Op=AmdGpu::ColorControl::OperationMode;
    if(r.color_control.mode==Op::EliminateFastClear||r.color_control.mode==Op::FmaskDecompress||r.color_control.mode==Op::Resolve||r.primitive_type==AmdGpu::PrimitiveType::None)return false;
    const bool depth=r.depth_render_override.force_z_dirty&&r.depth_render_override.force_z_valid&&r.depth_buffer.DepthValid()&&r.depth_buffer.DepthWriteValid()&&r.depth_buffer.DepthAddress()!=r.depth_buffer.DepthWriteAddress();
    const bool stencil=r.depth_render_override.force_stencil_dirty&&r.depth_render_override.force_stencil_valid&&r.depth_buffer.StencilValid()&&r.depth_buffer.StencilWriteValid()&&r.depth_buffer.StencilAddress()!=r.depth_buffer.StencilWriteAddress();
    return !(r.color_control.mode==Op::Disable&&(depth||stencil));
}
bool Rasterizer::BeforeDrawPipePacket(u32 op,const u32* words,u32 count,int asc_pipe_id){
    if(!draw_pipe_state)return false;auto& s=*draw_pipe_state;using O=AmdGpu::PM4ItOpcode;
    const auto opcode=static_cast<O>(op);
    const auto drain=[&](u32 reason){
        const u64 before=s.queue->wait_ns;DrainDrawPipe(reason);
        if(const u64 ns=s.queue->wait_ns-before){++s.op_drains[op&255];s.op_ns[op&255]+=ns;}
    };
    // ISS_DRAW_PIPE_DEFER: the fence goes to the worker behind the draws it follows, which
    // signals it once they are recorded (what draining before signaling here achieves).
    const auto defer_fence=[&](u32 pipe_id){
        if(!s.defer||!UseDrawPipe()||liverpool->CpSplitActive()||count*4>iss::DrawPipe::MaxPayload/2)return false;
        using DS=AmdGpu::DataSelect;
        ++s.fence_seq;
        const auto note_data=[&](VAddr address,DS sel,u32 lo,u32 hi){
            switch(sel){
            case DS::None:break;
            case DS::Data32Low:s.NotePendingWrite(address,lo,true);break;
            case DS::Data64:s.NotePendingWrite(address,lo,true);s.NotePendingWrite(address+4,hi,true);break;
            default:s.NotePendingWrite(address,0,false);s.NotePendingWrite(address+4,0,false);break;
            }
        };
        if(opcode==O::EventWriteEos){
            const auto& e=*reinterpret_cast<const AmdGpu::PM4CmdEventWriteEos*>(words);
            s.NotePendingWrite(e.Address<VAddr>(),e.DataDWord(),
                               e.command.Value()==AmdGpu::PM4CmdEventWriteEos::Command::SignalFence);
        }else if(opcode==O::EventWriteEop){
            const auto& e=*reinterpret_cast<const AmdGpu::PM4CmdEventWriteEop*>(words);
            note_data(reinterpret_cast<VAddr>(e.Address<u32>()),e.data_sel.Value(),e.data_lo,e.data_hi);
        }else if(opcode==O::ReleaseMem){
            const auto& e=*reinterpret_cast<const AmdGpu::PM4CmdReleaseMem*>(words);
            note_data(e.Address<VAddr>(),e.data_sel.Value(),e.data_lo,e.data_hi);
        }
        u8* out=s.queue->Begin(static_cast<u32>(sizeof(PipePacket)+PipeAlign(count*4)));
        auto& p=*reinterpret_cast<PipePacket*>(out);p={};p.fence_op=op;p.fence_words=count;p.fence_pipe=pipe_id;p.fence_seq=s.fence_seq;
        p.async=async_compute_translation;p.queue_id=liverpool->CurrentQueue();
        std::memcpy(out+sizeof(PipePacket),words,count*4);s.queue->Commit();
        s.deferred_fences.fetch_add(1,std::memory_order_relaxed);return true;
    };
    u32 base=0;
    switch(opcode){
    case O::EventWriteEos:case O::EventWriteEop:
        if(asc_pipe_id<0&&defer_fence(0))return true;
        drain(iss::DrawPipe::Packet);return false;
    case O::ReleaseMem:
        if(asc_pipe_id>=0&&defer_fence(static_cast<u32>(asc_pipe_id)))return true;
        drain(iss::DrawPipe::Packet);return false;
    case O::WaitRegMem:
        // Held once the worker reaches it (ordered behind the fences that satisfy it): the
        // worker applies its effect on the upload epoch there. Otherwise drain and poll.
        if(s.defer&&s.waits&&count>=7&&UseDrawPipe()&&!liverpool->CpSplitActive()&&
           s.WaitHoldsInOrder(*reinterpret_cast<const AmdGpu::PM4CmdWaitRegMem*>(words))){
            u8* out=s.queue->Begin(static_cast<u32>(sizeof(PipePacket)+PipeAlign(count*4)));
            auto& p=*reinterpret_cast<PipePacket*>(out);p={};p.fence_op=op;p.fence_words=count;
            p.async=async_compute_translation;p.queue_id=liverpool->CurrentQueue();
            std::memcpy(out+sizeof(PipePacket),words,count*4);s.queue->Commit();
            s.wait_skips.fetch_add(1,std::memory_order_relaxed);return true;
        }
        s.wait_drains.fetch_add(1,std::memory_order_relaxed);
        drain(iss::DrawPipe::Packet);return false;
    case O::DrawIndirect:case O::DrawIndexIndirect:
        // DrawIndirect queues them (ISS_DRAW_PIPE_DEFER) or drains itself.
        if(s.defer&&asc_pipe_id<0)return false;
        drain(iss::DrawPipe::Packet);return false;
    case O::SetConfigReg:base=AmdGpu::Regs::ConfigRegWordOffset;break;
    case O::SetContextReg:base=AmdGpu::Regs::ContextRegWordOffset;break;
    case O::SetShReg:base=AmdGpu::Regs::ShRegWordOffset;break;
    case O::SetUconfigReg:base=AmdGpu::Regs::UconfigRegWordOffset;break;
    case O::ClearState:s.Mark(0,AmdGpu::Regs::NumRegs);return false;
    case O::DrawIndex2:case O::DrawIndexOffset2:case O::DrawIndexAuto:
    case O::DispatchDirect:case O::NumInstances:case O::IndexType:
    case O::IndexBase:case O::IndexBufferSize:case O::ContextControl:return false;
    // No stage-B state: AcquireMem is an empty case in both processors, SetBase only sets
    // indirect_args_addr (read by indirect draws/dispatches, which drain), and an indirect
    // buffer only makes A parse a nested stream whose packets pass through this filter.
    case O::AcquireMem:case O::SetBase:case O::IndirectBuffer:return false;
    case O::Nop:
        if(count>1&&words[1]==u32(AmdGpu::PM4CmdNop::PayloadType::PatchedFlip))drain(iss::DrawPipe::Fence);
        return false;
    case O::DmaData:
        if(count>=7&&words[4]==0x3022c)return false; // Same explicit no-op as original Liverpool.
        // Immediate -> GDS cannot mutate guest resource tables inspected by stage A.
        // The scalar fill and every dependent dispatch still run in order on B.
        if(count>=7&&s.pipe_gds&&((words[1]>>29)&3)==2&&((words[1]>>20)&3)==1)return false;
        [[fallthrough]];
    default:drain(iss::DrawPipe::Packet);return false;
    }
    if(count>=2)s.Mark(base+(words[1]&0xffff),count-2);
    return false;
}
bool Rasterizer::PostDrawPipeGdsFill(VAddr address,u32 size,u32 value){
    if(!UseDrawPipe()||!draw_pipe_state->pipe_gds)return false;
    auto& s=*draw_pipe_state;auto* out=s.queue->Begin(sizeof(PipePacket)+sizeof(PipeGdsFill));
    auto& p=*reinterpret_cast<PipePacket*>(out);p={};p.async=async_compute_translation;p.queue_id=liverpool->CurrentQueue();
    *reinterpret_cast<PipeGdsFill*>(out+sizeof(p))={address,size,value};s.queue->Commit();return true;
}
bool Rasterizer::PostDrawPipe(const Pipeline* pipeline,bool compute,bool indexed,u32 index_offset,
                              const std::array<u64,5>* indirect){
    if(!UseDrawPipe())return false;
    auto& s=*draw_pipe_state;
    // Future geometry/tessellation stages may perform other indirect UD reads: serial fallback.
    for(const auto* st:pipeline->GetStages())if(st && st->sw_stage!=Shader::SwStage::Vertex&&st->sw_stage!=Shader::SwStage::Fragment&&st->sw_stage!=Shader::SwStage::Compute)return false;
    struct Capture {const Shader::Info* info;std::array<Shader::Info::UdRead,32> reads;u32 count;};
    std::array<Capture,6> captures;u32 nstages=0;
    for(const auto* info:pipeline->GetStages()){
        if(!info)continue;
        if(info->user_data.size()>Shader::NUM_USER_DATA_REGS || info->flattened_ud_buf.size()>iss::DrawPipe::MaxPayload/4)return false;
        auto& c=captures[nstages++];c.info=info;c.count=0;
        if(!compute&&info->sw_stage==Shader::SwStage::Vertex){
            const auto& attrs=static_cast<const GraphicsPipeline*>(pipeline)->GetFetchShader().attributes;
            if(attrs.size()>c.reads.size())return false;
            for(const auto& attr:attrs)c.reads[c.count++]={attr.sgpr_base,attr.dword_offset,info->ReadUdReg<AmdGpu::Buffer>(attr.sgpr_base,attr.dword_offset)};
        }
    }
    // Direct draw packets also mutate these fields outside SET_* register packets.
#define ISS_PIPE_FIELD(field) s.Mark(offsetof(AmdGpu::Regs,field)/4,(sizeof(liverpool->regs.field)+3)/4)
    ISS_PIPE_FIELD(index_buffer_type);ISS_PIPE_FIELD(index_base_address);ISS_PIPE_FIELD(max_index_size);
    ISS_PIPE_FIELD(num_indices);ISS_PIPE_FIELD(draw_initiator);ISS_PIPE_FIELD(num_instances);ISS_PIPE_FIELD(cp_strmout_cntl);
#undef ISS_PIPE_FIELD
    u64 size=sizeof(PipePacket)+PipeAlign(s.changed_count*sizeof(u16))+s.changed_count*DrawPipeState::BlockWords*4;
    for(u32 i=0;i<nstages;++i){const auto& c=captures[i];size+=sizeof(PipeStage)+PipeAlign((c.info->user_data.size()+c.info->flattened_ud_buf.size())*4)+c.count*sizeof(Shader::Info::UdRead);}
    if(compute)size+=PipeAlign(sizeof(AmdGpu::ComputeProgram));
    const bool verify=s.verify_interval&&(s.queue->packets%s.verify_interval)==0;
    if(verify)size+=sizeof(AmdGpu::Regs);
    if(size>iss::DrawPipe::MaxPayload)return false;
    u8* out=s.queue->Begin(static_cast<u32>(size));u8* begin=out;
    auto& p=*reinterpret_cast<PipePacket*>(out);p={};p.pipeline=pipeline;p.blocks=s.changed_count;p.stages=nstages;p.index_offset=index_offset;p.compute=compute;p.indexed=indexed;p.verify=verify;
    p.async=async_compute_translation;p.queue_id=liverpool->CurrentQueue();p.context_changes=liverpool->ContextRegChanges();p.cb=liverpool->last_cb_extent;p.db=liverpool->last_db_extent;
    if(s.fast&&!compute){
        // The changes since the last posted draw (or more: draws run here keep them).
        p.num_changed=liverpool->num_changed_regs;
        std::memcpy(p.changed.data(),liverpool->changed_regs.data(),std::min<u32>(p.num_changed,32)*sizeof(u32));
        liverpool->num_changed_regs=0;
    }
    if(s.fast)liverpool->note_changed_regs=true;
    if(indirect){p.indirect=true;p.ind=*indirect;s.deferred_indirect.fetch_add(1,std::memory_order_relaxed);}
    out+=sizeof(p);
    std::memcpy(out,s.changed.data(),s.changed_count*sizeof(u16));out+=PipeAlign(s.changed_count*sizeof(u16));
    for(u32 i=0;i<s.changed_count;++i){std::memcpy(out,liverpool->regs.reg_array.data()+s.changed[i]*DrawPipeState::BlockWords,DrawPipeState::BlockWords*4);out+=DrawPipeState::BlockWords*4;}
    for(u32 i=0;i<nstages;++i){const auto& c=captures[i];const auto& info=*c.info;auto& h=*reinterpret_cast<PipeStage*>(out);h={c.info,info.pgm_base,u32(info.user_data.size()),u32(info.flattened_ud_buf.size()),c.count,0};out+=sizeof(h);
        std::memcpy(out,info.user_data.data(),h.ud_size*4);std::memcpy(out+h.ud_size*4,info.flattened_ud_buf.data(),h.flat_size*4);out+=PipeAlign((h.ud_size+h.flat_size)*4);
        std::memcpy(out,c.reads.data(),c.count*sizeof(Shader::Info::UdRead));out+=c.count*sizeof(Shader::Info::UdRead);
    }
    if(compute){std::memcpy(out,&liverpool->GetCsRegs(),sizeof(AmdGpu::ComputeProgram));out+=PipeAlign(sizeof(AmdGpu::ComputeProgram));}
    if(verify){std::memcpy(out,&liverpool->regs,sizeof(AmdGpu::Regs));out+=sizeof(AmdGpu::Regs);}
    ASSERT(u64(out-begin)==size);s.Clear();s.queue->Commit();
    if((s.queue->packets&1023)!=0)return true;
    const auto now=Common::Perf::NowNs();
    if(now-s.last_report>=5'000'000'000ull){s.last_report=now;
        if(s.log){std::fprintf(s.log,"%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu",now,s.queue->packets,s.queue->completed_packets.load(),s.draws.load(),s.dispatches.load(),s.queue->busy_ns.load(),s.queue->wait_ns,s.queue->drains,s.verify_checks.load(),s.gds_fills.load());
            for(u32 i=0;i<8;++i)std::fprintf(s.log,",%llu",s.queue->reason_ns[i]);std::fputc('\n',s.log);}
        std::fprintf(stderr,"ISS_DRAW_PIPE packets=%llu done=%llu B_ms=%.3f A_wait_ms=%.3f drains=%llu verify=%llu\n",s.queue->packets,s.queue->completed_packets.load(),double(s.queue->busy_ns.load())/1e6,double(s.queue->wait_ns)/1e6,s.queue->drains,s.verify_checks.load());
        // Cumulative drain attribution: reason count/ms, write-fault guards, then PM4 opcodes.
        std::fprintf(stderr,"ISS_DRAW_PIPE_DEFER fences=%llu indirect=%llu wait_skips=%llu wait_drains=%llu\n",
                     s.deferred_fences.load(),s.deferred_indirect.load(),s.wait_skips.load(),s.wait_drains.load());
        std::fprintf(stderr,"ISS_DRAW_PIPE_DRAINS guard=%llu reasons",s.guard_commands.load());
        for(u32 i=0;i<iss::DrawPipe::NumReasons;++i)std::fprintf(stderr," %u:%llu:%.1f",i,s.queue->reason_count[i],double(s.queue->reason_ns[i])/1e6);
        std::fputs(" ops",stderr);
        for(u32 i=0;i<256;++i)if(s.op_drains[i])std::fprintf(stderr," %02x:%llu:%.1f",i,s.op_drains[i],double(s.op_ns[i])/1e6);
        std::fputc('\n',stderr);
    }
    return true;
}
static bool FastDrawEnabled(); // fast_draw.inl
void Rasterizer::RunDrawPipePacket(void* context,const u8* data,u32 size){
    auto& self=*static_cast<Rasterizer*>(context);auto& s=*self.draw_pipe_state;
    const auto& p=*reinterpret_cast<const PipePacket*>(data);const u8* in=data+sizeof(p);
    if(p.fence_op){
        ASSERT(size==sizeof(PipePacket)+PipeAlign(p.fence_words*4));
        s.async=p.async;s.queue_id=p.queue_id;active_draw_pipe=&s;
        self.scheduler.PopPendingOperations();
        self.liverpool->RunDeferredFence(p.fence_op,reinterpret_cast<const u32*>(in),p.fence_pipe);
        if(p.fence_seq)s.fences_done.store(p.fence_seq,std::memory_order_release);
        active_draw_pipe=nullptr;return;
    }
    if(!p.pipeline){
        ASSERT(size==sizeof(PipePacket)+sizeof(PipeGdsFill));const auto& fill=*reinterpret_cast<const PipeGdsFill*>(in);
        s.async=p.async;s.queue_id=p.queue_id;active_draw_pipe=&s;
        self.FillBuffer(fill.address,fill.size,fill.value,true);
        active_draw_pipe=nullptr;s.gds_fills.fetch_add(1,std::memory_order_relaxed);return;
    }
    const auto* blocks=reinterpret_cast<const u16*>(in);in+=PipeAlign(p.blocks*sizeof(u16));
    for(u32 i=0;i<p.blocks;++i){ASSERT(blocks[i]<DrawPipeState::NumBlocks);std::memcpy(s.regs.reg_array.data()+blocks[i]*DrawPipeState::BlockWords,in,DrawPipeState::BlockWords*4);in+=DrawPipeState::BlockWords*4;}
    ASSERT(p.stages<=Shader::Info::ud_snapshots.size());
    for(u32 i=0;i<p.stages;++i){const auto& st=*reinterpret_cast<const PipeStage*>(in);in+=sizeof(st);const auto* ud=reinterpret_cast<const u32*>(in);in+=PipeAlign((st.ud_size+st.flat_size)*4);const auto* reads=reinterpret_cast<const Shader::Info::UdRead*>(in);in+=st.reads*sizeof(Shader::Info::UdRead);
        Shader::Info::ud_snapshots[i]={st.info,ud,st.ud_size,ud+st.ud_size,st.flat_size,st.base,reads,st.reads};}
    s.cs=nullptr;if(p.compute){s.cs=reinterpret_cast<const AmdGpu::ComputeProgram*>(in);in+=PipeAlign(sizeof(AmdGpu::ComputeProgram));}
    if(p.verify){
        const auto* expected=reinterpret_cast<const AmdGpu::Regs*>(in);
        for(u32 i=0;i<AmdGpu::Regs::NumRegs;++i)ASSERT_MSG(s.regs.reg_array[i]==expected->reg_array[i],"draw pipe register mismatch at {:#x}: {:#x} != {:#x}",i,s.regs.reg_array[i],expected->reg_array[i]);
        in+=sizeof(AmdGpu::Regs);s.verify_checks.fetch_add(1,std::memory_order_relaxed);
    }
    ASSERT(u64(in-data)==size);s.cb=p.cb;s.db=p.db;s.async=p.async;s.queue_id=p.queue_id;s.context_changes=p.context_changes;
    Shader::Info::ud_snapshots_used.store(true,std::memory_order_relaxed);
    active_draw_pipe=&s;Shader::Info::num_ud_snapshots=p.stages;
    self.scheduler.PopPendingOperations();
    if(p.compute){self.DispatchRecord(static_cast<const ComputePipeline*>(p.pipeline));s.dispatches.fetch_add(1,std::memory_order_relaxed);}
    else{
        bool done=false;
        if(p.indirect){
            self.DrawIndirectRecord(static_cast<const GraphicsPipeline*>(p.pipeline),p.indexed,p.ind[0],
                                    static_cast<u32>(p.ind[1]),static_cast<u32>(p.ind[2]),static_cast<u32>(p.ind[3]),p.ind[4]);
            done=true;
        }else if(s.fast&&FastDrawEnabled()){
            self.pipe_pipeline=static_cast<const GraphicsPipeline*>(p.pipeline);
            self.pipe_num_changed_regs=p.num_changed;
            std::memcpy(self.pipe_changed_regs.data(),p.changed.data(),std::min<u32>(p.num_changed,32)*sizeof(u32));
            done=self.TryFastDraw(p.indexed,p.index_offset);
        }
        if(!done)self.DrawRecord(static_cast<const GraphicsPipeline*>(p.pipeline),p.indexed,p.index_offset);
        s.draws.fetch_add(1,std::memory_order_relaxed);
    }
    Shader::Info::num_ud_snapshots=0;active_draw_pipe=nullptr;
}
