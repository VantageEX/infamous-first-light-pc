// SPDX-License-Identifier: GPL-2.0-or-later
// Linux host: this port's first-start shader compilation (port/gpu/shadps4/video_core/
// renderer_vulkan/vk_precompile.cpp) on this renderer. The job file lists draws as the pipeline
// cache sees them (registers, shader code, flattened user data, vertex buffer descriptors), made
// on the player's machine from the player's own game files (port/tools/xpps). Before the game
// starts, each graphics record is looked up exactly as if the game drew it; the pipelines are
// built by the driver on every core, registered with the shader cache like the game's own and
// released (the game builds them again from the caches when it draws). Compute is left to
// ISS_AOT_COMPUTE, which compiles every compute shader of the game files.
// Included at the end of overrides/vk_pipeline_cache.cpp (namespace Vulkan).

// iss::assert_throws (host_hooks.cpp): declared at the top of vk_pipeline_cache.cpp.

namespace {
namespace Jobs {

constexpr char Magic[8] = {'I', 'S', 'S', 'P', 'J', 'O', 'B', '1'};

struct VertexSharp {
    u32 sgpr_base, dword_offset;
    std::array<u32, 4> sharp;
};
struct Stage {
    u32 sw_stage, hw_stage;
    u64 hash;
    std::vector<u8> code;
    std::array<u32, 16> user_data;
    std::vector<u32> flat;
    std::vector<u8> fetch_code;
    u32 fetch_sgpr = 0;
    std::vector<VertexSharp> vertex_sharps;
};
struct Record {
    u32 kind;
    std::vector<std::pair<u32, u32>> regs;
    std::vector<Stage> stages;
};

class Reader {
public:
    explicit Reader(std::vector<u8> bytes_) : bytes{std::move(bytes_)} {}
    template <typename T>
    bool Get(T& value) {
        if (at + sizeof(T) > bytes.size()) return false;
        std::memcpy(&value, bytes.data() + at, sizeof(T));
        at += sizeof(T);
        return true;
    }
    bool Bytes(std::vector<u8>& out, u32 size) {
        if (at + size > bytes.size()) return false;
        out.assign(bytes.begin() + at, bytes.begin() + at + size);
        at += size;
        return true;
    }
    bool Done() const { return at == bytes.size(); }

private:
    std::vector<u8> bytes;
    size_t at = 0;
};

bool Read(const std::filesystem::path& path, std::vector<Record>& records) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::vector<u8> bytes{std::istreambuf_iterator<char>(in), {}};
    Reader r{std::move(bytes)};
    char magic[8];
    u32 version;
    if (!r.Get(magic) || std::memcmp(magic, Magic, 8) != 0 || !r.Get(version) || version != 1) {
        std::fprintf(stderr, "PRECOMPILE_JOBS %s: not a job file\n", path.string().c_str());
        return false;
    }
    while (!r.Done()) {
        Record rec{};
        u32 n;
        if (!r.Get(rec.kind) || !r.Get(n)) return false;
        rec.regs.resize(n);
        for (auto& [reg, value] : rec.regs) {
            if (!r.Get(reg) || !r.Get(value)) return false;
        }
        u32 stages;
        if (!r.Get(stages)) return false;
        rec.stages.resize(stages);
        for (auto& st : rec.stages) {
            u32 size;
            if (!r.Get(st.sw_stage) || !r.Get(st.hw_stage) || !r.Get(st.hash) || !r.Get(size) ||
                !r.Bytes(st.code, size) || !r.Get(st.user_data) || !r.Get(n))
                return false;
            st.flat.resize(n);
            for (auto& w : st.flat) {
                if (!r.Get(w)) return false;
            }
            if (!r.Get(size) || !r.Bytes(st.fetch_code, size) || !r.Get(st.fetch_sgpr) ||
                !r.Get(n))
                return false;
            st.vertex_sharps.resize(n);
            for (auto& vs : st.vertex_sharps) {
                if (!r.Get(vs.sgpr_base) || !r.Get(vs.dword_offset) || !r.Get(vs.sharp)) return false;
            }
        }
        records.push_back(std::move(rec));
    }
    return true;
}

/// Host memory standing in for the guest memory the registers point at (shader code, fetch
/// shaders, vertex buffer descriptor tables): 256-aligned, below 2^48 (40-bit, 256-byte units).
class Arena {
public:
    void* Place(const void* data, size_t size) {
        // Whole pages, page-aligned, one spare page after: the recompiler reads code up to the
        // end of its page (as guest memory allows).
        constexpr size_t Page = 4096;
        const size_t bytes = (size + Page - 1) / Page * Page + Page;
        auto& block = blocks.emplace_back(std::make_unique<u8[]>(bytes + Page));
        auto* aligned = reinterpret_cast<u8*>((reinterpret_cast<uintptr_t>(block.get()) + Page - 1) &
                                              ~uintptr_t{Page - 1});
        // The slack holds s_endpgm, not zeros: a decoder that runs past a block (the fetch
        // shader a vertex shader calls) stops there instead of decoding 64 KiB of zeros.
        auto* words = reinterpret_cast<u32*>(aligned);
        std::fill(words, words + bytes / sizeof(u32), data ? 0xBF810000u : 0u);
        if (data) std::memcpy(aligned, data, size);
        ranges.emplace_back(reinterpret_cast<uintptr_t>(aligned),
                            reinterpret_cast<uintptr_t>(aligned) + bytes);
        return aligned;
    }
    bool Contains(const void* address) const {
        const auto a = reinterpret_cast<uintptr_t>(address);
        for (const auto& [begin, end] : ranges) {
            if (a >= begin && a < end) return true;
        }
        return false;
    }

private:
    std::vector<std::unique_ptr<u8[]>> blocks;
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
};
/// The arena of the record being looked up or built on this thread (Shader::Info::precompile_code_ok).
thread_local const Arena* current_arena{};


void SetPointer(AmdGpu::UserData& user_data, u32 sgpr, const void* pointer) {
    const auto value = reinterpret_cast<uintptr_t>(pointer);
    user_data[sgpr] = u32(value);
    user_data[sgpr + 1] = u32(value >> 32);
}

/// Shader code stays for the whole precompile, one copy per shader: the program cache keeps
/// a pointer to the code of the record that first used a shader, and translates the shader's
/// later permutations from it (in game the code stays in guest memory).
struct CodeStore {
    Arena arena;
    std::unordered_map<u64, void*> by_hash;
    void* Get(const Stage& stage) {
        const u64 key = stage.hash ^ (u64(stage.code.size()) << 40);
        auto [it, is_new] = by_hash.try_emplace(key, nullptr);
        if (is_new) it->second = arena.Place(stage.code.data(), stage.code.size());
        return it->second;
    }
};
thread_local CodeStore* code_store{};
bool CodeInArena(const void* address) {
    return (current_arena && current_arena->Contains(address)) ||
           (code_store && code_store->arena.Contains(address));
}

void PlaceStage(Arena& arena, const Stage& stage, u64& address, AmdGpu::UserData& user_data) {
    void* code = code_store ? code_store->Get(stage) : arena.Place(stage.code.data(), stage.code.size());
    address = reinterpret_cast<uintptr_t>(code) >> 8;
    std::copy(stage.user_data.begin(), stage.user_data.end(), user_data.begin());
    if (!stage.fetch_code.empty() && stage.fetch_sgpr + 1 < user_data.size()) {
        SetPointer(user_data, stage.fetch_sgpr,
                   arena.Place(stage.fetch_code.data(), stage.fetch_code.size()));
    }
    std::unordered_map<u32, u32> table_dwords;
    for (const auto& vs : stage.vertex_sharps) {
        auto& dwords = table_dwords[vs.sgpr_base];
        dwords = std::max(dwords, vs.dword_offset + 4);
    }
    for (const auto& [sgpr, dwords] : table_dwords) {
        auto* table = static_cast<u32*>(arena.Place(nullptr, dwords * sizeof(u32)));
        for (const auto& vs : stage.vertex_sharps) {
            if (vs.sgpr_base == sgpr) std::memcpy(table + vs.dword_offset, vs.sharp.data(), 16);
        }
        if (sgpr + 1 < user_data.size()) SetPointer(user_data, sgpr, table);
    }
}

/// One pipeline for the build threads: copies of what the lookup chose for a record.
struct Build {
    std::unique_ptr<Arena> arena;
    std::unordered_map<u64, std::vector<u32>> flats;
    GraphicsPipelineKey key{};
    std::array<std::unique_ptr<Shader::Info>, MaxShaderStages> info_copies;
    std::array<std::vector<u32>, MaxShaderStages> user_data;
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    GraphicsPipeline::SerializationSupport sdata{};
    bool ok = false;
};

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream in(path);
    return std::string{std::istreambuf_iterator<char>(in), {}};
}
void WriteText(const std::filesystem::path& path, const std::string& text) {
    auto temp = path;
    temp += ".tmp";
    {
        std::ofstream out(temp, std::ios::trunc);
        out << text;
    }
    std::error_code ec;
    std::filesystem::rename(temp, path, ec);
}

} // namespace Jobs
} // namespace

void PipelineCache::PrecompileJobs(const std::function<void(u32 done, u32 total)>& progress) {
    std::filesystem::path jobs;
    if (const char* p = std::getenv("TLG_PRECOMPILE_JOBS"); p && *p) {
        jobs = p;
    } else {
        jobs = Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "cache" /
               (Common::ElfInfo::Instance().GameSerial().empty()
                    ? std::string{"CUSA00223"}
                    : std::string{Common::ElfInfo::Instance().GameSerial()}) ;
        jobs += "-jobs.bin";
    }
    std::error_code ec;
    if (const char* p = std::getenv("ISS_PRECOMPILE_JOBS"); (p && p[0] == '0') ||
        !std::filesystem::exists(jobs, ec)) {
        std::fprintf(stderr, "PRECOMPILE_JOBS off (%s)\n",
                     std::filesystem::exists(jobs, ec) ? "ISS_PRECOMPILE_JOBS=0" : "no job file");
        return;
    }
    // Once per job file, driver and renderer build: the driver's pipeline cache is lost when the
    // driver or the GPU changes.
    const auto uuid = instance.GetPipelineCacheUUID();
    u64 uuid_hash = 0;
    for (const u8 byte : uuid) uuid_hash = uuid_hash * 1099511628211ull ^ byte;
    const std::string fingerprint =
        fmt::format("{} {} {:x}:{:x}:{:x}:{:x} {}\n", std::filesystem::file_size(jobs, ec),
                    std::filesystem::last_write_time(jobs, ec).time_since_epoch().count(),
                    instance.GetVendorID(), instance.GetDeviceID(), instance.GetDriverVersion(),
                    uuid_hash, __DATE__ " " __TIME__);
    const auto done_file =
        Common::FS::GetUserPath(Common::FS::PathType::CacheDir) / "precompiled-jobs.txt";
    if (Jobs::ReadText(done_file) == fingerprint) {
        std::fprintf(stderr, "PRECOMPILE_JOBS done before (%s)\n", done_file.string().c_str());
        return;
    }
    std::vector<Jobs::Record> records;
    if (!Jobs::Read(jobs, records)) {
        std::fprintf(stderr, "PRECOMPILE_JOBS %s unreadable\n", jobs.string().c_str());
        return;
    }
    u32 total_graphics = 0;
    for (const auto& r : records) total_graphics += r.kind == 0;
    const auto started = std::chrono::steady_clock::now();
    if (progress) progress(0, total_graphics); // the bar shows from the start
    // A record that takes over 2 s is named in the log (a lookup stuck in one shader).
    std::atomic<u32> current_record{0};
    std::atomic<bool> watching{true};
    std::thread watchdog([&] {
        u32 last = ~0u;
        auto since = std::chrono::steady_clock::now();
        while (watching.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            const u32 now_record = current_record.load();
            if (now_record != last) {
                last = now_record;
                since = std::chrono::steady_clock::now();
            } else if (std::chrono::steady_clock::now() - since > std::chrono::seconds(2)) {
                std::fprintf(stderr, "PRECOMPILE_JOBS record %u is taking over 2 s\n", now_record);
                since = std::chrono::steady_clock::now() + std::chrono::hours(1);
            }
        }
    });

    // Build threads, each with its own driver pipeline cache (merged into the game's at the end).
    unsigned threads = std::max(1u, std::thread::hardware_concurrency() - 1);
    if (const char* p = std::getenv("ISS_PRECOMPILE_THREADS"); p && *p) {
        threads = unsigned(std::strtoul(p, nullptr, 10)); // 0: built on this thread
    }
    std::vector<vk::UniquePipelineCache> caches;
    for (unsigned i = 0; i < threads; ++i) {
        auto [result, cache] = instance.GetDevice().createPipelineCacheUnique({});
        if (result != vk::Result::eSuccess) break;
        caches.push_back(std::move(cache));
    }
    std::mutex mutex;
    std::condition_variable work, space;
    std::deque<std::unique_ptr<Jobs::Build>> queued;
    std::vector<std::unique_ptr<Jobs::Build>> finished;
    bool stopping = false;
    size_t running = 0;
    const size_t limit = caches.size() * 4;
    std::vector<std::thread> pool;
    for (size_t t = 0; t < caches.size(); ++t) {
        pool.emplace_back([&, t] {
            Common::SetCurrentThreadName(fmt::format("iss:Precompile{}", t).c_str());
            for (;;) {
                std::unique_ptr<Jobs::Build> b;
                {
                    std::unique_lock lock{mutex};
                    work.wait(lock, [&] { return stopping || !queued.empty(); });
                    if (queued.empty()) return;
                    b = std::move(queued.front());
                    queued.pop_front();
                    ++running;
                }
                Shader::Info::precompile_flats = &b->flats;
                Jobs::current_arena = b->arena.get();
                Shader::Info::precompile_code_ok = Jobs::CodeInArena;
                iss::assert_throws = true;
                try {
                    GraphicsPipeline pipeline{instance, scheduler, desc_heap, profile, b->key,
                                              *caches[t], b->infos, b->runtime_infos,
                                              b->fetch ? &*b->fetch : nullptr, b->modules,
                                              b->sdata, false, false};
                    b->ok = true;
                } catch (const std::exception&) {
                    b->ok = false;
                }
                iss::assert_throws = false;
                Shader::Info::precompile_flats = nullptr;
                Shader::Info::precompile_code_ok = nullptr;
                Jobs::current_arena = nullptr;
                {
                    std::scoped_lock lock{mutex};
                    --running;
                    finished.push_back(std::move(b));
                }
                space.notify_one();
            }
        });
    }
    u32 built = 0, failed = 0, skipped = 0, done = 0;
    std::unordered_set<u64> seen;
    const auto collect = [&] {
        std::vector<std::unique_ptr<Jobs::Build>> list;
        {
            std::scoped_lock lock{mutex};
            list = std::exchange(finished, {});
        }
        for (auto& b : list) {
            if (b->ok) {
                RegisterPipelineData(b->key, std::hash<GraphicsPipelineKey>{}(b->key), b->sdata);
                ++num_new_pipelines;
                ++built;
            } else {
                ++failed;
            }
        }
    };

    // Kept until the process ends: the game translates later permutations of these programs
    // from this code (~8 KiB per shader).
    static auto* const store = new Jobs::CodeStore;
    Jobs::code_store = store;
    auto& regs = liverpool->regs;
    const auto saved = std::make_unique<AmdGpu::Regs>(regs);
    // The permutation memo keys on the fetch shader's code at the guest address the record's
    // user data holds (not mapped before the game starts): off while the records are looked up.
    auto& tuning = Common::Perf::GetTuning();
    const auto spec_memo = tuning.spec_memo.load();
    tuning.spec_memo = 0;
    for (const auto& record : records) {
        if (record.kind != 0) continue;
        ++done;
        current_record.store(done);
        if (progress && (done & 63) == 0) progress(done, total_graphics);
        if ((done % 1000) == 0) {
            std::fprintf(stderr, "PRECOMPILE_JOBS progress %u / %u, %u built, %u skipped, %u failed, %.0f s\n",
                         done, total_graphics, built, skipped, failed,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        }
        collect();
        auto arena = std::make_unique<Jobs::Arena>();
        std::unordered_map<u64, std::vector<u32>> flats;
        for (const auto& stage : record.stages) flats[stage.hash] = stage.flat;
        Shader::Info::precompile_flats = &flats;
        Jobs::current_arena = arena.get();
        Shader::Info::precompile_code_ok = Jobs::CodeInArena;
        iss::assert_throws = true;
        std::unique_ptr<Jobs::Build> b;
        try {
            regs.reg_array.fill(0);
            for (const auto& [reg, value] : record.regs) {
                if (reg < regs.reg_array.size()) regs.reg_array[reg] = value;
            }
            static int logged = 0;
            if (logged < 6) {
                ++logged;
                for (const auto& stage : record.stages) {
                    std::fprintf(stderr, "PRECOMPILE_JOBS record stage sw %u hw %u hash %llx code %zu B fetch %zu B at s%u\n",
                                 stage.sw_stage, stage.hw_stage, (unsigned long long)stage.hash,
                                 stage.code.size(), stage.fetch_code.size(), stage.fetch_sgpr);
                }
            }
            for (const auto& stage : record.stages) {
                auto* program = const_cast<AmdGpu::ShaderProgram*>(regs.ProgramForStage(stage.hw_stage));
                if (!program) continue;
                u64 address{};
                Jobs::PlaceStage(*arena, stage, address, program->user_data);
                program->address = address;
                static int shown = 0;
                if (!stage.fetch_code.empty() && shown < 1) {
                    std::string words;
                    for (size_t w = 0; w + 4 <= stage.fetch_code.size(); w += 4) {
                        u32 v;
                        std::memcpy(&v, stage.fetch_code.data() + w, 4);
                        words += fmt::format(" {:08x}", v);
                    }
                    std::fprintf(stderr, "PRECOMPILE_JOBS fetch code of record %u:%s\n", done, words.c_str());
                }
                if (!stage.fetch_code.empty() && shown < 4) {
                    ++shown;
                    std::fprintf(stderr, "PRECOMPILE_JOBS placed fetch at s%u: %08x %08x (program %p hw %u)\n",
                                 stage.fetch_sgpr, program->user_data[stage.fetch_sgpr],
                                 program->user_data[stage.fetch_sgpr + 1], (void*)program, stage.hw_stage);
                }
            }
            if (done == 25 || done == 26) {
                for (const auto& stage : record.stages) {
                    const auto* program = regs.ProgramForStage(stage.hw_stage);
                    const u32* code = program ? program->template Address<const u32*>() : nullptr;
                    const auto* info = code ? AmdGpu::FindBinaryInfo(code) : nullptr;
                    std::fprintf(stderr, "PRECOMPILE_JOBS record %u stage sw %u hw %u: code %zu B, binary info %s length %u at +%lld B, stage enabled mask %08x\n",
                                 done, stage.sw_stage, stage.hw_stage, stage.code.size(), info ? "found" : "missing",
                                 info ? info->length : 0u,
                                 info ? (long long)(reinterpret_cast<const u8*>(info) - reinterpret_cast<const u8*>(code)) : -1ll,
                                 regs.reg_array[0xa2d5]);
                }
            }
            if (!RefreshGraphicsKey()) {
                ++failed;
                static int refused = 0;
                if (refused < 6) {
                    ++refused;
                    std::fprintf(stderr, "PRECOMPILE_JOBS record %u: the lookup refused it\n", done);
                }
            } else if (graphics_pipelines.contains(graphics_key) ||
                       !seen.insert(std::hash<GraphicsPipelineKey>{}(graphics_key)).second) {
                ++skipped;
            } else {
                b = std::make_unique<Jobs::Build>();
                b->key = graphics_key;
                b->runtime_infos = runtime_infos;
                b->modules = modules;
                if (fetch_shader) b->fetch = *fetch_shader;
                for (u32 i = 0; i < MaxShaderStages; ++i) {
                    if (!infos[i]) continue;
                    b->info_copies[i] = std::make_unique<Shader::Info>(*infos[i]);
                    b->user_data[i].assign(infos[i]->user_data.begin(), infos[i]->user_data.end());
                    b->info_copies[i]->user_data = b->user_data[i];
                    b->infos[i] = b->info_copies[i].get();
                }
            }
        } catch (const std::exception& e) {
            ++failed;
            b.reset();
            static int reasons = 0;
            if (reasons < 12) {
                ++reasons;
                std::fprintf(stderr, "PRECOMPILE_JOBS record %u failed: %s\n", done, e.what());
            }
        }
        iss::assert_throws = false;
        Shader::Info::precompile_flats = nullptr;
        Shader::Info::precompile_code_ok = nullptr;
        Jobs::current_arena = nullptr;
        if (!b) continue;
        b->arena = std::move(arena);
        b->flats = std::move(flats);
        if (caches.empty()) {
            // ISS_PRECOMPILE_THREADS=0: built here, into the game's cache.
            Shader::Info::precompile_flats = &b->flats;
            iss::assert_throws = true;
            try {
                GraphicsPipeline pipeline{instance, scheduler, desc_heap, profile, b->key,
                                          *pipeline_cache, b->infos, b->runtime_infos,
                                          b->fetch ? &*b->fetch : nullptr, b->modules, b->sdata,
                                          false, false};
                RegisterPipelineData(b->key, std::hash<GraphicsPipelineKey>{}(b->key), b->sdata);
                ++num_new_pipelines;
                ++built;
            } catch (const std::exception&) {
                ++failed;
            }
            iss::assert_throws = false;
            Shader::Info::precompile_flats = nullptr;
            continue;
        }
        std::unique_lock lock{mutex};
        space.wait(lock, [&] { return queued.size() + running < limit; });
        queued.push_back(std::move(b));
        work.notify_one();
    }
    watching.store(false);
    watchdog.join();
    {
        std::scoped_lock lock{mutex};
        stopping = true;
    }
    work.notify_all();
    for (auto& thread : pool) thread.join();
    collect();
    regs = *saved;
    tuning.spec_memo = spec_memo;
    Jobs::code_store = nullptr;
    std::vector<vk::PipelineCache> sources;
    for (const auto& cache : caches) sources.push_back(*cache);
    if (!sources.empty() &&
        instance.GetDevice().mergePipelineCaches(*pipeline_cache, sources) != vk::Result::eSuccess) {
        std::fprintf(stderr, "PRECOMPILE_JOBS merging the build threads' caches failed\n");
    }
    if (progress) progress(total_graphics, total_graphics);
    std::fprintf(stderr,
                 "PRECOMPILE_JOBS %u graphics records in %.1f s on %zu threads: %u pipelines built, "
                 "%u already there, %u failed\n",
                 total_graphics,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(),
                 caches.size(), built, skipped, failed);
    Jobs::WriteText(done_file, fingerprint);
}
