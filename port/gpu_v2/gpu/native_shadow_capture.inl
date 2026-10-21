// Engine-thread incremental capture. Included in native_shadow.cpp's private namespace.
// Draw payload comes from sceGnmDrawIndex arguments. State setters are decoded once
// as they are emitted; original PM4 stays available for learning/verification/fallback.
class DirectCapture;
thread_local DirectCapture* active_capture{};
bool DirectCaptureEnabled() {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_NATIVE_SHADOW_DIRECT_CAPTURE");
        return p && p[0] == '1';
    }();
    return enabled;
}
class DirectCapture {
public:
    u32 last_opcode{}, last_reg{};
    DirectCapture(Batch& batch, const u32* start, const u32* limit)
        : b(batch), cursor(start), limit(limit), previous(active_capture) {
        b.state_regs.clear(); b.state_values.clear(); b.states.clear();
        b.draws.clear(); b.final_regs.clear();
        current.user_inherit = (1u << DrawUserWords) - 1;
        current.instances_inherit = current.index_type_inherit = true;
        active_capture = this;
    }
    ~DirectCapture() { active_capture = previous; }
    void DrawIndexed(const u32* packet, u32 count, u64 address, u32 initiator) {
        if (!Consume(packet) || packet + 6 > limit) { valid = false; return; }
        if (dirty) {
            b.states.push_back({inherit, u32(b.state_values.size())});
            b.state_values.insert(b.state_values.end(), values.begin(), values.end());
            dirty = false;
        }
        Draw d = current;
        d.packet = packet; d.max_index = count; d.index_count = count;
        d.index_lo = u32(address); d.index_hi = u32(address >> 32);
        d.initiator = initiator; d.state = u32(b.states.size() - 1);
        b.draws.push_back(d);
        cursor = packet + 6;
    }
    bool Finish(const u32* end) {
        if (!Consume(end) || b.draws.empty()) return false;
        const u32 count = u32(b.state_regs.size());
        std::array<u32, MaxStateRegs> order{};
        for (u32 k = 0; k < count; ++k) order[k] = k;
        std::sort(order.begin(), order.begin() + count, [&](u32 a, u32 c) {
            return b.state_regs[a] < b.state_regs[c];
        });
        std::vector<u32> packed;
        packed.reserve(b.states.size() * count);
        for (auto& state : b.states) {
            u64 sorted_inherit = 0;
            const auto first = state.first_value;
            state.first_value = u32(packed.size());
            for (u32 k = 0; k < count; ++k) {
                packed.push_back(b.state_values[first + order[k]]);
                sorted_inherit |= ((state.inherit >> order[k]) & 1) << k;
            }
            state.inherit = sorted_inherit;
        }
        b.state_values.swap(packed);
        for (u32 k = 0; k < count; ++k)
            if (!((inherit >> k) & 1)) b.final_regs.emplace_back(b.state_regs[k], values[k]);
        for (u32 k = 0; k < DrawUserWords; ++k)
            if (!((current.user_inherit >> k) & 1))
                b.final_regs.emplace_back(VsUserData2 + k, current.user[k]);
        std::sort(b.state_regs.begin(), b.state_regs.end());
        std::sort(b.final_regs.begin(), b.final_regs.end());
        b.has_instances = !current.instances_inherit;
        b.final_instances = current.instances;
        b.has_index_type = !current.index_type_inherit;
        b.final_index_type = current.index_type;
        return true;
    }
private:
    bool Consume(const u32* end) {
        if (!valid || end < cursor || end > limit) return valid = false;
        while (cursor < end) {
            const u32* p = cursor;
            if ((*p >> 30) == 2) { ++cursor; continue; }
            if ((*p >> 30) != 3) return valid = false;
            const u32 count = ((*p >> 16) & 0x3FFF) + 1;
            if (u64(end - p) < u64(count) + 1) return valid = false;
            const u32 op = (*p >> 8) & 255;
            last_opcode = op;
            switch (op) {
            case 0x10: break;
            case 0x2A: current.index_type = p[1]; current.index_type_inherit = false; break;
            case 0x2F: current.instances = p[1]; current.instances_inherit = false; break;
            case 0x50:
                if (count < 6 || p[4] != 0x3022C) return valid = false;
                break;
            case 0x68: case 0x69: case 0x76: case 0x79: {
                if (count < 2 || (op == 0x76 && p[1] >= 0x200)) return valid = false;
                const u32 base = op == 0x68 ? ConfigBase : op == 0x69 ? ContextBase :
                                 op == 0x76 ? ShBase : UconfigBase;
                for (u32 i = 0; i < count - 1; ++i) {
                    const u32 reg = base + p[1] + i, value = p[i + 2];
                    last_reg = reg;
                    if (op == 0x69 && RejectTargetWrite(reg, value)) return valid = false;
                    if (reg >= VsUserData2 && reg < VsUserData2 + DrawUserWords) {
                        current.user[reg - VsUserData2] = value;
                        current.user_inherit &= ~(1u << (reg - VsUserData2));
                        continue;
                    }
                    auto it = std::find(b.state_regs.begin(), b.state_regs.end(), reg);
                    const u32 k = u32(it - b.state_regs.begin());
                    if (it == b.state_regs.end()) {
                        if (k == MaxStateRegs) return valid = false;
                        b.state_regs.push_back(reg);
                    }
                    dirty |= ((inherit >> k) & 1) != 0 || values[k] != value;
                    values[k] = value; inherit &= ~(1ull << k);
                }
                break;
            }
            default: return valid = false;
            }
            cursor += count + 1;
        }
        return true;
    }
    Batch& b;
    const u32* cursor;
    const u32* limit;
    DirectCapture* previous;
    std::array<u32, MaxStateRegs> values{};
    u64 inherit = ~0ull;
    Draw current{};
    bool valid = true, dirty = true;
};

bool SameCapture(const Batch& a, const Batch& b) {
    if (a.state_regs != b.state_regs || a.draws.size() != b.draws.size() ||
        a.has_instances != b.has_instances || a.final_instances != b.final_instances ||
        a.has_index_type != b.has_index_type || a.final_index_type != b.final_index_type) return false;
    auto final = b.final_regs;
    std::sort(final.begin(), final.end());
    if (a.final_regs != final) return false;
    for (u32 i = 0; i < a.draws.size(); ++i) {
        const auto& x = a.draws[i]; const auto& y = b.draws[i];
        if (x.packet != y.packet || x.user != y.user || x.user_inherit != y.user_inherit ||
            x.index_lo != y.index_lo || x.index_hi != y.index_hi || x.index_count != y.index_count ||
            x.max_index != y.max_index || x.initiator != y.initiator || x.instances != y.instances ||
            x.index_type != y.index_type || x.instances_inherit != y.instances_inherit ||
            x.index_type_inherit != y.index_type_inherit) return false;
        const auto& xs = a.states[x.state]; const auto& ys = b.states[y.state];
        if (xs.inherit != ys.inherit) return false;
        for (u32 k = 0; k < a.state_regs.size(); ++k)
            if (a.state_values[xs.first_value + k] != b.state_values[ys.first_value + k]) return false;
    }
    return true;
}
bool FinishCapture(DirectCapture* capture, Batch& batch, const u32* first, const u32* end) {
    if (!capture) return Scan(batch, first, end);
    static const bool check = [] {
        const char* p = std::getenv("ISS_NATIVE_SHADOW_CAPTURE_CHECK");
        return p && p[0] == '1';
    }();
    static std::atomic<u64> captured{}, rejected{}, compared{}, mismatches{};
    bool valid = capture->Finish(end);
    if (check) {
        Batch reference;
        const bool reference_ok = Scan(reference, first, end);
        if (valid) {
            const bool same = reference_ok && SameCapture(batch, reference);
            ++compared;
            if (!same) { ++mismatches; valid = false; }
        } else {
            static std::atomic<u32> details{};
            if (details.fetch_add(1) < 24)
                std::fprintf(stderr, "NATIVE_SHADOW_CAPTURE_REJECT opcode=%x reg=%x draws=%zu reference_ok=%d reference_draws=%zu\n",
                             capture->last_opcode, capture->last_reg, batch.draws.size(), reference_ok, reference.draws.size());
        }
    }
    if (valid) ++captured; else ++rejected;
    if (((captured.load() + rejected.load()) & 16383) == 1)
        std::fprintf(stderr, "NATIVE_SHADOW_CAPTURE captured=%llu rejected=%llu compared=%llu mismatches=%llu\n",
                     captured.load(), rejected.load(), compared.load(), mismatches.load());
    return valid;
}
