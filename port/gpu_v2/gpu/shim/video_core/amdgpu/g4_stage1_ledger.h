// Serial G4 admission ledger. Every call is made under Liverpool::submit_mutex.
// No thread, guest pointer, Vulkan object, or coroutine is owned here.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace AmdGpu {

template <std::size_t N>
class G4Stage1Ledger {
public:
    using U32 = std::uint32_t;
    using U64 = std::uint64_t;
    struct Token {
        U32 qid{};
        U64 generation{};
        U64 sequence{};
        friend bool operator==(const Token&, const Token&) = default;
    };
    struct Target { U64 generation{}, sequence{}; };
    struct MarkSnapshot { U64 id{}; std::array<Target, N> targets{}; };

    G4Stage1Ledger() noexcept {
        queues[0].mapped = true;
        queues[0].generation = 1;
    }

    bool Map(U32 qid, U64& generation) noexcept {
        if (qid == 0 || qid >= N) return false;
        auto& q = queues[qid];
        if (q.mapped || q.closing || q.issued != q.completed || q.generation == UINT64_MAX)
            return false;
        ++q.generation;
        q.issued = q.completed = 0;
        q.mapped = true;
        generation = q.generation;
        return true;
    }

    bool BeginClose(U32 qid, U64& generation) noexcept {
        if (qid == 0 || qid >= N) return false;
        auto& q = queues[qid];
        if (!q.mapped || q.closing) return false;
        q.closing = true;
        generation = q.generation;
        return true;
    }

    bool Drained(U32 qid, U64 generation) const noexcept {
        if (qid == 0 || qid >= N) return false;
        const auto& q = queues[qid];
        return q.mapped && q.closing && q.generation == generation &&
               q.completed == q.issued;
    }

    bool FinishClose(U32 qid, U64 generation) noexcept {
        if (!Drained(qid, generation)) return false;
        auto& q = queues[qid];
        q.mapped = false;
        q.closing = false;
        return true;
    }

    bool CanIssue(U32 qid) const noexcept {
        return qid < N && queues[qid].mapped && !queues[qid].closing;
    }

    bool Issue(U32 qid, Token& token) noexcept {
        if (!CanIssue(qid) || queues[qid].issued == UINT64_MAX) return false;
        auto& q = queues[qid];
        token = Token{qid, q.generation, ++q.issued};
        return true;
    }

    bool Complete(Token token) noexcept {
        if (token.qid >= N) return false;
        auto& q = queues[token.qid];
        if (!q.mapped || q.generation != token.generation ||
            token.sequence != q.completed + 1 || token.sequence > q.issued) return false;
        q.completed = token.sequence;
        return true;
    }

    MarkSnapshot Capture(U64 id) const noexcept {
        MarkSnapshot result{};
        result.id = id;
        for (std::size_t qid = 0; qid < N; ++qid)
            result.targets[qid] = {queues[qid].generation, queues[qid].issued};
        return result;
    }

    bool Ready(const MarkSnapshot& mark) const noexcept {
        for (std::size_t qid = 0; qid < N; ++qid) {
            const auto& q = queues[qid];
            const auto& target = mark.targets[qid];
            if (q.generation < target.generation) return false;
            if (q.generation == target.generation && q.completed < target.sequence) return false;
            // A newer generation requires the old one to have drained at FinishClose.
        }
        return true;
    }

    U64 Generation(U32 qid) const noexcept { return qid < N ? queues[qid].generation : 0; }
    U64 Issued(U32 qid) const noexcept { return qid < N ? queues[qid].issued : 0; }
    U64 Completed(U32 qid) const noexcept { return qid < N ? queues[qid].completed : 0; }
    bool Mapped(U32 qid) const noexcept { return qid < N && queues[qid].mapped; }

private:
    struct Queue { U64 generation{}, issued{}, completed{}; bool mapped{}, closing{}; };
    std::array<Queue, N> queues{};
};

} // namespace AmdGpu
