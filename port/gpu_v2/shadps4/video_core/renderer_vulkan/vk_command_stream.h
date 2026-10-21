// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

/**
 * Vulkan commands recorded by the command processor thread and replayed later on a command
 * buffer by the recording thread of its scheduler (threaded recording): the command processor no
 * longer spends its time in the driver. Commands are stored one after the other in reused memory
 * blocks, with the data their arguments point to (copied when recorded).
 */
class CommandStream {
public:
    CommandStream() = default;
    ~CommandStream() {
        Reset();
    }

    CommandStream(const CommandStream&) = delete;
    CommandStream& operator=(const CommandStream&) = delete;

    /// Appends a command: func(vk::CommandBuffer) runs when the stream is executed.
    template <typename Func>
    void Record(Func&& func) {
        using Command = TypedCommand<std::decay_t<Func>>;
        void* const memory = Allocate(sizeof(Command), alignof(Command));
        CommandBase* const command = new (memory) Command(std::forward<Func>(func));
        if (last) {
            last->next = command;
        } else {
            first = command;
        }
        last = command;
        ++num_commands;
    }

    /// Copies count objects in memory that stays valid until the commands are executed.
    template <typename T>
    T* Copy(const T* data, size_t count) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (count == 0 || data == nullptr) {
            return nullptr;
        }
        void* const memory = Allocate(sizeof(T) * count, alignof(T));
        std::memcpy(memory, data, sizeof(T) * count);
        return static_cast<T*>(memory);
    }

    const char* CopyString(const char* str) {
        return str ? Copy(str, std::strlen(str) + 1) : nullptr;
    }

    [[nodiscard]] bool Empty() const noexcept {
        return first == nullptr;
    }

    [[nodiscard]] size_t NumCommands() const noexcept {
        return num_commands;
    }

    /// Replays the commands on the command buffer, then forgets them (the memory is kept).
    void Execute(vk::CommandBuffer cmdbuf) {
        for (CommandBase* command = first; command != nullptr;) {
            CommandBase* const next = command->next;
            command->Execute(cmdbuf);
            command->~CommandBase();
            command = next;
        }
        first = nullptr;
        last = nullptr;
        num_commands = 0;
        Rewind();
    }

    /// Forgets the commands without executing them (the memory is kept).
    void Reset() {
        for (CommandBase* command = first; command != nullptr;) {
            CommandBase* const next = command->next;
            command->~CommandBase();
            command = next;
        }
        first = nullptr;
        last = nullptr;
        num_commands = 0;
        Rewind();
    }

private:
    struct CommandBase {
        virtual ~CommandBase() = default;
        virtual void Execute(vk::CommandBuffer cmdbuf) = 0;
        CommandBase* next{};
    };

    template <typename Func>
    struct TypedCommand final : CommandBase {
        template <typename F>
        explicit TypedCommand(F&& f) : func{std::forward<F>(f)} {}

        void Execute(vk::CommandBuffer cmdbuf) override {
            func(cmdbuf);
        }

        Func func;
    };

    static constexpr size_t BlockSize = 64 * 1024;

    void* Allocate(size_t size, size_t align) {
        if (size > BlockSize / 2) {
            // Large data (long barrier or copy lists) gets a block of its own.
            auto& block = large_blocks.emplace_back(std::make_unique<std::byte[]>(size + align));
            void* pointer = block.get();
            size_t space = size + align;
            return std::align(align, size, pointer, space);
        }
        while (true) {
            if (block_index == blocks.size()) {
                blocks.emplace_back(std::make_unique<std::byte[]>(BlockSize));
            }
            const size_t offset = (block_offset + align - 1) & ~(align - 1);
            if (offset + size <= BlockSize) {
                block_offset = offset + size;
                return blocks[block_index].get() + offset;
            }
            ++block_index;
            block_offset = 0;
        }
    }

    void Rewind() {
        block_index = 0;
        block_offset = 0;
        large_blocks.clear();
    }

    std::vector<std::unique_ptr<std::byte[]>> blocks;
    std::vector<std::unique_ptr<std::byte[]>> large_blocks;
    size_t block_index{};
    size_t block_offset{};
    CommandBase* first{};
    CommandBase* last{};
    size_t num_commands{};
};

} // namespace Vulkan
