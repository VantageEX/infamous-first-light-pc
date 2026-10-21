// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <map>
#include <mutex>
#include "common/types.h"
namespace Core::FileSys {
enum class FileType {Equeue};
struct File {FileType type;};
class HandleTable {
public:
    s32 CreateHandle(){std::scoped_lock guard{lock};s32 id=next++;files.emplace(id,File{});return id;}
    File *GetFile(s32 id){std::scoped_lock guard{lock};auto it=files.find(id);return it==files.end()?nullptr:&it->second;}
    void DeleteHandle(s32 id){std::scoped_lock guard{lock};files.erase(id);}
private:
    std::mutex lock;
    std::map<s32,File> files;
    s32 next=65536; // Disjoint from the native runtime's file-descriptor table.
};
}
