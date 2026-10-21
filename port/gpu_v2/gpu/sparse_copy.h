// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <stdint.h>
#include <string.h>
// backing() returns a pointer only when the entire requested range is mapped.
// Preserve sparse semantics: gaps are zero-filled; no resource range is truncated.
static inline void iss_copy_sparse(uint64_t address,uint8_t *dest,uint64_t size,
                                   uint8_t *(*backing)(uint64_t,uint64_t)){
    if(!size)return;
    uint8_t *whole=backing(address,size);
    if(whole){memcpy(dest,whole,(size_t)size);return;}
    while(size){
        uint64_t part=16384-(address&16383);
        if(part>size)part=size;
        uint8_t *source=backing(address,part);
        if(source && part==16384){
            // A large sparse allocation often consists of long mapped runs
            // separated by a few holes. Probe only after the first mapped
            // page; backing() confirms the entire candidate is contiguous.
            static const uint64_t runs[]={4ULL<<20,1ULL<<20,256ULL<<10,64ULL<<10};
            for(unsigned i=0;i<sizeof(runs)/sizeof(runs[0]);++i){
                if(runs[i]>size)continue;
                uint8_t *run=backing(address,runs[i]);
                if(run){source=run;part=runs[i];break;}
            }
        }
        if(source)memcpy(dest,source,(size_t)part);else memset(dest,0,(size_t)part);
        address+=part;dest+=part;size-=part;
    }
}
