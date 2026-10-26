/* Tracy zones and names from the C runtime (runtime_trace.h). */
#include "runtime_trace.h"

#ifdef TRACY_ENABLE
#include <string.h>
#include <tracy/TracyC.h>

/* The zones of the waits in progress on this thread; the wait slots keep two levels and count the
 * rest, so deeper ones are counted here too and get no zone. */
#define TRACE_DEPTH 8
static _Thread_local TracyCZoneCtx wait_zones[TRACE_DEPTH];
static _Thread_local unsigned wait_depth;

void runtime_trace_wait_begin(const char *kind, const char *name) {
    unsigned depth = wait_depth++;
    if (depth >= TRACE_DEPTH) return;
    const char *k = kind ? kind : "wait";
    /* The source location is the kind's: one row per kind in the profiler's statistics. */
    uint64_t srcloc = ___tracy_alloc_srcloc_name(0, "runtime", 7, "wait", 4, k, strlen(k), 0x8a5a44);
    wait_zones[depth] = ___tracy_emit_zone_begin_alloc(srcloc, 1);
    if (name && *name) ___tracy_emit_zone_text(wait_zones[depth], name, strnlen(name, 256));
}
void runtime_trace_wait_end(void) {
    if (!wait_depth) return;
    unsigned depth = --wait_depth;
    if (depth < TRACE_DEPTH) ___tracy_emit_zone_end(wait_zones[depth]);
}
void runtime_trace_thread_name(const char *name) { ___tracy_set_thread_name(name); }
void runtime_trace_frame(const char *name) { ___tracy_emit_frame_mark(name); }
#else
void runtime_trace_wait_begin(const char *kind, const char *name) { (void)kind; (void)name; }
void runtime_trace_wait_end(void) {}
void runtime_trace_thread_name(const char *name) { (void)name; }
void runtime_trace_frame(const char *name) { (void)name; }
#endif
