/* Tracy zones and names from the C runtime (docs/profiling.md). The profiling build (TLG_PROFILE)
 * compiles the runtime with TRACY_ENABLE and links the Tracy client; any other build makes these
 * calls do nothing. */
#ifndef RUNTIME_TRACE_H
#define RUNTIME_TRACE_H

/* A blocking wait of the calling thread (runtime_wait_slot_begin/end): a zone named by its kind
 * ("mutex", "semaphore", "GPU readback"...) with the object's name as its text. Strictly nested,
 * per thread, as the wait slots are. */
void runtime_trace_wait_begin(const char *kind, const char *name);
void runtime_trace_wait_end(void);
/* The calling thread's name in the profiler (Linux names hold 15 characters, Windows' MinGW
 * threads none the profiler sees). */
void runtime_trace_thread_name(const char *name);
/* A secondary frame set (the game's own flips, beside the presented frames). */
void runtime_trace_frame(const char *name);

#endif
