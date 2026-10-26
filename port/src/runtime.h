#ifndef TLG_RUNTIME_H
#define TLG_RUNTIME_H
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
/* Detailed log mode (ISS_VERBOSE_LOG=1/2 -> TLG_VERBOSE_LOG): 0 off, else the level. Read once per
 * file; the extra logging of the runtime files is guarded by it, so it costs one test when off. */
static inline int runtime_verbose(void) {
    static int level=-1;
    if (level<0) { const char *e=getenv("TLG_VERBOSE_LOG"); level=e ? atoi(e) : 0; if (level<0) level=0; }
    return level;
}
#ifdef _WIN32
#include "compat_win.h"
/* No signals on Windows: the exception handler resumes the thread in tlg_longjmp
 * (gpu/shim/tlg_toggles.h uses the same buffer type). */
typedef tlg_jmp_buf sigjmp_buf;
#define sigsetjmp(buffer, save) tlg_setjmp(buffer)
#else
#include <setjmp.h>
#endif
/* Recovery point for speculative guest memory reads on this thread (probe.c fault handler). */
extern __thread sigjmp_buf *runtime_fault_recover;
/* Restarts the game (in-game settings menu, render resolution change). */
void runtime_restart(void);
#define ABI __attribute__((sysv_abi))
typedef void (ABI *GuestCallback)(void);
void runtime_start(uint64_t capabilities);
uintptr_t runtime_resolve(const char *name, int is_data);
void runtime_report(void);
/* Called at the end of runtime_report (the loader adds its own counters). */
void runtime_set_report_hook(void (*hook)(void));
/* Called by every guest thread the runtime creates, before its entry (test hooks). */
void runtime_set_thread_start_hook(void (*hook)(void));
void runtime_finalize(void *dso);
uintptr_t runtime_mutex_resolve(const char *name);
void runtime_mutex_report(void);
uintptr_t runtime_memory_resolve(const char *name);
void runtime_memory_report(void);
int runtime_memory_is_mapped(uintptr_t address, uint64_t size);
const char *runtime_import_name(const char *name);
uintptr_t runtime_rwlock_resolve(const char *name);
void runtime_rwlock_report(void);
void runtime_set_libc_tls(const void *data, uint64_t filesz, uint64_t memsz);
void runtime_set_module_tls(uint64_t module, const void *data, uint64_t filesz, uint64_t memsz);
void runtime_set_procparam(void *param);
void **runtime_application_heap_api(void);
uintptr_t runtime_thread_resolve(const char *name);
void runtime_thread_report(void);
void runtime_set_main_tls(const void *data, uint64_t filesz, uint64_t memsz, uint64_t align);
void runtime_thread_attach_main(void);
int32_t *runtime_errno(void);
uintptr_t runtime_sema_resolve(const char *name);
void runtime_sema_report(void);
unsigned runtime_sema_waiters(uint32_t id);
uintptr_t runtime_time_resolve(const char *name);
void runtime_content_configure(const uint32_t values[5]);
uintptr_t runtime_content_resolve(const char *name);
void runtime_content_report(void);
typedef struct { const char *name; void *function; } RuntimeExport;
#define RUNTIME_LOOKUP(table, nid) runtime_lookup(table, sizeof(table)/sizeof(*(table)), nid)
const char *runtime_symbol(const char *nid);
uintptr_t runtime_lookup(const RuntimeExport *table, size_t count, const char *nid);
uintptr_t runtime_kernel_resolve(const char *name);
uintptr_t runtime_file_resolve(const char *name);
uintptr_t runtime_services_resolve(const char *name);
uintptr_t runtime_tlg_resolve(const char *name);
uintptr_t runtime_gr2_resolve(const char *name);
uintptr_t runtime_gr2_np_resolve(const char *name);
uintptr_t runtime_gr2_font_resolve(const char *name);
uintptr_t runtime_gr2_json_resolve(const char *name);
uintptr_t runtime_infamous_resolve(const char *name);
uintptr_t runtime_gr2_audio3d_resolve(const char *name);
void runtime_gr2_audio3d_report(void);
void runtime_file_report(void);
void runtime_file_configure(const char *app0, const char *user);
int runtime_file_mount(const char *guest, const char *host);
void runtime_file_unmount(const char *guest);
int runtime_file_translate(const char *guest, char *out, size_t size);
int64_t runtime_file_open(const char *path, int flags, int mode);
int64_t runtime_file_close(int fd);
int64_t runtime_file_read(int fd, void *buffer, uint64_t size);
int64_t runtime_file_pread(int fd, void *buffer, uint64_t size, int64_t offset);
int64_t runtime_file_write(int fd, const void *buffer, uint64_t size);
int64_t runtime_file_lseek(int fd, int64_t offset, int whence);
int64_t runtime_file_stat(const char *path, void *guest_stat);
int64_t runtime_file_fstat(int fd, void *guest_stat);
int64_t runtime_file_getdents(int fd, char *buffer, uint64_t size, int64_t *base);
void runtime_thread_keys_cleanup(void);
/* Guest-visible errno values are FreeBSD's. */
int32_t runtime_guest_errno(int host_errno);
void *runtime_low_map(size_t size, int prot);
/* Windows: reserves the PS4 user address range as a placeholder (no-op elsewhere). */
void runtime_memory_reserve(void);
uintptr_t runtime_ajm_resolve(const char *name);
void runtime_ajm_report(void);
/* AjmDecMp3ParseFrame (guest layout). */
typedef struct {
    uint64_t frame_size;
    uint32_t num_channels, samples_per_channel, bitrate, sample_rate, encoder_delay, num_frames, total_samples, ofl_type;
} Mp3Frame;
typedef struct Mp3Decoder Mp3Decoder;
int runtime_mp3_parse(const unsigned char *data, uint32_t size, int parse_ofl, int layer3_only, Mp3Frame *frame);
Mp3Decoder *runtime_mp3_open(void);
void runtime_mp3_close(Mp3Decoder *decoder);
void runtime_mp3_reset(Mp3Decoder *decoder);
int runtime_mp3_decode(Mp3Decoder *decoder, const unsigned char *data, int size, float *pcm, int max_samples, int *channels);
uintptr_t runtime_videodec_resolve(const char *name);
uintptr_t runtime_audio_resolve(const char *name);
void runtime_audio_report(void);
/* Host-side AudioOut ports (48 kHz; format as sceAudioOutOpen's param): handle or error. */
int32_t runtime_audio_host_open(int32_t type, uint32_t frames, uint32_t format);
int32_t runtime_audio_host_output(int32_t handle, const void *data, int pace);
int32_t runtime_audio_host_close(int32_t handle);
uintptr_t runtime_pad_resolve(const char *name);
void runtime_pad_report(void);
/* probe --gamepad: prints one "Gamepad:" line for the first pad found (returns 1) or none (0). */
int runtime_pad_describe(void);
uintptr_t runtime_rtc_resolve(const char *name);
const char *runtime_file_user_dir(void);
void runtime_savedata_configure(const char *title);
uintptr_t runtime_savedata_resolve(const char *name);
void runtime_savedata_report(void);
/* OrbisSaveDataParam (1328 bytes) of a save directory of the current title. */
#define RUNTIME_SAVE_PARAM_SIZE 1328
int runtime_savedata_param(int32_t user, const char *dir, void *param);
int runtime_savedata_icon(int32_t user, const char *dir, char *out, size_t size);
uint64_t runtime_savedata_size(int32_t user, const char *dir); /* bytes on disk, 0 unknown */
void runtime_thread_attach_host(const char *name);

/* Freeze report (runtime_freeze.c). Each thread keeps what it is blocked on in a wait slot:
 * the runtime's blocking calls mark it before they block and clear it after (a few stores).
 * Two levels are kept, so a GPU readback inside a file read shows both. */
#define RUNTIME_WAIT_LEVELS 2
typedef struct RuntimeWaitLevel {
    const char *kind;           /* static text ("mutex", "semaphore", ...); NULL: none */
    const void *object;         /* handle or address of what is waited on */
    const void *const *owner;   /* where the holder's thread record is kept (mutexes), or NULL */
    uintptr_t frame;            /* frame of the guest-called import: [frame] guest rbp, [frame+8] return */
    uint64_t since_ms;          /* runtime_wait_now_ms() when the wait started */
    char name[48];              /* object name (tail of long ones) */
} RuntimeWaitLevel;
typedef struct RuntimeWait {
    unsigned depth;             /* nesting; levels past RUNTIME_WAIT_LEVELS are only counted */
    uint32_t generation;        /* changes on every begin/end: readers detect a torn copy */
    /* TLG_WAIT_STATS=1 only (0 otherwise): what this thread spent blocked in marked waits, so a
     * stutter can be attributed to a thread and to the object it waited for. The outermost wait
     * is the one that counts (nested ones are inside it). */
    uint64_t wait_start_ns;     /* start of the outermost wait in progress (0: none counted) */
    uint64_t wait_ns;           /* summed time blocked */
    uint64_t wait_max_ns;       /* the longest single wait */
    uint64_t wait_max_end_ms;   /* runtime_wait_now_ms() when that wait ended */
    uint32_t wait_count;        /* marked waits counted */
    const char *wait_max_kind;  /* static text of the longest wait (NULL: none yet) */
    char wait_max_name[48];     /* its object name, if any */
    RuntimeWaitLevel level[RUNTIME_WAIT_LEVELS];
} RuntimeWait;
/* Frame of the calling (guest-called) function, for runtime_wait_begin. Use it in the ABI
 * entry itself: there [frame+8] is the guest call site and [frame] the guest frame chain. */
#define RUNTIME_GUEST_FRAME() ((uintptr_t)__builtin_frame_address(0))
uint64_t runtime_wait_now_ms(void);
/* The calling thread's slot: its guest thread record, else a host slot (NULL when all are taken). */
RuntimeWait *runtime_wait_slot(void);
void runtime_wait_slot_begin(RuntimeWait *w, const char *kind, const void *object, const char *name,
                             const void *const *owner, uintptr_t frame, uint64_t now_ms);
void runtime_wait_slot_end(RuntimeWait *w);
void runtime_wait_begin(const char *kind, const void *object, const char *name,
                        const void *const *owner, uintptr_t frame);
void runtime_wait_end(void);
/* Consistent copy of a slot another thread may be changing (0: it kept changing). */
int runtime_wait_copy(const RuntimeWait *from, RuntimeWait *to);
/* Thread record of the calling guest (or attached host) thread, NULL on other threads. */
void *runtime_thread_self(void);
typedef struct RuntimeThreadInfo {
    const void *record;         /* guest thread record (compare with a mutex owner); NULL: host slot */
    uint64_t os_tid;            /* gettid() / GetCurrentThreadId() */
    char name[32];
    int finished;
    RuntimeWait wait;
} RuntimeThreadInfo;
/* Guest threads and host wait slots; returns how many were written (at most max). */
size_t runtime_thread_snapshot(RuntimeThreadInfo *out, size_t max);
/* One line: "waiting on mutex 'name' 0x... for 1234 ms, held by 'Main'" ("running" without a
 * wait). threads/count resolve owners to names. */
void runtime_freeze_describe_wait(char *out, size_t size, const RuntimeWait *wait, uint64_t now_ms,
                                  const RuntimeThreadInfo *threads, size_t count);
/* Monitor state machine: called with the frames shown so far; returns 1 when a report is due
 * (once per freeze), 2 when frames resumed after a report, else 0. */
typedef struct FreezeMonitor {
    uint64_t seconds;           /* report after this long without a new frame (0: never) */
    uint64_t last_frames, last_change_ms;
    uint64_t freeze_ms;         /* after a 2: how long the frames had stopped */
    int reported;
} FreezeMonitor;
int runtime_freeze_step(FreezeMonitor *m, uint64_t frames, uint64_t now_ms);
/* Guest image range (guest offsets in the report) and the monitor thread (seconds 0: none). */
void runtime_freeze_start(uintptr_t image, uint64_t image_size, unsigned seconds);
/* Writes a report now (every thread: name, wait, RIP and frame chain) to out. */
void runtime_freeze_report(FILE *out, const char *head);
/* The registers of thread tid (os id) now: its RIP and up to max return addresses of its frame
 * chain; 0 when it could not be read. For the guest sampler (runtime_sampler.c). */
int runtime_freeze_sample(uint64_t tid, uintptr_t *rip, uintptr_t *chain, unsigned max, unsigned *frames);
/* Reads memory that may not be there (0 then) without faulting. */
int runtime_freeze_read(uintptr_t address, void *out, size_t size);
/* The game image runtime_freeze_start was given. */
void runtime_freeze_image(uintptr_t *base, uint64_t *size);
/* The profiling session's guest sampler (TLG_SESSION_DIR): starts once, after the image is
 * mapped. */
void runtime_sampler_start(void);
#endif
