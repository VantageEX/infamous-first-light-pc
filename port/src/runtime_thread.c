/* Guest threads on host pthreads. Each guest thread owns a FreeBSD-style TCB
 * (variant II: static TLS below the TCB). The loader rewrites the eboot's
 * `mov rax, fs:[0]` into `mov rax, gs:[0]`, so GS base = guest TCB while glibc
 * keeps FS. Priorities/affinity are recorded, not enforced by a PS4 scheduler; TLG_THREAD_PRIO=1
 * additionally carries the priority over to the host scheduler (see apply_prio).
 * Windows: GS is the TEB and cannot be moved. The guest TCB lives in a TLS slot of the TEB
 * instead, and the loader rewrites the load to `mov rax, gs:[0x1480+8*slot]` (TlsSlots),
 * the same 9-byte instruction with another displacement. */
#define _GNU_SOURCE
#include "runtime.h"
#include "runtime_trace.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <setjmp.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <asm/prctl.h>
#endif
#define ERR(n) ((int32_t)(UINT32_C(0x80020000)|(n)))
#define ATTR_MAGIC UINT32_C(0x41545452)
#define STACK_MARGIN (256*1024)
#define MIN_STACK (64*1024)
#define DEFAULT_STACK (1024*1024)
#define DEFAULT_PRIO 700

typedef void *(ABI *GuestEntry)(void *);
typedef struct ThreadAttr {
    uint32_t magic;
    int detached, policy, prio, inherit;
    uint64_t stack, guard, affinity;
    struct ThreadAttr *next;
} ThreadAttr;
typedef struct GuestThread {
    /* Guest-visible TCB is allocated separately; this is the ScePthread handle. */
    uint64_t *tcb;
    unsigned char *tls_block;
    pthread_t host;
    GuestEntry entry;
    void *argument, *result;
    ThreadAttr attr;
    char name[32];
    int detached, finished, joined, host_owned;
    uint64_t os_tid;  /* freeze report: matches the OS thread list */
    uint32_t fp_mxcsr; uint16_t fp_x87cw; /* the creator's FP control: see host_start */
    RuntimeWait wait; /* freeze report: what the thread is blocked on */
#ifdef _WIN32
    tlg_jmp_buf exit_jump; /* no SEH unwinding through guest frames */
#else
    jmp_buf exit_jump;
#endif
    struct GuestThread *next;
} GuestThread;

static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static ThreadAttr *attributes;
static GuestThread *threads;
static _Thread_local GuestThread *current;
static _Thread_local int32_t guest_errno;
static const unsigned char *tls_template;
static uint64_t tls_filesz, tls_memsz, tls_align=16;
static size_t created, joined_count, exited;
static int wait_stats(void);        /* TLG_WAIT_STATS: per-thread wait accounting, further down */
static void wait_stats_start(void); /* its monitor thread, started with the main thread */

void runtime_set_main_tls(const void *data,uint64_t filesz,uint64_t memsz,uint64_t align) {
    tls_template=data; tls_filesz=filesz; tls_memsz=memsz; tls_align=align ? align : 16;
}
static uint64_t tls_offset(void) { return (tls_memsz+tls_align-1)&~(tls_align-1); }
#ifdef _WIN32
/* TEB TlsSlots[64] start at gs:0x1480; slots 64+ live elsewhere and cannot be used. */
static DWORD tls_slot=TLS_OUT_OF_INDEXES;
uint32_t runtime_thread_tcb_offset(void) {
    if (tls_slot==TLS_OUT_OF_INDEXES) {
        tls_slot=TlsAlloc();
        if (tls_slot>=64) { fprintf(stderr,"STOP: no TEB TLS slot below 64 (got %lu)\n",tls_slot); exit(21); }
    }
    return 0x1480+8*(uint32_t)tls_slot;
}
static void set_gs(void *base) {
    runtime_thread_tcb_offset();
    if (!TlsSetValue(tls_slot,base)) { fprintf(stderr,"STOP: TlsSetValue failed (%lu)\n",GetLastError()); exit(21); }
}
#define tls_block_alloc(size) _aligned_malloc(size,64)
#else
static void set_gs(void *base) {
    if (syscall(SYS_arch_prctl,ARCH_SET_GS,(unsigned long)base)) { perror("STOP: arch_prctl(ARCH_SET_GS)"); exit(21); }
}
#define tls_block_alloc(size) aligned_alloc(64,size)
#endif
static uint64_t os_thread_id(void) {
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return (uint64_t)gettid();
#endif
}
/* ---- Guest priority (TLG_THREAD_PRIO=1) ----
 * PS4 priorities run 0x100 (highest) .. 0x2FF (lowest) with 0x2BC (700, the game's default) the
 * normal value (OpenOrbis UserService). Only a value above it lowers the host priority: nice
 * 0..19 (Linux) / BELOW_NORMAL..LOWEST (Windows), one step per 4 guest points. Raising the host
 * priority needs privileges and a wrong guess could starve the frame path, so it is never done;
 * a host that refuses the change keeps the recorded value. Off by default: the value is recorded
 * only, as it is on PS4. */
#define GUEST_PRIO_NORMAL 700
static int thread_prio_enabled(void) {
    static int value=-1;
    if (value<0) { const char *e=getenv("TLG_THREAD_PRIO"); value=e && e[0]=='1'; }
    return value;
}
static int host_nice(int prio) {
    if (prio<=GUEST_PRIO_NORMAL) return 0;
    int nice=(prio-GUEST_PRIO_NORMAL+3)/4;
    return nice>19 ? 19 : nice;
}
/* Applied on the thread itself at start (the creator does not know the host thread id yet) and on
 * the target thread for scePthreadSetprio. */
static void apply_prio(uint64_t os_tid,int prio) {
    if (!thread_prio_enabled() || !os_tid) return;
    int nice=host_nice(prio);
    if (!nice) return; /* the normal band and above: leave the host priority alone */
#ifdef _WIN32
    int level=nice<=5 ? THREAD_PRIORITY_BELOW_NORMAL : THREAD_PRIORITY_LOWEST;
    HANDLE thread=os_tid==os_thread_id() ? GetCurrentThread()
                                         : OpenThread(THREAD_SET_INFORMATION,FALSE,(DWORD)os_tid);
    if (!thread) return;
    if (!SetThreadPriority(thread,level) && runtime_verbose())
        fprintf(stderr,"Verbose: SetThreadPriority(%llu, %d) failed (%lu)\n",
                (unsigned long long)os_tid,level,GetLastError());
    if (thread!=GetCurrentThread()) CloseHandle(thread);
#else
    if (setpriority(PRIO_PROCESS,(id_t)os_tid,nice) && runtime_verbose())
        fprintf(stderr,"Verbose: setpriority(%llu, %d) failed (%d)\n",(unsigned long long)os_tid,nice,errno);
#endif
}
/* Build TCB/static TLS for the calling host thread and point GS at it. */
static void attach(GuestThread *t) {
    uint64_t offset=tls_offset();
    size_t total=offset+256;
    unsigned char *block=tls_block_alloc((total+63)&~(size_t)63);
    if (!block) { fputs("Cannot allocate guest TLS\n",stderr); exit(1); }
    memset(block,0,total);
    if (tls_filesz) memcpy(block,tls_template,tls_filesz);
    uint64_t *tcb=(uint64_t *)(block+offset);
    static uint64_t dtv[3];
    tcb[0]=(uint64_t)(uintptr_t)tcb;         /* tcb_self */
    tcb[1]=(uint64_t)(uintptr_t)dtv;         /* tcb_dtv (static module only) */
    tcb[2]=(uint64_t)(uintptr_t)t;           /* tcb_thread */
    t->tls_block=block; t->tcb=tcb;
    set_gs(tcb);
    t->os_tid=os_thread_id();
    current=t;
}
static GuestThread *new_thread(void) {
    GuestThread *t=calloc(1,sizeof(*t));
    if (!t) return NULL;
    t->attr=(ThreadAttr){.magic=ATTR_MAGIC,.policy=1,.prio=DEFAULT_PRIO,.stack=DEFAULT_STACK,.affinity=0x7f};
    return t;
}
static void publish(GuestThread *t) {
    pthread_mutex_lock(&lock); t->next=threads; threads=t; pthread_mutex_unlock(&lock);
}
/* Host threads that call into guest code without being created by the guest
 * (the main thread, test threads) get a record on first use. */
GuestThread *runtime_thread_current(void) {
    if (current) return current;
    GuestThread *t=new_thread();
    if (!t) { fputs("Cannot allocate guest thread\n",stderr); exit(1); }
    t->host=pthread_self(); t->host_owned=1;
    snprintf(t->name,sizeof(t->name),"host");
    attach(t); publish(t);
    return t;
}
/* Host threads that call guest code (HLE decoders invoking guest callbacks) need
 * a TCB/TLS like guest threads; the record lives until process exit. */
void runtime_thread_attach_host(const char *name) {
    GuestThread *t=runtime_thread_current();
    snprintf(t->name,sizeof(t->name),"%s",name ? name : "host");
    runtime_trace_thread_name(t->name);
}
void runtime_thread_attach_main(void) {
    GuestThread *t=runtime_thread_current();
    snprintf(t->name,sizeof(t->name),"main");
    runtime_trace_thread_name("main");
    if (wait_stats()) wait_stats_start();
}
static GuestThread *find_thread(void *handle) {
    GuestThread *found=NULL;
    pthread_mutex_lock(&lock);
    for (GuestThread *t=threads;t;t=t->next) if (t==handle) { found=t; break; }
    pthread_mutex_unlock(&lock);
    return found;
}
static ThreadAttr *find_attr(ThreadAttr **slot) {
    if (!slot || !*slot) return NULL;
    ThreadAttr *found=NULL;
    pthread_mutex_lock(&lock);
    for (ThreadAttr *a=attributes;a;a=a->next) if (a==*slot && a->magic==ATTR_MAGIC) { found=a; break; }
    pthread_mutex_unlock(&lock);
    return found;
}

static ABI void *thread_self(void) { return runtime_thread_current(); }
void *runtime_thread_self(void) { return current; }

/* ---- Wait slots (freeze report) ----
 * Guest threads keep theirs in the thread record; host threads (GPU library, decoders) that
 * block in a marked wait get one of a fixed set on first use, matched by OS thread id (a
 * thread that reuses the id of a finished one takes its slot over). When the set runs out, the
 * first slot whose owner has exited is handed over instead of leaving the thread unrecorded
 * (long sessions create and destroy host threads): the new owner starts with empty
 * TLG_WAIT_STATS accounting. */
#define HOST_WAIT_SLOTS 64
static struct { uint64_t os_tid; RuntimeWait wait; } host_waits[HOST_WAIT_SLOTS];
static unsigned host_wait_count;
static _Thread_local RuntimeWait *host_wait;
#define HOST_WAIT_CLAIMING UINT64_MAX /* os_tid of a slot being handed over */
/* 0 for a thread that never existed and for one that exited; a tid reused by a live thread only
 * makes the check pessimistic (that slot is left alone). */
static int thread_alive(uint64_t tid) {
#ifdef _WIN32
    HANDLE thread=OpenThread(SYNCHRONIZE,FALSE,(DWORD)tid);
    if (!thread) return 0;
    int alive=WaitForSingleObject(thread,0)==WAIT_TIMEOUT;
    CloseHandle(thread);
    return alive;
#else
    return syscall(SYS_tgkill,(pid_t)getpid(),(pid_t)tid,0)==0 || errno!=ESRCH;
#endif
}
uint64_t runtime_wait_now_ms(void) {
#ifdef _WIN32
    return GetTickCount64();
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_COARSE,&t);
    return (uint64_t)t.tv_sec*1000+(uint64_t)t.tv_nsec/1000000;
#endif
}
/* TLG_WAIT_STATS=1: per-thread wait accounting (runtime_wait_slot_begin/end) and a summary of it
 * every 5 s from a monitor thread started with the main thread. Off by default: marking a wait
 * costs one store then, no clock read. */
static int wait_stats(void) {
    static int value=-1;
    if (value<0) { const char *e=getenv("TLG_WAIT_STATS"); value=e && e[0]=='1'; }
    return value;
}
static uint64_t wait_now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*1000000000+(uint64_t)t.tv_nsec;
}
RuntimeWait *runtime_wait_slot(void) {
    if (current) return &current->wait;
    if (host_wait) return host_wait;
    /* No lock: this may run inside a fault handler, on a thread that faulted holding one. */
    uint64_t tid=os_thread_id();
    unsigned count=__atomic_load_n(&host_wait_count,__ATOMIC_ACQUIRE);
    for (unsigned i=0;i<count && i<HOST_WAIT_SLOTS && !host_wait;++i)
        if (__atomic_load_n(&host_waits[i].os_tid,__ATOMIC_ACQUIRE)==tid) host_wait=&host_waits[i].wait;
    if (!host_wait) {
        unsigned i=__atomic_fetch_add(&host_wait_count,1,__ATOMIC_ACQ_REL);
        if (i>=HOST_WAIT_SLOTS) {
            i=HOST_WAIT_SLOTS;
            for (unsigned j=0;j<HOST_WAIT_SLOTS;++j) {
                uint64_t owner=__atomic_load_n(&host_waits[j].os_tid,__ATOMIC_ACQUIRE);
                /* 0: a first-use claim is about to store its tid; HOST_WAIT_CLAIMING: another
                 * thread is taking this slot over. */
                if (!owner || owner==HOST_WAIT_CLAIMING || thread_alive(owner)) continue;
                /* Two threads may find the same dead owner: only the one that swaps it out
                 * resets the slot. */
                if (!__atomic_compare_exchange_n(&host_waits[j].os_tid,&owner,HOST_WAIT_CLAIMING,0,
                                                 __ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE)) continue;
                /* The generation stays odd while the slot is cleared, so a report reading it
                 * (runtime_wait_copy) retries instead of taking a torn copy. */
                RuntimeWait *w=&host_waits[j].wait;
                const uint32_t generation=__atomic_load_n(&w->generation,__ATOMIC_ACQUIRE)|1u;
                __atomic_store_n(&w->generation,generation,__ATOMIC_RELEASE);
                RuntimeWait empty={0};
                empty.generation=generation;
                memcpy(w,&empty,sizeof(empty));
                __atomic_store_n(&w->generation,generation+1,__ATOMIC_RELEASE);
                i=j; break;
            }
            if (i>=HOST_WAIT_SLOTS) return NULL; /* every slot is held by a live thread */
        }
        __atomic_store_n(&host_waits[i].os_tid,tid,__ATOMIC_RELEASE);
        host_wait=&host_waits[i].wait;
    }
    return host_wait;
}
/* The name is copied (the object may go away while a report reads it); long names keep
 * their tail, the distinctive part of a path. */
static void copy_wait_name(char *out,size_t size,const char *name) {
    size_t n=name ? strnlen(name,512) : 0;
    if (n<size) { memcpy(out,name ? name : "",n); out[n]=0; return; }
    memcpy(out,"...",3);
    memcpy(out+3,name+n-(size-4),size-4);
    out[size-1]=0;
}
void runtime_wait_slot_begin(RuntimeWait *w,const char *kind,const void *object,const char *name,
                             const void *const *owner,uintptr_t frame,uint64_t now_ms) {
    if (!w) return;
    unsigned depth=w->depth++;
    /* The outermost wait is the time the thread is blocked (nested waits happen inside it). */
    if (!depth && wait_stats()) { w->wait_start_ns=wait_now_ns(); ++w->wait_count; }
    if (depth>=RUNTIME_WAIT_LEVELS) return;
    RuntimeWaitLevel *l=&w->level[depth];
    __atomic_add_fetch(&w->generation,1,__ATOMIC_ACQ_REL); /* odd: being written */
    l->object=object; l->owner=owner; l->frame=frame; l->since_ms=now_ms;
    copy_wait_name(l->name,sizeof(l->name),name);
    __atomic_store_n(&l->kind,kind,__ATOMIC_RELEASE);
    __atomic_add_fetch(&w->generation,1,__ATOMIC_RELEASE);
}
void runtime_wait_slot_end(RuntimeWait *w) {
    if (!w || !w->depth) return;
    unsigned depth=--w->depth;
    if (!depth && w->wait_start_ns) { /* wait_start_ns is set only for a counted begin */
        uint64_t spent=wait_now_ns()-w->wait_start_ns;
        w->wait_start_ns=0;
        w->wait_ns+=spent;
        if (spent>w->wait_max_ns) {
            w->wait_max_ns=spent; w->wait_max_end_ms=runtime_wait_now_ms();
            w->wait_max_kind=w->level[0].kind;
            copy_wait_name(w->wait_max_name,sizeof(w->wait_max_name),w->level[0].name);
        }
    }
    if (depth>=RUNTIME_WAIT_LEVELS) return;
    __atomic_add_fetch(&w->generation,1,__ATOMIC_ACQ_REL);
    __atomic_store_n(&w->level[depth].kind,NULL,__ATOMIC_RELEASE);
    __atomic_add_fetch(&w->generation,1,__ATOMIC_RELEASE);
}
void runtime_wait_begin(const char *kind,const void *object,const char *name,const void *const *owner,uintptr_t frame) {
    runtime_wait_slot_begin(runtime_wait_slot(),kind,object,name,owner,frame,runtime_wait_now_ms());
    runtime_trace_wait_begin(kind,name); /* the profiling build's zone (runtime_trace.h) */
}
void runtime_wait_end(void) { runtime_trace_wait_end(); runtime_wait_slot_end(runtime_wait_slot()); }
int runtime_wait_copy(const RuntimeWait *from,RuntimeWait *to) {
    for (int attempt=0;attempt<8;++attempt) {
        uint32_t before=__atomic_load_n(&from->generation,__ATOMIC_ACQUIRE);
        if (before&1) continue;
        memcpy(to,(const void *)from,sizeof(*to));
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        if (__atomic_load_n(&from->generation,__ATOMIC_ACQUIRE)==before) return 1;
    }
    memset(to,0,sizeof(*to));
    return 0;
}
size_t runtime_thread_snapshot(RuntimeThreadInfo *out,size_t max) {
    size_t n=0;
    pthread_mutex_lock(&lock);
    for (GuestThread *t=threads;t && n<max;t=t->next,++n) {
        out[n]=(RuntimeThreadInfo){.record=t,.os_tid=t->os_tid,.finished=t->finished};
        memcpy(out[n].name,t->name,sizeof(out[n].name)); out[n].name[sizeof(out[n].name)-1]=0;
        runtime_wait_copy(&t->wait,&out[n].wait);
    }
    unsigned hosts=__atomic_load_n(&host_wait_count,__ATOMIC_ACQUIRE);
    for (unsigned i=0;i<hosts && i<HOST_WAIT_SLOTS && n<max;++i,++n) {
        const uint64_t tid=__atomic_load_n(&host_waits[i].os_tid,__ATOMIC_ACQUIRE);
        out[n]=(RuntimeThreadInfo){.os_tid=tid==HOST_WAIT_CLAIMING ? 0 : tid};
        runtime_wait_copy(&host_waits[i].wait,&out[n].wait);
    }
    pthread_mutex_unlock(&lock);
    return n;
}
static ABI int32_t *guest_error(void) { return &guest_errno; }
int32_t *runtime_errno(void) { return &guest_errno; }

/* ---- Wait statistics monitor (TLG_WAIT_STATS=1) ----
 * Every 5 s: one line per thread whose longest wait ended in the window, so a 30-150 ms stutter
 * can be attributed to a thread and to the object it waited for. Threads that waited nothing new
 * stay quiet, and the report is off unless the toggle is set. */
#define WAIT_STATS_PERIOD_MS 5000
#define WAIT_STATS_THREADS 256
static void *wait_stats_monitor(void *argument) {
    (void)argument;
    static RuntimeThreadInfo infos[WAIT_STATS_THREADS];
    for (;;) {
#ifdef _WIN32
        compat_sleep_ns((uint64_t)WAIT_STATS_PERIOD_MS*1000000);
#else
        struct timespec nap={WAIT_STATS_PERIOD_MS/1000,(long)(WAIT_STATS_PERIOD_MS%1000)*1000000};
        while (nanosleep(&nap,&nap) && errno==EINTR) {}
#endif
        const uint64_t now=runtime_wait_now_ms();
        const size_t count=runtime_thread_snapshot(infos,WAIT_STATS_THREADS);
        int printed=0;
        for (size_t i=0;i<count;++i) {
            const RuntimeWait *w=&infos[i].wait;
            if (!w->wait_count || w->wait_max_end_ms+WAIT_STATS_PERIOD_MS<now) continue;
            if (!printed) {
                printf("Runtime: wait stats (last %u s, TLG_WAIT_STATS)\n",WAIT_STATS_PERIOD_MS/1000);
                printed=1;
            }
            printf("  %s (tid %llu): %u waits, %.1f ms blocked, longest %.1f ms in '%s'",
                   infos[i].record ? infos[i].name : "host thread",(unsigned long long)infos[i].os_tid,
                   w->wait_count,(double)w->wait_ns/1e6,(double)w->wait_max_ns/1e6,
                   w->wait_max_kind ? w->wait_max_kind : "wait");
            if (w->wait_max_name[0]) printf(" '%s'",w->wait_max_name);
            printf(" (%llu ms ago)\n",(unsigned long long)(now>=w->wait_max_end_ms ? now-w->wait_max_end_ms : 0));
        }
        if (printed) fflush(stdout);
    }
    return NULL;
}
static void wait_stats_start(void) {
    static int started;
    if (__atomic_exchange_n(&started,1,__ATOMIC_ACQ_REL)) return;
    pthread_t thread;
    if (pthread_create(&thread,NULL,wait_stats_monitor,NULL)) {
        fputs("Runtime: wait stats thread not started\n",stderr);
        return;
    }
#ifndef _WIN32
    pthread_setname_np(thread,"tlg:waitstats");
#endif
    pthread_detach(thread);
    printf("Runtime: wait stats every %u ms (TLG_WAIT_STATS)\n",WAIT_STATS_PERIOD_MS);
}

static ABI int32_t attr_init(ThreadAttr **out) {
    if (!out) return ERR(22);
    ThreadAttr *a=calloc(1,sizeof(*a));
    if (!a) return ERR(12);
    *a=(ThreadAttr){.magic=ATTR_MAGIC,.policy=1,.prio=DEFAULT_PRIO,.inherit=4,.stack=DEFAULT_STACK,.guard=4096,.affinity=0x7f};
    pthread_mutex_lock(&lock); a->next=attributes; attributes=a; pthread_mutex_unlock(&lock);
    *out=a; return 0;
}
static ABI int32_t attr_destroy(ThreadAttr **slot) {
    ThreadAttr *a=find_attr(slot);
    if (!a) return ERR(22);
    pthread_mutex_lock(&lock);
    ThreadAttr **link=&attributes;
    while (*link!=a) link=&(*link)->next;
    *link=a->next; a->magic=0;
    pthread_mutex_unlock(&lock);
    free(a); *slot=NULL; return 0;
}
static ABI int32_t attr_get(void *thread,ThreadAttr **out) {
    ThreadAttr *a=find_attr(out);
    if (!a || !thread) return ERR(22);
    GuestThread *t=find_thread(thread);
    if (!t) return ERR(3);
    ThreadAttr *next=a->next;
    *a=t->attr; a->magic=ATTR_MAGIC; a->next=next;
    a->detached=t->detached;
    return 0;
}
static ABI int32_t attr_set_stack(ThreadAttr **slot,uint64_t size) {
    ThreadAttr *a=find_attr(slot);
    if (!a || size<16384) return ERR(22);
    a->stack=size; return 0;
}
static ABI int32_t attr_get_stack(ThreadAttr **slot,uint64_t *size) {
    ThreadAttr *a=find_attr(slot);
    if (!a || !size) return ERR(22);
    *size=a->stack; return 0;
}
static ABI int32_t attr_set_detach(ThreadAttr **slot,int state) {
    ThreadAttr *a=find_attr(slot);
    if (!a || (state!=0 && state!=1)) return ERR(22);
    a->detached=state; return 0;
}
static ABI int32_t attr_get_detach(ThreadAttr **slot,int *state) {
    ThreadAttr *a=find_attr(slot);
    if (!a || !state) return ERR(22);
    *state=a->detached; return 0;
}
static ABI int32_t attr_set_policy(ThreadAttr **slot,int policy) {
    ThreadAttr *a=find_attr(slot);
    if (!a || policy<1 || policy>3) return ERR(22);
    a->policy=policy; return 0;
}
static ABI int32_t attr_set_inherit(ThreadAttr **slot,int inherit) {
    ThreadAttr *a=find_attr(slot);
    if (!a || (inherit!=0 && inherit!=4)) return ERR(22);
    a->inherit=inherit; return 0;
}
static ABI int32_t attr_set_param(ThreadAttr **slot,const int *param) {
    ThreadAttr *a=find_attr(slot);
    if (!a || !param) return ERR(22);
    a->prio=*param; return 0;
}
static ABI int32_t attr_get_param(ThreadAttr **slot,int *param) {
    ThreadAttr *a=find_attr(slot);
    if (!a || !param) return ERR(22);
    *param=a->prio; return 0;
}
static ABI int32_t attr_set_affinity(ThreadAttr **slot,uint64_t mask) {
    ThreadAttr *a=find_attr(slot);
    if (!a) return ERR(22);
    a->affinity=mask; return 0;
}
static ABI int32_t attr_affinity(ThreadAttr **slot,uint64_t *mask) {
    ThreadAttr *a=find_attr(slot);
    if (!a || !mask) return ERR(22);
    *mask=a->affinity; return 0;
}
static ABI int32_t attr_set_guard(ThreadAttr **slot,uint64_t size) {
    ThreadAttr *a=find_attr(slot);
    if (!a) return ERR(22);
    a->guard=size; return 0;
}

/* Linux thread names hold 15 characters; longer ones would be rejected. */
static void set_host_name(const char *name) {
    char host[16]={0};
    memcpy(host,name,strnlen(name,sizeof(host)-1));
    pthread_setname_np(pthread_self(),host);
    runtime_trace_thread_name(name); /* the whole name, in the profiling build */
}
#ifdef _WIN32
#define exit_setjmp(t) tlg_setjmp((t)->exit_jump)
#define exit_longjmp(t) tlg_longjmp((t)->exit_jump,1)
#else
#define exit_setjmp(t) setjmp((t)->exit_jump)
#define exit_longjmp(t) longjmp((t)->exit_jump,1)
#endif
static void (*thread_start_hook)(void);
void runtime_set_thread_start_hook(void (*hook)(void)) { thread_start_hook=hook; }
static void *host_start(void *p) {
    GuestThread *t=p;
    attach(t);
    if (thread_start_hook) thread_start_hook();
    /* The creator's SSE (MXCSR) and x87 control words, as FreeBSD's libthr gives a new thread
     * and as Linux happens to copy them on clone. Windows starts every thread at its defaults:
     * inFAMOUS turns on flush-to-zero (MXCSR bit 15) in a thread, then creates its job workers
     * from it; without the bit there, denormals reach divisions in the particle/light code
     * (overexposed fire and smoke, flickering headlights, Windows only). */
    __asm__ volatile("ldmxcsr %0\n\tfldcw %1" : : "m"(t->fp_mxcsr), "m"(t->fp_x87cw));
    set_host_name(t->name);
    apply_prio(t->os_tid,t->attr.prio); /* the priority is ours now; the creator had no host tid */
    if (!exit_setjmp(t)) t->result=t->entry(t->argument);
    runtime_thread_keys_cleanup();
    pthread_mutex_lock(&lock); t->finished=1; ++exited; pthread_mutex_unlock(&lock);
    return t->result;
}
static int32_t create(GuestThread **out,ThreadAttr **attr_slot,GuestEntry entry,void *argument,const char *name) {
    if (!out || !entry) return ERR(22);
    ThreadAttr *a=NULL;
    if (attr_slot && !(a=find_attr(attr_slot))) return ERR(22);
    GuestThread *t=new_thread();
    if (!t) return ERR(12);
    if (a) { t->attr=*a; t->attr.next=NULL; }
    t->entry=entry; t->argument=argument; t->detached=t->attr.detached;
    __asm__ volatile("stmxcsr %0\n\tfnstcw %1" : "=m"(t->fp_mxcsr), "=m"(t->fp_x87cw));
    snprintf(t->name,sizeof(t->name),"%s",name ? name : "guest");
    pthread_attr_t host;
    pthread_attr_init(&host);
    uint64_t stack=t->attr.stack<MIN_STACK ? MIN_STACK : t->attr.stack;
    /* Stacks below 1 TiB as on PS4; guest code may pack stack addresses. */
    size_t stack_bytes=(size_t)stack+STACK_MARGIN;
#ifdef _WIN32
    /* Windows places the stack itself (low: no high-entropy ASLR) and commits all of it:
     * guest code has no stack probes and would step over the guard page. */
    pthread_attr_setstacksize(&host,stack_bytes);
#else
    void *stack_memory=runtime_low_map(stack_bytes,PROT_READ|PROT_WRITE);
    if (stack_memory) pthread_attr_setstack(&host,stack_memory,stack_bytes);
    else pthread_attr_setstacksize(&host,stack_bytes);
#endif
    publish(t);
    /* Publish the handle before the thread can run and inspect itself. */
    *out=t;
    int e=pthread_create(&t->host,&host,host_start,t);
    pthread_attr_destroy(&host);
    if (e) { fprintf(stderr,"STOP: host pthread_create failed: %d\n",e); exit(21); }
    if (t->detached) pthread_detach(t->host);
    pthread_mutex_lock(&lock); ++created; pthread_mutex_unlock(&lock);
    printf("Runtime: guest thread '%s' created (stack=%llu, prio=%d)\n",t->name,(unsigned long long)stack,t->attr.prio);
    return 0;
}
static ABI int32_t thread_create(GuestThread **out,ThreadAttr **attr,GuestEntry entry,void *argument,const char *name) {
    return create(out,attr,entry,argument,name);
}
static ABI int32_t thread_join(GuestThread *t,void **result) {
    if (!find_thread(t) || t->host_owned) return ERR(3);
    if (t==current) return ERR(11);
    if (t->detached || t->joined) return ERR(22);
    t->joined=1;
    void *value=NULL;
    int e=pthread_join(t->host,&value);
    if (e) { fprintf(stderr,"STOP: host pthread_join failed: %d\n",e); exit(21); }
    if (result) *result=t->result;
    pthread_mutex_lock(&lock); ++joined_count; pthread_mutex_unlock(&lock);
    return 0;
}
static ABI int32_t thread_detach(GuestThread *t) {
    if (!find_thread(t)) return ERR(3);
    if (t->detached) return ERR(22);
    t->detached=1;
    if (!t->host_owned) pthread_detach(t->host);
    return 0;
}
static ABI __attribute__((noreturn)) void thread_exit(void *value) {
    GuestThread *t=runtime_thread_current();
    if (t->host_owned) { fputs("STOP: pthread_exit on host-owned/main thread\n",stderr); exit(21); }
    t->result=value;
    /* No host unwinder: guest frames have no registered FDEs. */
    exit_longjmp(t);
}
static ABI int32_t thread_yield(void) { sched_yield(); return 0; }
static ABI int32_t thread_get_prio(GuestThread *t,int *prio) {
    if (!find_thread(t)) return ERR(3);
    if (!prio) return ERR(22);
    *prio=t->attr.prio; return 0;
}
static ABI int32_t thread_set_prio(GuestThread *t,int prio) {
    if (!find_thread(t)) return ERR(3);
    t->attr.prio=prio;
    apply_prio(t->os_tid,prio); /* 0 until the thread has started: it applies its own then */
    return 0;
}
static ABI int32_t thread_set_affinity(GuestThread *t,uint64_t mask) {
    if (!find_thread(t)) return ERR(3);
    t->attr.affinity=mask; return 0;
}
static ABI int32_t thread_get_affinity(GuestThread *t,uint64_t *mask) {
    if (!find_thread(t)) return ERR(3);
    if (!mask) return ERR(22);
    *mask=t->attr.affinity; return 0;
}
static ABI int32_t thread_rename(GuestThread *t,const char *name) {
    if (!find_thread(t)) return ERR(3);
    if (!name) return ERR(22);
    snprintf(t->name,sizeof(t->name),"%s",name);
    if (t==current) set_host_name(t->name);
    return 0;
}
static ABI int32_t thread_equal(GuestThread *a,GuestThread *b) { return a==b; }
static ABI int32_t thread_get_name(GuestThread *t,char *name) {
    if (!find_thread(t)) return ERR(3);
    if (!name) return ERR(22);
    snprintf(name,32,"%s",t->name); return 0;
}
/* Small stable per-thread id (the TID the kernel would report is not observable by the game). */
static ABI int32_t thread_get_id(void) {
    static int32_t next=1;
    static _Thread_local int32_t id;
    if (!id) id=__atomic_fetch_add(&next,1,__ATOMIC_RELAXED);
    return id;
}
/* The guest-provided stack memory is ignored: host threads always get a host stack of the requested size. */
static ABI int32_t attr_set_stack_addr(ThreadAttr **slot,void *address) {
    (void)address;
    return find_attr(slot) ? 0 : ERR(22);
}

/* POSIX (libScePosix) variants return positive errno values. */
static int32_t posix(int32_t r) { return r ? (int32_t)((uint32_t)r&0xffff) : 0; }
static ABI int32_t posix_attr_init(ThreadAttr **a) { return posix(attr_init(a)); }
static ABI int32_t posix_attr_destroy(ThreadAttr **a) { return posix(attr_destroy(a)); }
static ABI int32_t posix_attr_set_detach(ThreadAttr **a,int s) { return posix(attr_set_detach(a,s)); }
static ABI int32_t posix_attr_set_stack(ThreadAttr **a,uint64_t s) { return posix(attr_set_stack(a,s)); }
static ABI int32_t posix_attr_set_param(ThreadAttr **a,const int *p) { return posix(attr_set_param(a,p)); }
static ABI int32_t posix_create(GuestThread **t,ThreadAttr **a,GuestEntry e,void *arg) { return posix(create(t,a,e,arg,"posix")); }
static ABI int32_t posix_create_name(GuestThread **t,ThreadAttr **a,GuestEntry e,void *arg,const char *name) { return posix(create(t,a,e,arg,name)); }
static ABI int32_t posix_join(GuestThread *t,void **r) { return posix(thread_join(t,r)); }

uintptr_t runtime_thread_resolve(const char *name) {
    static const struct { const char *nid; void *fn; } table[]={
        {"aI+OeCz8xrQ#libkernel",thread_self}, {"EotR8a3ASf4#libkernel",thread_self},
        {"9BcDykPmo1I#libkernel",guest_error},
        {"nsYoNRywwNg#libkernel",attr_init}, {"62KCwEMmzcM#libkernel",attr_destroy},
        {"x1X76arYMxU#libkernel",attr_get}, {"8+s5BzZjxSg#libkernel",attr_affinity},
        {"UTXzJbWhhTE#libkernel",attr_set_stack}, {"-Wreprtu0Qs#libkernel",attr_set_detach},
        {"4+h9EzwKF4I#libkernel",attr_set_policy}, {"DzES9hQF4f4#libkernel",attr_set_param},
        {"3qxgM4ezETA#libkernel",attr_set_affinity}, {"eXbUSpEaTsA#libkernel",attr_set_inherit},
        {"JaRMy+QcpeU#libkernel",attr_get_detach}, {"-fA+7ZlGDQs#libkernel",attr_get_stack},
        {"FXPWHNk8Of0#libkernel",attr_get_param}, {"El+cQ20DynU#libkernel",attr_set_guard},
        {"6UgtwV+0zb4#libkernel",thread_create}, {"onNY9Byn-W8#libkernel",thread_join},
        {"4qGrR6eoP9Y#libkernel",thread_detach}, {"3kg7rT0NQIs#libkernel",thread_exit},
        {"T72hz6ffq08#libkernel",thread_yield}, {"1tKyG7RlMJo#libkernel",thread_get_prio},
        {"W0Hpm2X0uPE#libkernel",thread_set_prio}, {"bt3CTBKmGyI#libkernel",thread_set_affinity}, {"rcrVFJsQWRY#libkernel",thread_get_affinity},
        {"GBUY7ywdULE#libkernel",thread_rename}, {"3PtV6p3QNX4#libkernel",thread_equal},
        {"wtkt-teR1so#libkernel",posix_attr_init}, {"zHchY8ft5pk#libkernel",posix_attr_destroy},
        {"E+tyo3lp5Lw#libkernel",posix_attr_set_detach}, {"2Q0z6rnBrTE#libkernel",posix_attr_set_stack},
        {"euKRgm0Vn2M#libkernel",posix_attr_set_param}, {"OxhIB8LB-PQ#libkernel",posix_create},
        {"Jmi+9w9u0E4#libkernel",posix_create_name}, {"h9CcP3J0oVM#libkernel",posix_join},
        {"FJrT5LuUBAU#libkernel",thread_exit},
        {"How7B8Oet6k#libkernel",thread_get_name}, {"EI-5-jlq2dE#libkernel",thread_get_id},
        {"F+yfmduIBB8#libkernel",attr_set_stack_addr},
    };
    for (size_t i=0;i<sizeof(table)/sizeof(*table);++i)
        if (!strcmp(name,table[i].nid)) return (uintptr_t)table[i].fn;
    return 0;
}
void runtime_thread_report(void) {
    pthread_mutex_lock(&lock);
    printf("Runtime: guest threads created=%zu, exited=%zu, joined=%zu\n",created,exited,joined_count);
    pthread_mutex_unlock(&lock);
}

