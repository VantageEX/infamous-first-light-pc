/* Guest file system: PS4 mount points mapped onto host directories.
 *   /app0, /hostapp  -> game package root (read-only by convention)
 *   /temp0, /download0, /data, and mounts added by SaveData -> user directory
 * Guest descriptors are small integers in our own table; stdio 0-2 pass through.
 * Paths containing ".." components are rejected rather than normalized. */
#define _GNU_SOURCE
#include "runtime.h"
#include "gpu/tlggpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
/* 64-bit sizes and times; directories are never opened as CRT descriptors. */
typedef struct _stat64 HostStat;
#define host_stat(path,s) _stat64(path,s)
#define host_fstat(fd,s) _fstat64(fd,s)
#define host_lseek _lseeki64
#define O_DIRECTORY 0x40000000 /* tlg marker, removed before _open */
#else
typedef struct stat HostStat;
#define host_stat(path,s) stat(path,s)
#define host_fstat(fd,s) fstat(fd,s)
#define host_lseek lseek
#endif
#define ERR(n) ((int32_t)(UINT32_C(0x80020000)|(n)))
#define MAX_FILES 1024
#define MAX_MOUNTS 16

typedef struct { int64_t sec, nsec; } GuestTimespec;
typedef struct {
    uint32_t dev, ino;
    uint16_t mode, nlink;
    uint32_t uid, gid, rdev;
    GuestTimespec atime, mtime, ctime;
    int64_t size, blocks;
    uint32_t blksize, flags, gen;
    int32_t lspare;
    GuestTimespec birthtime;
} GuestStat;
_Static_assert(sizeof(GuestStat)==120,"FreeBSD stat layout");

typedef struct { char *names; size_t count, *offsets; unsigned char *types; } Listing;
/* Windows: `offset` is the descriptor's position, kept here (under `io`) instead of in the CRT/
 * Win32 file pointer. Positioned reads there move that pointer (compat_pread restores it, racing
 * with any other thread's read on the descriptor), and the CRT's read() reports a GPU-protected
 * destination page as EINVAL, not EFAULT, so it was never retried: short or misplaced reads made
 * inFAMOUS say "the disc or hard disk is corrupted" at the graffiti. Reads at the position go
 * through compat_pread like pread. */
typedef struct { int used, host; Listing *dir; size_t position; char path[512];
                 int64_t offset; int append; pthread_mutex_t io; } File;
typedef struct { char guest[64]; char host[512]; } Mount;
static File files[MAX_FILES];
static Mount mounts[MAX_MOUNTS];
static size_t mount_count, opens, reads, writes, missing;
static size_t bounce_reads, direct_reads; /* TLG_FILE_DIRECT: which path each chunk of a read took */
static uint64_t bytes_read;
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;

int runtime_file_mount(const char *guest,const char *host) {
    pthread_mutex_lock(&lock);
    for (size_t i=0;i<mount_count;++i) if (!strcmp(mounts[i].guest,guest)) {
        snprintf(mounts[i].host,sizeof(mounts[i].host),"%s",host);
        pthread_mutex_unlock(&lock); return 0;
    }
    if (mount_count==MAX_MOUNTS || strlen(guest)>=64 || strlen(host)>=512) { pthread_mutex_unlock(&lock); return -1; }
    snprintf(mounts[mount_count].guest,64,"%s",guest);
    snprintf(mounts[mount_count].host,512,"%s",host);
    ++mount_count;
    pthread_mutex_unlock(&lock);
    return 0;
}
void runtime_file_unmount(const char *guest) {
    pthread_mutex_lock(&lock);
    for (size_t i=0;i<mount_count;++i) if (!strcmp(mounts[i].guest,guest)) {
        mounts[i]=mounts[--mount_count]; break;
    }
    pthread_mutex_unlock(&lock);
}
static char user_root[512]="user";
const char *runtime_file_user_dir(void) { return user_root; }
void runtime_file_configure(const char *app0,const char *user) {
    char path[600];
    snprintf(user_root,sizeof(user_root),"%s",user);
    runtime_file_mount("/app0",app0);
    runtime_file_mount("/hostapp",app0);
    const char *writable[]={"temp0","download0","data"};
    mkdir(user,0755);
    for (int i=0;i<3;++i) {
        snprintf(path,sizeof(path),"%s/%s",user,writable[i]);
        mkdir(path,0755);
        char guest[32]; snprintf(guest,sizeof(guest),"/%s",writable[i]);
        runtime_file_mount(guest,path);
    }
}
/* Resolve a guest path to a host path; returns 0 or a host errno. */
static int translate(const char *guest,char *out,size_t size) {
    if (!guest || !*guest) return ENOENT;
    char buffer[1024];
    if (guest[0]!='/') snprintf(buffer,sizeof(buffer),"/app0/%s",guest);
    else snprintf(buffer,sizeof(buffer),"%s",guest);
    for (const char *p=buffer;(p=strstr(p,".."));p+=2)
        if ((p==buffer || p[-1]=='/') && (p[2]==0 || p[2]=='/')) return EACCES;
    pthread_mutex_lock(&lock);
    size_t best=0; const Mount *m=NULL;
    for (size_t i=0;i<mount_count;++i) {
        size_t n=strlen(mounts[i].guest);
        if (!strncmp(buffer,mounts[i].guest,n) && (buffer[n]=='/' || !buffer[n]) && n>best) { best=n; m=&mounts[i]; }
    }
    int result=0;
    if (!m) result=ENOENT;
    else if ((size_t)snprintf(out,size,"%s%s",m->host,buffer+best)>=size) result=ENAMETOOLONG;
    pthread_mutex_unlock(&lock);
    if (result==ENOENT) fprintf(stderr,"Runtime: no mount for guest path %s\n",guest);
    return result;
}
static int host_flags(int flags) {
    int r;
    switch (flags&3) { case 0: r=O_RDONLY; break; case 1: r=O_WRONLY; break; default: r=O_RDWR; }
    if (flags&0x4) r|=O_NONBLOCK;
    if (flags&0x8) r|=O_APPEND;
    if (flags&0x80) r|=O_SYNC;
    if (flags&0x200) r|=O_CREAT;
    if (flags&0x400) r|=O_TRUNC;
    if (flags&0x800) r|=O_EXCL;
    if (flags&0x20000) r|=O_DIRECTORY;
#ifdef _WIN32
    r|=O_BINARY;
#endif
    return r|O_CLOEXEC;
}
static void free_listing(Listing *l) { if (l) { free(l->names); free(l->offsets); free(l->types); free(l); } }
static Listing *list_directory(const char *path) {
    DIR *d=opendir(path);
    if (!d) return NULL;
    Listing *l=calloc(1,sizeof(*l));
    size_t capacity=0,bytes=0,cap_names=0;
    struct dirent *e;
    while (l && (e=readdir(d))) {
        size_t n=strlen(e->d_name)+1;
        if (l->count==capacity) {
            capacity=capacity ? capacity*2 : 64;
            l->offsets=realloc(l->offsets,capacity*sizeof(size_t));
            l->types=realloc(l->types,capacity);
        }
        if (bytes+n>cap_names) { cap_names=(bytes+n)*2; l->names=realloc(l->names,cap_names); }
        if (!l->offsets || !l->types || !l->names) { fputs("Out of memory listing directory\n",stderr); exit(1); }
        memcpy(l->names+bytes,e->d_name,n);
        l->offsets[l->count]=bytes;
#ifdef _WIN32
        /* MinGW's dirent has no d_type. */
        char child[1100]; HostStat entry;
        snprintf(child,sizeof(child),"%s/%s",path,e->d_name);
        l->types[l->count]=host_stat(child,&entry) ? 0 : S_ISDIR(entry.st_mode) ? 4 : S_ISREG(entry.st_mode) ? 8 : 0;
#else
        unsigned char type=e->d_type;
        if (type==DT_LNK || type==DT_UNKNOWN) {
            struct stat entry;
            if (!fstatat(dirfd(d),e->d_name,&entry,0))
                type=S_ISDIR(entry.st_mode) ? DT_DIR : S_ISREG(entry.st_mode) ? DT_REG : type;
        }
        l->types[l->count]=type==DT_DIR ? 4 : type==DT_REG ? 8 : type==DT_LNK ? 10 : 0;
#endif
        ++l->count; bytes+=n;
    }
    closedir(d);
    return l;
}
static void convert_stat(const HostStat *s,GuestStat *g) {
    memset(g,0,sizeof(*g));
    g->dev=(uint32_t)s->st_dev; g->ino=(uint32_t)s->st_ino;
    g->nlink=(uint16_t)s->st_nlink;
    g->size=s->st_size;
#ifdef _WIN32
    /* FreeBSD mode bits: directories rwxr-xr-x, files rw-r--r--. */
    g->mode=(uint16_t)(S_ISDIR(s->st_mode) ? 0040755 : 0100644);
    g->blocks=(s->st_size+511)/512; g->blksize=4096;
    g->atime=(GuestTimespec){s->st_atime,0};
    g->mtime=(GuestTimespec){s->st_mtime,0};
    g->ctime=(GuestTimespec){s->st_ctime,0};
#else
    g->mode=(uint16_t)s->st_mode;
    g->blocks=s->st_blocks; g->blksize=(uint32_t)s->st_blksize;
    g->atime=(GuestTimespec){s->st_atim.tv_sec,s->st_atim.tv_nsec};
    g->mtime=(GuestTimespec){s->st_mtim.tv_sec,s->st_mtim.tv_nsec};
    g->ctime=(GuestTimespec){s->st_ctim.tv_sec,s->st_ctim.tv_nsec};
#endif
    g->birthtime=g->ctime;
}
static File *get(int fd) {
    if (fd<3 || fd>=MAX_FILES || !files[fd].used) return NULL;
    return &files[fd];
}
/* TLG_AUDIO_TRACE=1: sound file opens and failed reads (missing game sounds). */
static int audio_trace(void) { static int v=-1; if (v<0) { const char *e=getenv("TLG_AUDIO_TRACE"); v=e && e[0]=='1'; } return v; }
/* Game mounts (including linked mod overlays) are read-only. Saves use other mounts. */
static int game_path(const char *p) {
    if (!p || !*p) return 0;
    if (*p!='/') return 1;
    return (!strncmp(p,"/app0",5) && (!p[5] || p[5]=='/')) ||
           (!strncmp(p,"/hostapp",8) && (!p[8] || p[8]=='/'));
}
/* All operations return >=0 or -(host errno); wrappers adapt the convention. */
static void log_io_problem(const char *what,const char *detail,int64_t n,int error);
static int64_t do_open(const char *guest,int flags,int mode) {
    if (game_path(guest) && (flags & (3|0x8|0x200|0x400|0x800))) {
        if (runtime_verbose()) log_io_problem("open (read-only game mount)",guest,-EROFS,EROFS);
        return -EROFS;
    }
    char path[1024];
    int e=translate(guest,path,sizeof(path));
    if (e) {
        if (runtime_verbose()) log_io_problem("open (path not translated)",guest,-e,e);
        return -e;
    }
#ifdef _WIN32
    /* Windows cannot open a directory as a CRT descriptor: directories are listings only. */
    (void)mode;
    HostStat s;
    Listing *dir=NULL;
    int host=-1;
    if (!host_stat(path,&s) && S_ISDIR(s.st_mode)) {
        if (flags&3) return -EISDIR;
        if (!(dir=list_directory(path))) return -EACCES;
    } else if (flags&0x20000) {
        e=errno==ENOENT ? ENOENT : ENOTDIR;
        if (e==ENOENT) { ++missing; printf("Runtime: open(%s) -> not found\n",guest); }
        return -e;
    } else {
        host=open(path,host_flags(flags)&~O_DIRECTORY,_S_IREAD|_S_IWRITE);
        if (host<0) {
            e=errno;
            if (e==ENOENT) { ++missing; printf("Runtime: open(%s) -> not found\n",guest); }
            else { compat_last_io_error=GetLastError(); log_io_problem("open",guest,host,e); }
            return -e;
        }
        if (host_fstat(host,&s)) memset(&s,0,sizeof(s));
    }
#else
    int host=open(path,host_flags(flags),mode ? mode : 0644);
    if (host<0) {
        e=errno;
        if (e==ENOENT) { ++missing; printf("Runtime: open(%s) -> not found\n",guest); }
        else if (runtime_verbose()) log_io_problem("open",guest,host,e);
        return -e;
    }
    struct stat s;
    Listing *dir=NULL;
    if (!fstat(host,&s) && S_ISDIR(s.st_mode)) dir=list_directory(path);
#endif
    pthread_mutex_lock(&lock);
    int fd=-1;
    for (int i=3;i<MAX_FILES;++i) if (!files[i].used) { fd=i; break; }
    if (fd<0) { pthread_mutex_unlock(&lock); if (host>=0) close(host); free_listing(dir); if (runtime_verbose()) log_io_problem("open (descriptor table full)",guest,-EMFILE,EMFILE); return -EMFILE; }
    files[fd]=(File){.used=1,.host=host,.dir=dir,.append=(flags&0x8)!=0};
    pthread_mutex_init(&files[fd].io,NULL);
    snprintf(files[fd].path,sizeof(files[fd].path),"%s",guest);
    tlggpu_note_open(guest); /* region trace (TLG_REGION_TRACE=1) */
    ++opens;
    pthread_mutex_unlock(&lock);
    if (runtime_verbose()) {
        static unsigned verbose_opens;
        const unsigned n=__atomic_fetch_add(&verbose_opens,1,__ATOMIC_RELAXED);
        if (n<256 || !(n%1000)) printf("Verbose: file open #%u (%s) -> fd %d, flags %#x, %lld bytes%s\n",n,guest,fd,flags,(long long)s.st_size,dir ? ", directory" : "");
    }
    if (audio_trace() && strstr(guest,"sound/")) printf("Audio trace: open(%s) -> fd %d, %lld bytes\n",guest,fd,(long long)s.st_size);
    const char *mod_trace=getenv("TLG_MOD_TRACE"), *mod_root=getenv("TLG_MODS_DIR");
    if (mod_trace && mod_trace[0]=='1' && mod_root) {
        char actual[PATH_MAX],root[PATH_MAX];
        static unsigned traced;
        if (realpath(path,actual) && realpath(mod_root,root)) {
            size_t n=strlen(root);
            if (!strncmp(actual,root,n) && (actual[n]=='/' || actual[n]=='\\') &&
                __atomic_fetch_add(&traced,1,__ATOMIC_RELAXED)<32)
                printf("Mods: open %s -> %s\n",guest,actual);
        }
    }
    return fd;
}
static int64_t do_close(int fd) {
    if (fd>=0 && fd<3) return 0;
    pthread_mutex_lock(&lock);
    File *f=get(fd);
    if (!f) { pthread_mutex_unlock(&lock); return -EBADF; }
    if (runtime_verbose()) {
        static unsigned verbose_closes;
        const unsigned n=__atomic_fetch_add(&verbose_closes,1,__ATOMIC_RELAXED);
        if (n<256 || !(n%1000)) printf("Verbose: file close #%u (%s) fd %d\n",n,f->path,fd);
    }
    if (f->host>=0) close(f->host);
    free_listing(f->dir);
    pthread_mutex_destroy(&f->io);
    *f=(File){0};
    pthread_mutex_unlock(&lock);
    return 0;
}
static int host_fd(int fd) {
    if (fd>=0 && fd<3) return fd;
    File *f=get(fd);
    return f ? f->host : -1;
}
#ifdef _WIN32
/* The CRT reads and writes at most INT_MAX bytes per call. */
static ssize_t host_read(int h,void *buffer,uint64_t size) { return read(h,buffer,(unsigned)(size>0x7ffff000 ? 0x7ffff000 : size)); }
static ssize_t host_write(int h,const void *buffer,uint64_t size) { return write(h,buffer,(unsigned)(size>0x7ffff000 ? 0x7ffff000 : size)); }
#else
#define host_read read
#define host_write write
#endif
/* Pages of the destination may be write-protected for GPU tracking: the kernel's copy then
 * fails with EFAULT instead of faulting to our handler. A user-mode write to each page first
 * goes through the handler, which unprotects it (and records the upcoming write). */
static void touch_for_write(void *buffer,uint64_t size) {
    if (!size) return;
    uintptr_t p=(uintptr_t)buffer & ~(uintptr_t)4095, end=(uintptr_t)buffer+size;
    for (; p<end; p+=4096) {
        volatile unsigned char *b=(volatile unsigned char *)(p<(uintptr_t)buffer ? (uintptr_t)buffer : p);
        *b=*b;
    }
}
/* A failed or short game read is logged (the first 32): on Windows "The disc or Hard Disk Drive
 * is corrupt" is the game's answer to EIO from a file read, so the log must say why. */
static void log_read_problem(const char *what,int fd,uint64_t size,int64_t offset,ssize_t n,int error) {
    static int logged;
    if (__atomic_fetch_add(&logged,1,__ATOMIC_RELAXED)>=32) return;
#ifdef _WIN32
    printf("Runtime: %s(fd %d, %llu bytes @%lld) %s %lld, errno %d, Win32 error %lu\n",what,fd,(unsigned long long)size,
           (long long)offset,n<0 ? "failed" : "short:",(long long)n,error,compat_last_io_error);
#else
    printf("Runtime: %s(fd %d, %llu bytes @%lld) %s %lld, errno %d\n",what,fd,(unsigned long long)size,(long long)offset,
           n<0 ? "failed" : "short:",(long long)n,error);
#endif
}
/* A failed path operation or write is logged (the first 32), like reads: on Windows the game
 * shows "The disc or Hard Disk Drive is corrupt" when a FIOS operation gets EIO. */
static void log_io_problem(const char *what,const char *detail,int64_t n,int error) {
    static int logged;
    if (__atomic_fetch_add(&logged,1,__ATOMIC_RELAXED)>=32) return;
#ifdef _WIN32
    printf("Runtime: %s(%s) failed %lld, errno %d, Win32 error %lu\n",what,detail,(long long)n,error,compat_last_io_error);
#else
    printf("Runtime: %s(%s) failed %lld, errno %d\n",what,detail,(long long)n,error);
#endif
}
/* Reads `size` bytes (offset<0: at the file position), re-touching the rest of the destination
 * after a short read or EFAULT: with precise readbacks the GPU tracker can protect a page again
 * between touch_for_write and the kernel copy, which then stops early without an error. The
 * game treats a short texture read as fatal (JsTex.cpp ASSERT readSize == sizeAlign.m_size). */
/* The kernel copy into a GPU-protected guest page fails (ERROR_NOACCESS on Windows, EFAULT on
 * Linux) instead of faulting to our handler, and touching the pages first is not enough: the GPU
 * can protect them again before the copy (Wine: 64 retries in a row, then a short read; Linux
 * with precise readbacks: 'gave up after 64 retries' on streaming reads, 2026-10-15). Read into
 * a host buffer instead and copy in user mode, where every fault goes through the handler. */
enum { BOUNCE_BYTES=4u<<20 };
/* TLG_FILE_DIRECT=1: read straight into the guest buffer first (one copy instead of two, which
 * the large streaming reads care about) and fall back to the bounce buffer as soon as the kernel
 * copy stops early (EFAULT: the GPU tracker protected a page again) or comes back short of what
 * was asked for. Default off: the bounce path is the one that survived precise readbacks. */
static int file_direct(void) {
    static int value=-1;
    if (value<0) { const char *e=getenv("TLG_FILE_DIRECT"); value=e && e[0]=='1'; }
    return value;
}
/* Straight into the guest buffer; EFAULT here is the GPU tracker's page protection. The pages
 * are touched first only on the out-of-memory path (as before): with TLG_FILE_DIRECT the touch
 * would turn every read into tracked CPU writes, and an EFAULT moves the rest to the bounce. */
static ssize_t direct_read(int h,unsigned char *at,uint64_t size,int64_t offset,int touch) {
    if (touch) touch_for_write(at,size);
    ssize_t n=offset<0 ? host_read(h,at,size) : pread(h,at,size,offset);
    if (n>0) __atomic_add_fetch(&direct_reads,1,__ATOMIC_RELAXED);
    return n;
}
static ssize_t bounce_read(int h,unsigned char *at,uint64_t size,int64_t offset) {
    static _Thread_local unsigned char *bounce;
    if (!bounce && !(bounce=malloc(BOUNCE_BYTES))) return -2;   /* no buffer: read directly */
    __atomic_add_fetch(&bounce_reads,1,__ATOMIC_RELAXED);
    uint64_t want=size<BOUNCE_BYTES ? size : BOUNCE_BYTES;
    ssize_t n=offset<0 ? host_read(h,bounce,want) : pread(h,bounce,want,offset);
    /* The GPU library may write it itself or prepare the range (renderer v3: one invalidation
       instead of a fault per page); else the copy below. */
    if (n>0 && !tlggpu_cpu_write(at,bounce,-1,(uint64_t)n)) memcpy(at,bounce,(size_t)n);
    return n;
}
static ssize_t full_io(int h,void *buffer,uint64_t size,int64_t offset) {
    unsigned char *at=buffer;
    uint64_t done=0;
    int stalls=0;
    int use_bounce=!file_direct();
    for (; done<size && stalls<64;) {
        ssize_t n=use_bounce ? bounce_read(h,at+done,size-done,offset<0 ? -1 : offset+(int64_t)done)
                             : direct_read(h,at+done,size-done,offset<0 ? -1 : offset+(int64_t)done,0);
        if (n==-2) {   /* no bounce buffer (out of memory): read straight into the guest */
            n=direct_read(h,at+done,size-done,offset<0 ? -1 : offset+(int64_t)done,1);
        } else if (!use_bounce) {
            if (n>=0 && (uint64_t)n<size-done) use_bounce=1;      /* short read: bounce the rest */
            else if (n<0 && errno==EFAULT) { use_bounce=1; continue; } /* protected page */
        }
        if (n<0) {
            if (errno==EINTR || errno==EFAULT) { ++stalls; continue; }
            if (done) log_read_problem("io (error after a partial read)",h,size,offset,(ssize_t)done,errno);
            return done ? (ssize_t)done : -1;
        }
        if (n==0) break; /* end of file */
        done+=(uint64_t)n;
    }
    if (stalls>=64) log_read_problem("io (gave up after 64 retries)",h,size,offset,(ssize_t)done,EFAULT);
    return (ssize_t)done;
}
static int64_t do_read(int fd,void *buffer,uint64_t size) {
    int h=host_fd(fd);
    if (h<0) return -EBADF;
#ifdef _WIN32
    File *f=fd>=3 ? get(fd) : NULL;
    ssize_t n;
    if (f) {
        pthread_mutex_lock(&f->io);
        n=full_io(h,buffer,size,f->offset);
        if (n>0) f->offset+=n;
        pthread_mutex_unlock(&f->io);
    } else n=full_io(h,buffer,size,-1);
#else
    ssize_t n=full_io(h,buffer,size,-1);
#endif
    if (n<0) { int e=errno; log_read_problem("read",fd,size,-1,n,e); return -e; }
    __atomic_add_fetch(&reads,1,__ATOMIC_RELAXED); __atomic_add_fetch(&bytes_read,(uint64_t)n,__ATOMIC_RELAXED);
    return n;
}
static int64_t do_pread(int fd,void *buffer,uint64_t size,int64_t offset) {
    int h=host_fd(fd);
    if (h<0) return -EBADF;
    ssize_t n=full_io(h,buffer,size,offset);
    if (n<0) { int e=errno; log_read_problem("pread",fd,size,offset,n,e); return -e; }
    __atomic_add_fetch(&reads,1,__ATOMIC_RELAXED); __atomic_add_fetch(&bytes_read,(uint64_t)n,__ATOMIC_RELAXED);
    return n;
}
#ifdef _WIN32
/* Windows: WriteFile reading a guest page the GPU tracker has read-protected fails
 * (ERROR_NOACCESS) instead of faulting to our handler. Copy each part to a host buffer in user
 * mode first, where the fault goes through the handler, then write from there. */
static ssize_t bounce_write(int h,const unsigned char *at,uint64_t size,int64_t offset) {
    static _Thread_local unsigned char *bounce;
    if (!bounce && !(bounce=malloc(BOUNCE_BYTES))) return offset<0 ? host_write(h,at,size) : pwrite(h,at,size,offset);
    uint64_t done=0;
    while (done<size) {
        const uint64_t part=size-done<BOUNCE_BYTES ? size-done : BOUNCE_BYTES;
        memcpy(bounce,at+done,(size_t)part);
        ssize_t n=offset<0 ? host_write(h,bounce,part) : pwrite(h,bounce,part,offset+(int64_t)done);
        if (n<0) return done ? (ssize_t)done : -1;
        done+=(uint64_t)n;
        if ((uint64_t)n<part) break;
    }
    return (ssize_t)done;
}
#endif
static int64_t do_write(int fd,const void *buffer,uint64_t size) {
    int h=host_fd(fd);
    if (h<0) return -EBADF;
#ifdef _WIN32
    /* The CRT writes at its own file pointer: put it at the descriptor's position first. */
    File *f=fd>=3 ? get(fd) : NULL;
    ssize_t n;
    if (f) {
        pthread_mutex_lock(&f->io);
        if (f->append || host_lseek(h,f->offset,SEEK_SET)>=0) {
            n=bounce_write(h,buffer,size,-1);
            const int64_t at=host_lseek(h,0,SEEK_CUR);
            if (at>=0) f->offset=at;
        } else n=-1;
        pthread_mutex_unlock(&f->io);
    } else n=host_write(h,buffer,size);
#else
    ssize_t n=host_write(h,buffer,size);
#endif
    if (n<0) {
        int e=errno; char what[64];
        snprintf(what,sizeof(what),"fd %d, %llu bytes",fd,(unsigned long long)size);
        log_io_problem("write",what,n,e);
        return -e;
    }
    __atomic_add_fetch(&writes,1,__ATOMIC_RELAXED);
    return n;
}
static int64_t do_pwrite(int fd,const void *buffer,uint64_t size,int64_t offset) {
    int h=host_fd(fd);
    if (h<0) return -EBADF;
#ifdef _WIN32
    ssize_t n=bounce_write(h,buffer,size,offset);
#else
    ssize_t n=pwrite(h,buffer,size,offset);
#endif
    if (n<0) {
        int e=errno; char what[80];
        snprintf(what,sizeof(what),"fd %d, %llu bytes @%lld",fd,(unsigned long long)size,(long long)offset);
        log_io_problem("pwrite",what,n,e);
        return -e;
    }
    return n;
}
static int64_t do_lseek(int fd,int64_t offset,int whence) {
    File *f=get(fd);
    if (!f) return -EBADF;
    if (whence<0 || whence>2) return -EINVAL;
    if (f->dir) {
        /* Directory offsets are entry indices for getdirentries. */
        if (whence==0 && offset>=0) { f->position=(size_t)offset; return offset; }
        return -EINVAL;
    }
#ifdef _WIN32
    pthread_mutex_lock(&f->io);
    int64_t base=0;
    if (whence==1) base=f->offset;
    else if (whence==2) {
        HostStat s;
        if (host_fstat(f->host,&s)) { pthread_mutex_unlock(&f->io); return -errno; }
        base=(int64_t)s.st_size;
    }
    const int64_t r=base+offset;
    if (r<0) { pthread_mutex_unlock(&f->io); return -EINVAL; }
    f->offset=r;
    host_lseek(f->host,r,SEEK_SET); /* the CRT pointer follows, for anything that still uses it */
    pthread_mutex_unlock(&f->io);
    return r;
#else
    int64_t r=host_lseek(f->host,offset,whence);
    return r<0 ? -errno : r;
#endif
}
static int64_t do_fstat(int fd,GuestStat *out) {
    int h=host_fd(fd);
    HostStat s;
#ifdef _WIN32
    if (h<0) { /* directory: a listing only, stat the path it was opened with */
        File *f=get(fd); char path[1024];
        if (!f || !f->dir) return -EBADF;
        if (!out) return -EFAULT;
        if (translate(f->path,path,sizeof(path)) || host_stat(path,&s)) return -EBADF;
        convert_stat(&s,out); return 0;
    }
#else
    if (h<0) return -EBADF;
#endif
    if (!out) return -EFAULT;
    if (host_fstat(h,&s)) return -errno;
    convert_stat(&s,out); return 0;
}
static int64_t do_stat(const char *guest,GuestStat *out) {
    char path[1024]; HostStat s;
    int e=translate(guest,path,sizeof(path));
    if (e) return -e;
    if (!out) return -EFAULT;
    if (host_stat(path,&s)) return -errno;
    convert_stat(&s,out); return 0;
}
static int64_t do_getdents(int fd,char *buffer,uint64_t size,int64_t *basep) {
    pthread_mutex_lock(&lock);
    File *f=get(fd);
    int64_t result=0;
    if (!f) result=-EBADF;
    else if (!f->dir) result=-EINVAL;
    else if (!buffer) result=-EFAULT;
    else if (size<512) result=-EINVAL;
    else {
        if (basep) *basep=(int64_t)f->position;
        uint64_t written=0;
        while (f->position<f->dir->count) {
            const char *name=f->dir->names+f->dir->offsets[f->position];
            size_t n=strlen(name); if (n>255) n=255;
            uint16_t reclen=(uint16_t)((8+n+1+7)&~(size_t)7);
            if (written+reclen>size) break;
            char *p=buffer+written;
            memset(p,0,reclen);
            uint32_t ino=(uint32_t)f->position+1;
            memcpy(p,&ino,4); memcpy(p+4,&reclen,2);
            p[6]=(char)f->dir->types[f->position]; p[7]=(char)n;
            memcpy(p+8,name,n);
            written+=reclen; ++f->position;
        }
        result=(int64_t)written;
    }
    pthread_mutex_unlock(&lock);
    return result;
}
static int64_t path_op(const char *guest,int op,int mode) {
    if (game_path(guest)) return -EROFS;
    char path[1024];
    int e=translate(guest,path,sizeof(path));
    if (e) return -e;
    int r= op==0 ? mkdir(path,mode ? mode : 0755) : op==1 ? rmdir(path) : unlink(path);
    if (r) {
        int e=errno;
#ifdef _WIN32
        compat_last_io_error=GetLastError();
#endif
        if (e!=ENOENT && e!=EEXIST) log_io_problem(op==0 ? "mkdir" : op==1 ? "rmdir" : "unlink",guest,r,e);
        return -e;
    }
    return 0;
}
static int64_t do_rename(const char *from,const char *to) {
    if (game_path(from) || game_path(to)) return -EROFS;
    char a[1024],b[1024];
    int e=translate(from,a,sizeof(a));
    if (!e) e=translate(to,b,sizeof(b));
    if (e) return -e;
#ifdef _WIN32
    /* POSIX rename replaces an existing target; the CRT's fails. */
    if (MoveFileExA(a,b,MOVEFILE_REPLACE_EXISTING)) return 0;
    compat_last_io_error=GetLastError();
    e=compat_errno_from_win32(compat_last_io_error);
    char what[600];
    snprintf(what,sizeof(what),"%s -> %s",from,to);
    log_io_problem("rename",what,-1,e);
    return -e;
#else
    return rename(a,b) ? -errno : 0;
#endif
}
static int64_t do_ftruncate(int fd,int64_t length) {
    int h=host_fd(fd);
    if (h<0) return -EBADF;
#ifdef _WIN32
    return _chsize_s(h,length) ? -errno : 0;
#else
    return ftruncate(h,length) ? -errno : 0;
#endif
}
static int64_t do_truncate(const char *guest,int64_t length) {
    if (game_path(guest)) return -EROFS;
    char path[1024];
    int e=translate(guest,path,sizeof(path));
    if (e) return -e;
#ifdef _WIN32
    int h=open(path,O_RDWR|O_BINARY);
    if (h<0) return -errno;
    int r=_chsize_s(h,length) ? -errno : 0;
    close(h);
    return r;
#else
    return truncate(path,length) ? -errno : 0;
#endif
}
static int64_t do_fsync(int fd) { int h=host_fd(fd); if (h<0) return -EBADF; return fsync(h) ? -errno : 0; }
static int64_t do_access(const char *guest,int mode) {
    char path[1024];
    int e=translate(guest,path,sizeof(path));
    if (e) return -e;
#ifdef _WIN32
    return access(path,mode&6) ? -errno : 0; /* X_OK is invalid for the CRT */
#else
    return access(path,mode&7) ? -errno : 0;
#endif
}

/* Convention adapters: sceKernel* -> Orbis error codes, POSIX -> -1 + errno. */
static int64_t sce(int64_t r) { return r<0 ? ERR(runtime_guest_errno((int)-r)) : r; }
static int64_t posix(int64_t r) { if (r<0) { *runtime_errno()=runtime_guest_errno((int)-r); return -1; } return r; }
#define PAIR(name,params,args) \
    static ABI int64_t sce_##name params { return sce(do_##name args); } \
    static ABI int64_t posix_##name params { return posix(do_##name args); }
/* Calls that may block on the disk are marked for the freeze report (object: the descriptor
 * or the path, name: the path). */
static const char *fd_path(int fd) { File *f=get(fd); return f ? f->path : NULL; }
#define PAIR_IO(name,params,args,kind,object,path) \
    static ABI int64_t sce_##name params { \
        runtime_wait_begin(kind,(const void *)(uintptr_t)(object),path,NULL,RUNTIME_GUEST_FRAME()); \
        int64_t r=sce(do_##name args); runtime_wait_end(); return r; } \
    static ABI int64_t posix_##name params { \
        runtime_wait_begin(kind,(const void *)(uintptr_t)(object),path,NULL,RUNTIME_GUEST_FRAME()); \
        int64_t r=posix(do_##name args); runtime_wait_end(); return r; }
PAIR_IO(open,(const char *p,int f,int m),(p,f,m),"file open",p,p)
PAIR(close,(int fd),(fd))
PAIR_IO(read,(int fd,void *b,uint64_t n),(fd,b,n),"file read",fd,fd_path(fd))
PAIR_IO(pread,(int fd,void *b,uint64_t n,int64_t o),(fd,b,n,o),"file read",fd,fd_path(fd))
PAIR_IO(write,(int fd,const void *b,uint64_t n),(fd,b,n),"file write",fd,fd_path(fd))
PAIR_IO(pwrite,(int fd,const void *b,uint64_t n,int64_t o),(fd,b,n,o),"file write",fd,fd_path(fd))
PAIR(lseek,(int fd,int64_t o,int w),(fd,o,w))
PAIR(fstat,(int fd,GuestStat *s),(fd,s))
PAIR(stat,(const char *p,GuestStat *s),(p,s))
PAIR(getdents,(int fd,char *b,uint64_t n,int64_t *base),(fd,b,n,base))
PAIR(rename,(const char *a,const char *b),(a,b))
PAIR(ftruncate,(int fd,int64_t l),(fd,l))
PAIR(truncate,(const char *p,int64_t l),(p,l))
PAIR_IO(fsync,(int fd),(fd),"file sync",fd,fd_path(fd))
static ABI int64_t posix_access(const char *p,int m) { return posix(do_access(p,m)); }
static ABI int64_t sce_mkdir(const char *p,int m) { return sce(path_op(p,0,m)); }
static ABI int64_t posix_mkdir(const char *p,int m) { return posix(path_op(p,0,m)); }
static ABI int64_t sce_rmdir(const char *p) { return sce(path_op(p,1,0)); }
static ABI int64_t posix_rmdir(const char *p) { return posix(path_op(p,1,0)); }
static ABI int64_t sce_unlink(const char *p) { return sce(path_op(p,2,0)); }
static ABI int64_t posix_unlink(const char *p) { return posix(path_op(p,2,0)); }
static ABI int32_t sce_check_reachability(const char *p) {
    GuestStat s; return (int32_t)sce(do_stat(p,&s));
}

static const RuntimeExport exports[]={
    {"sceKernelOpen",sce_open}, {"open",posix_open}, {"_open",posix_open},
    {"sceKernelClose",sce_close}, {"close",posix_close}, {"_close",posix_close},
    {"sceKernelRead",sce_read}, {"read",posix_read}, {"_read",posix_read},
    {"sceKernelPread",sce_pread}, {"pread",posix_pread},
    {"sceKernelWrite",sce_write}, {"write",posix_write}, {"_write",posix_write},
    {"sceKernelPwrite",sce_pwrite}, {"pwrite",posix_pwrite},
    {"sceKernelLseek",sce_lseek}, {"lseek",posix_lseek},
    {"sceKernelFstat",sce_fstat}, {"fstat",posix_fstat},
    {"sceKernelStat",sce_stat}, {"stat",posix_stat},
    {"sceKernelGetdirentries",sce_getdents}, {"getdirentries",posix_getdents},
    {"sceKernelRename",sce_rename}, {"rename",posix_rename},
    {"sceKernelFtruncate",sce_ftruncate}, {"ftruncate",posix_ftruncate},
    {"sceKernelTruncate",sce_truncate}, {"truncate",posix_truncate},
    {"sceKernelFsync",sce_fsync}, {"fsync",posix_fsync},
    {"access",posix_access}, {"sceKernelCheckReachability",sce_check_reachability},
    {"sceKernelMkdir",sce_mkdir}, {"mkdir",posix_mkdir},
    {"sceKernelRmdir",sce_rmdir}, {"rmdir",posix_rmdir},
    {"sceKernelUnlink",sce_unlink}, {"unlink",posix_unlink},
};
uintptr_t runtime_file_resolve(const char *name) { return RUNTIME_LOOKUP(exports,name); }
/* Host-side helpers for other modules (e.g. SaveData, Fios). */
int64_t runtime_file_open(const char *p,int f,int m) { return do_open(p,f,m); }
int64_t runtime_file_close(int fd) { return do_close(fd); }
int64_t runtime_file_read(int fd,void *b,uint64_t n) { return do_read(fd,b,n); }
int64_t runtime_file_pread(int fd,void *b,uint64_t n,int64_t o) { return do_pread(fd,b,n,o); }
int64_t runtime_file_write(int fd,const void *b,uint64_t n) { return do_write(fd,b,n); }
int64_t runtime_file_lseek(int fd,int64_t o,int w) { return do_lseek(fd,o,w); }
int64_t runtime_file_stat(const char *p,void *s) { return do_stat(p,s); }
int64_t runtime_file_fstat(int fd,void *s) { return do_fstat(fd,s); }
int64_t runtime_file_getdents(int fd,char *b,uint64_t n,int64_t *base) { return do_getdents(fd,b,n,base); }
int runtime_file_translate(const char *guest,char *out,size_t size) { return translate(guest,out,size); }
void runtime_file_report(void) {
    printf("Runtime: files opened=%zu, reads=%zu (%llu bytes; %zu chunks via the bounce buffer, %zu direct), writes=%zu, not found=%zu\n",
           opens,reads,(unsigned long long)bytes_read,bounce_reads,direct_reads,writes,missing);
}

