#if !__has_feature(coverage_sanitizer)
#error Incompatible compiler! Please use Clang 13.0 or higher
#endif
#define _XOPEN_SOURCE 500

#include "common.h"
#include "tracer.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <ftw.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <linux/limits.h>
#include <dirent.h>
#include <inttypes.h>
#include <sys/queue.h>
#include <stdio_ext.h>

extern "C" {

extern pid_t __wrap_fork() __attribute__((weak));
extern pid_t __real_fork() __attribute__((weak));

static char *upperdir;
static size_t prefixlen;

ATTRIBUTE_NO_SANITIZE_ALL
static int rm_helper(
        const char *fpath, const struct stat *sb,
        int typeflag, struct FTW *ftwbuf) {
    if (ftwbuf->level == 0) return 0;
    /* silently ignore errors in removing paths in merged fs;
     * even though we're doing DFS, we're only traversing the upperdir.
     * an empty dir in the upper fs does not imply the dir is empty in merged.
     * attempting to delete the dir in merged would fail if it's non-empty.
     */
    struct stat wh;
    if (remove(&fpath[prefixlen]));
    else if (lstat(fpath, &wh) == 0 && S_ISCHR(wh.st_mode) && wh.st_rdev == 0)
        if (remove(fpath)) {
            fprintf(stderr, "remove(%s): ", fpath);
            perror("Failed to remove whiteout");
        }
    return 0; // ignore any errors (e.g. permission denied)
}

ATTRIBUTE_NO_SANITIZE_ALL
static int cleanup_directory(const char *dir) {
    int r = nftw(dir, rm_helper, 64, FTW_DEPTH | FTW_PHYS);
    if (r) perror(dir);
    return r;
}

ATTRIBUTE_NO_SANITIZE_ALL
static void cleanup_fs() {
    // we use a second variable so as not to repeat work when TANGO_UPPERDIR is
    // not set
    static bool done = false;
    if (!done) {
        upperdir = getenv("TANGO_FS_UPPERDIR");
        if (upperdir) {
            prefixlen = strlen(upperdir);
        }
        done = true;
    }
    if (!upperdir) return;
    int r = cleanup_directory(upperdir);
    if (r) perror("Failed to clean up tmpfs");
}

// As parent's and child's file descriptors point to the same file
// descriptions (which stores file status and file offset) in the system
// wide, if a child changes the file offset, the parent shall recovere it.
// https://stackoverflow.com/questions/33899548/file-pointers-after-returning-from-a-forked-child-process
// https://pubs.opengroup.org/onlinepubs/009695399/functions/xsh_chap02_05.html#tag_02_05_01
#define MAX_FD 1024
typedef struct file_snapshot {
    off_t file_offset;
    off_t file_size;
} file_snapshot;
static file_snapshot file_snapshots[MAX_FD];

ATTRIBUTE_NO_SANITIZE_ALL
static bool save_file_offsets(void) {
    int pid = getpid(), dfd;
    char ppath[256];
    sprintf(ppath, "/proc/%d/fd/", pid);
    DIR *dir = opendir(ppath);
    if (!dir) { perror("Failed to open fd directory"); return false; }
    dfd = dirfd(dir);

    struct dirent *entry;
    int fd;
    off_t pos;
    struct stat sb;
    for (fd = 0; fd < MAX_FD; fd++) {
        file_snapshots[fd].file_offset = (off_t)-1;
        file_snapshots[fd].file_size = (off_t)-1;
    }
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type != DT_LNK) continue;
        fd = atoi(entry->d_name);
        if (fd < 3) continue;
        if (fd == dfd) continue;
        pos = lseek(fd, 0, SEEK_CUR);
        if (pos != (off_t)-1) {
            file_snapshots[fd].file_offset = pos;
        }
        fstat(fd, &sb);
        if (sb.st_size != (off_t)-1) {
            file_snapshots[fd].file_size = sb.st_size;
        }
    }
    closedir(dir);
    return true;
}

ATTRIBUTE_NO_SANITIZE_ALL
static void restore_file_offsets(void) {
    int fd;
    off_t pos, size;
    for (fd = 0; fd < MAX_FD; fd++) {
        pos = file_snapshots[fd].file_offset;
        if (pos != (off_t)-1) {
            lseek(fd, pos, SEEK_SET);
        }
        size = file_snapshots[fd].file_size;
        if (size != (off_t)-1) {
            ftruncate(fd, size);
        }
    }
}

typedef struct page {
    void *va;
    size_t sz;
    uint8_t data[0x1000];
    TAILQ_ENTRY(page) pages;
} Page;

TAILQ_HEAD(pagehead, page);
typedef struct pagehead pagehead;
pagehead *map_shared_pagehead;

ATTRIBUTE_NO_SANITIZE_ALL
static bool get_map_shared_pages(FILE *maps) {
    char *line = NULL;
    size_t linesz;
    ssize_t len;
    while ((len = getline(&line, &linesz, maps)) > 0) {
        if (line[len - 1] == '\n') {
            line[len - 1] = '\0';
        }
        uintptr_t start, end;
        unsigned int dmajor, dminor;
        char perm[4];
        uint64_t ino;
        int nread = -1;
        // 60de1d944000-60de1d946000 r--p 00000000 103:03 14155924 /usr/bin/cat
        if (sscanf(line, "%" PRIx64 "-%" PRIx64 " %s %*" PRIx64
                         " %x:%x %" PRIu64 " %n",
                &start, &end, perm, &dmajor, &dminor, &ino, &nread) < 6 || nread <= 0) {
            free(line);
            return false;
        }

        char *file = line + nread + strspn(line + nread, " \t");
        if (file[0] != '/' || (ino == 0 && dmajor == 0 && dminor == 0)) {
            /* This line doesn't indicate a file mapping.  */
            continue;
        }

        if (perm[3] == 's' && (strstr(file, "/dev/shm/tango_") == NULL)) {
            for (uintptr_t i = start; i < end; i += 0x1000) {
                struct page *map_shared_page = (page *)calloc(1, sizeof(page));
                map_shared_page->va = (void *)i;
                if (i + 0x1000 > end) {
                    map_shared_page->sz = i - end;
                } else {
                    map_shared_page->sz = 0x1000;
                }
                TAILQ_INSERT_TAIL(map_shared_pagehead, map_shared_page, pages);
            }
        }
    }
    free(line);
    return true;
}

ATTRIBUTE_NO_SANITIZE_ALL
static bool save_map_shared_pages(void) {
    // We also need to handle parent's MAP_SHARED memory.
    // Ref: https://chromium.googlesource.com/external/elfutils/+/refs/heads/master/libdwfl/linux-proc-maps.c
    FILE *maps = fopen("/proc/self/maps", "r");
    if (maps == NULL) { perror("Failed to open maps"); return false; }
    (void)__fsetlocking(maps, FSETLOCKING_BYCALLER);

    map_shared_pagehead = (pagehead *)calloc(1, sizeof(pagehead));
    TAILQ_INIT(map_shared_pagehead);
    get_map_shared_pages(maps);

    page *page;
    TAILQ_FOREACH(page, map_shared_pagehead, pages) {
        memcpy(page->data, page->va, page->sz);
    }

    fclose(maps);
    return true;
}

#define PAGEMAP_LENGTH 8

ATTRIBUTE_NO_SANITIZE_ALL
static bool is_dirty(uint64_t va) {
    if (geteuid() != 0) return true;

    FILE *pm = fopen("/proc/self/pagemap", "rb");
    if (pm == NULL) { perror("Failed to open pagemap"); return 0; }

    uint64_t pfn;
    if (fseek(pm, (va / getpagesize()) * PAGEMAP_LENGTH, SEEK_SET) != 0) {
        perror("Failed to seek pagemap");
        return 0;
    }
    if (fread(&pfn, PAGEMAP_LENGTH, 1, pm) != PAGEMAP_LENGTH) {
        perror("Failed to read pagemap");
        return 0;
    }
    fclose(pm);

    return pfn & (1ULL << 55);
}

ATTRIBUTE_NO_SANITIZE_ALL
static void clear_refs_write(void) {
    if (geteuid() != 0) return;

    FILE *cr = fopen("/proc/self/clear_refs", "r+");
    if (cr == NULL) { perror("Failed to open clear_refs"); return;}
    fwrite("4", 2, 1, cr);
    fclose(cr);
}

ATTRIBUTE_NO_SANITIZE_ALL
static void restore_map_shared_pages(void) {
    // Use soft-dirtypages to not copy all pages
    // Ref: https://stackoverflow.com/questions/12110212/mmaped-file-determine-which-page-is-dirty
    // Ref: https://www.kernel.org/doc/html/latest/admin-guide/mm/soft-dirty.html
    // Ref: https://lore.kernel.org/linux-mm/20220721183338.27871-2-peterx@redhat.com/
    page *page;
    TAILQ_FOREACH(page, map_shared_pagehead, pages) {
        if (is_dirty((uintptr_t)page->va)) {
            memcpy(page->va, page->data, page->sz);
        }
    }
}

__attribute__((used))
ATTRIBUTE_NO_SANITIZE_ALL
static void _forkserver() {
    int fifofd = -1;
    const char *shared = getenv("TANGO_SHAREDDIR");
    char fifopath[PATH_MAX];
    snprintf(fifopath, PATH_MAX, "%s/%s", shared, "input.pipe");

    save_file_offsets();
    if (getenv("SKIP_SHARED_PAGE_CHECK")) {
        save_map_shared_pages();
    }

    while(1) {
        cleanup_fs();
        CoverageTracer.ClearMaps();
        if (getenv("SKIP_SHARED_PAGE_CHECK")) {
            clear_refs_write();
        }
        int child_pid = fork();
        if (child_pid) {
            asm("int $3"); // trap and wait until fuzzer wakes us up
            int status, ret;
            //waitpid(child_pid, &status, 0);
            do {
                ret = waitpid(-1, &status, WNOHANG);
            } while (ret > 0);
            restore_file_offsets();
            if (getenv("SKIP_SHARED_PAGE_CHECK")) {
                restore_map_shared_pages();
            }
        } else {
            fifofd = open(fifopath, O_RDONLY);
            if (fifofd >= 0) {
                dup2(fifofd, STDIN_FILENO);
                close(fifofd);
            }
            break;
        }
    }
}

__attribute__((naked, used))
ATTRIBUTE_NO_SANITIZE_ALL
void forkserver() {
    asm volatile (
        "push %%rax\n"
        "push %%rcx\n"
        "push %%rdx\n"
        "push %%rsi\n"
        "push %%rdi\n"
        "push %%r8\n"
        "push %%r9\n"
        "push %%r10\n"
        "push %%r11\n"
        "push %%rsp\n"
        "call _forkserver\n"
        "pop %%rsp\n"
        "pop %%r11\n"
        "pop %%r10\n"
        "pop %%r9\n"
        "pop %%r8\n"
        "pop %%rdi\n"
        "pop %%rsi\n"
        "pop %%rdx\n"
        "pop %%rcx\n"
        "pop %%rax\n"
        "ret"
        : /* No outputs */
        : /* No inputs */
        : /* No clobbers */
    );
}

} // extern "C"
