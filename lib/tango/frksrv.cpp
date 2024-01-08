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

#define MAX_FD 1024
static off_t file_offset[MAX_FD];

__attribute__((used))
ATTRIBUTE_NO_SANITIZE_ALL
static void _forkserver() {
    int fifofd = -1;
    const char *shared = getenv("TANGO_SHAREDDIR");
    char fifopath[PATH_MAX];
    snprintf(fifopath, PATH_MAX, "%s/%s", shared, "input.pipe");

    // As parent's and child's file descriptors point to the same file
    // descriptions (which stores file status and file offset) in the system
    // wide, if a child changes the file offset, the parent shall recovere it.
    // https://stackoverflow.com/questions/33899548/file-pointers-after-returning-from-a-forked-child-process
    // https://pubs.opengroup.org/onlinepubs/009695399/functions/xsh_chap02_05.html#tag_02_05_01
    int pid = getpid(), dfd;
    char ppath[256];
    sprintf(ppath, "/proc/%d/fd/", pid);
    DIR *dir = opendir(ppath);
    if (!dir) { perror("Failed to open fd directory"); return; }
    dfd = dirfd(dir);

    struct dirent *entry;
    int fd;
    off_t pos;
    struct stat sb;
    for (fd = 0; fd < MAX_FD; fd++) file_offset[fd] = (off_t)-1;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type != DT_LNK) continue;
        fd = atoi(entry->d_name);
        if (fd < 3) continue;
        if (fd == dfd) continue;
        pos = lseek(fd, 0, SEEK_CUR);
        if (pos != (off_t)-1) file_offset[fd] = pos;
    }
    closedir(dir);

    while(1) {
        cleanup_fs();
        CoverageTracer.ClearMaps();
        int child_pid = fork();
        if (child_pid) {
            asm("int $3"); // trap and wait until fuzzer wakes us up
            int status, ret;
            //waitpid(child_pid, &status, 0);
            do {
                ret = waitpid(-1, &status, WNOHANG);
            } while (ret > 0);
            // recover file_offset
            for (fd = 0; fd < MAX_FD; fd++) {
                pos = file_offset[fd];
                if (pos != (off_t)-1) {
                    lseek(fd, pos, SEEK_SET);
                }
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
