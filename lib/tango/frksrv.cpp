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

extern "C" {

extern pid_t __wrap_fork() __attribute__((weak));
extern pid_t __real_fork() __attribute__((weak));

ATTRIBUTE_NO_SANITIZE_ALL
static int rm_helper(
        const char *fpath, const struct stat *sb,
        int typeflag, struct FTW *ftwbuf) {
    if (ftwbuf->level == 0) return 0;
    int r = remove(fpath);
    if (r) perror(fpath);
    return r;
}

ATTRIBUTE_NO_SANITIZE_ALL
static int cleanup_directory(const char *dir) {
    int r = nftw(dir, rm_helper, 64, FTW_DEPTH | FTW_PHYS);
    if (r) perror(dir);
    return r;
}


ATTRIBUTE_NO_SANITIZE_ALL
static void cleanup_fs() {
    static bool done = false;
    static const char *upperdir = NULL;
    if (!done) {
        upperdir = getenv("TANGO_UPPERDIR");
        done = true;
    }
    if (!upperdir) return;
    int r = cleanup_directory(upperdir);
    if (r) perror("Failed to clean up tmpfs");
}

__attribute__((used))
ATTRIBUTE_NO_SANITIZE_ALL
static void _forkserver() {
    int fifofd = -1;
    const char *wd = getenv("TANGO_WORKDIR");
    char *fifopath = (char *)malloc(PATH_MAX);
    snprintf(fifopath, PATH_MAX, "%s/%s", wd, "input.pipe");

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
        } else {
            fifofd = open(fifopath, O_RDONLY);
            free(fifopath);
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
