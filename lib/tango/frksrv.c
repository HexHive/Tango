#include "common.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <linux/limits.h>

extern uint8_t *edge_cnt;
extern size_t edge_sz;
extern pid_t __wrap_fork() __attribute__((weak));
extern pid_t __real_fork() __attribute__((weak));


__attribute__((no_sanitize("coverage")))
static inline void __reset_cov_map() {
    for (int i = 0; i < edge_sz; ++i)
        edge_cnt[i] = 0;
}

__attribute__((used, no_sanitize("coverage")))
static void _forkserver() {
    int fifofd = -1;
    // const char *wd = getenv("TANGO_WORKDIR");
    const char *wd = "./workdir";
    char *fifopath = (char *)malloc(PATH_MAX);
    snprintf(fifopath, PATH_MAX, "%s/%s", wd, "input.pipe");
    fputs("AT LEAST I'M HERE OKAY\n", stderr);fflush(stderr);

    while(1) {
        __reset_cov_map();
        int child_pid;
        if (__wrap_fork)
            child_pid = __real_fork();
        else
            child_pid = fork();
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
            dup2(fifofd, STDIN_FILENO);
            close(fifofd);
            break;
        }
    }
}

__attribute__((naked, used, no_sanitize("coverage")))
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