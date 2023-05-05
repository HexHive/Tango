#include "common.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/prctl.h>

int __wrap_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
int __real_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);

pid_t __wrap_fork();
pid_t __real_fork();

long int __wrap_random(void);

static struct linger no_linger = {0};
static uint32_t reuse = 1;

/* Refer to this SO answer on the nitty-gritty about TIME_WAIT
 * https://stackoverflow.com/a/14388707
 */
ATTRIBUTE_NO_SANITIZE_ALL
int __wrap_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    setsockopt(sockfd, SOL_SOCKET, SO_LINGER, &no_linger, sizeof(no_linger));
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse));
    fprintf(stderr, "Enable address-port reuse on socket: %d\n", sockfd);

    /* go to bind then */
    return __real_bind(sockfd, addr, addrlen);
}

ATTRIBUTE_NO_SANITIZE_ALL
pid_t __wrap_fork() {
    pid_t ppid = getpid();
    pid_t child_pid = __real_fork();
    if (child_pid == 0) {
        prctl(PR_SET_PDEATHSIG, SIGTERM);
        if (getppid() != ppid)
            exit(-256);
    }
    return child_pid;
}

ATTRIBUTE_NO_SANITIZE_ALL
long int __wrap_random(void) {
    static long int x = 0;
    return ++x;
}